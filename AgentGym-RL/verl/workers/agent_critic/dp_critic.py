# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement a multiprocess PPOCritic
"""
import itertools
import os

import torch
from torch import nn, optim

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.agent_trainer.ppo import core_algos
from verl.workers.agent_critic import BasePPOCritic
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOCritic']


def _debug_enabled() -> bool:
    return os.environ.get('VERL_IMPROVE_DEBUG_PROGRESS', '0') == '1'


def _debug_all_ranks() -> bool:
    return os.environ.get('VERL_IMPROVE_DEBUG_ALL_RANKS', '0') == '1'


def _debug(message: str) -> None:
    if not _debug_enabled():
        return
    rank = 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    if rank == 0 or _debug_all_ranks():
        print(f"[improve-dp-critic][rank={rank}] {message}", flush=True)


class DataParallelPPOCritic(BasePPOCritic):

    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        super().__init__(config=config)
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.use_remove_padding = self.config.model.get('use_remove_padding', False)
        print(f'Critic use_remove_padding={self.use_remove_padding}')

        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)

    def _forward_micro_batch(self, micro_batch):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.critic_module(input_ids=input_ids_rmpad,
                                            attention_mask=None,
                                            position_ids=position_ids_rmpad,
                                            use_cache=False)  # prevent model thinks we are generating
                values_rmpad = output.logits.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    values_rmpad = gather_outpus_and_unpad(values_rmpad,
                                                           gather_dim=0,
                                                           unpad_dim=0,
                                                           padding_size=pad_size)

                # pad it back
                values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen).squeeze(-1)
            else:
                output = self.critic_module(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids,
                                            use_cache=False)  # prevent model thinks we are generating
                values = output.logits.squeeze(-1)
            return values

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        _debug("_optimizer_step:clip_grad_norm:start")
        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        _debug("_optimizer_step:clip_grad_norm:end")
        _debug("_optimizer_step:optimizer_step:start")
        self.critic_optimizer.step()
        _debug("_optimizer_step:optimizer_step:end")
        return grad_norm

    def compute_values(self, data: DataProto) -> torch.Tensor:
        self.critic_module.eval()
        micro_batch_size = data.meta_info['micro_batch_size']
        select_keys = ['input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        values_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                values = self._forward_micro_batch(micro_batch)
            values_lst.append(values)
        values = torch.concat(values_lst, dim=0)
        attention_mask = data.batch['attention_mask']
        values = values * attention_mask

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == values.size(0), f"{len(indices)} vs. {values.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            values = values[revert_indices]

        return values

    def update_critic(self, data: DataProto):
        # make sure we are in training mode
        self.critic_module.train()
        metrics = {}
        _debug("update_critic:start")

        select_keys = ['input_ids', 'attention_mask', 'position_ids', 'values', 'returns', 'response_mask']
        batch = data.select(batch_keys=select_keys).batch
        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        for batch_idx, data in enumerate(dataloader):
            _debug(f"update_critic:minibatch_start idx={batch_idx} batch_size={len(data)}")
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu

            _debug(f"update_critic:minibatch_split idx={batch_idx} micro_batches={len(micro_batches)}")
            self.critic_optimizer.zero_grad()
            _debug(f"update_critic:zero_grad_done idx={batch_idx}")

            for micro_idx, data in enumerate(micro_batches):
                _debug(f"update_critic:microbatch_start idx={batch_idx}.{micro_idx} batch_size={len(data)}")
                data = data.cuda()  # critic device is cpu when using offload
                values = data['values']
                returns = data['returns']

                eos_mask = data['response_mask']

                _debug(f"update_critic:forward_start idx={batch_idx}.{micro_idx}")
                vpreds = self._forward_micro_batch(data)
                _debug(f"update_critic:forward_end idx={batch_idx}.{micro_idx}")

                if not (vpreds.shape[-1] == values.shape[-1] == returns.shape[-1] == eos_mask.shape[-1]):
                    # Align to the response tail if prompt positions leaked into one of the tensors.
                    target_len = min(vpreds.shape[-1], values.shape[-1], returns.shape[-1], eos_mask.shape[-1])
                    vpreds = vpreds[:, -target_len:]
                    values = values[:, -target_len:]
                    returns = returns[:, -target_len:]
                    eos_mask = eos_mask[:, -target_len:]

                # assert not torch.any(torch.isnan(vpreds)).item()

                vf_loss, vf_clipfrac = core_algos.compute_value_loss(vpreds=vpreds,
                                                                     values=values,
                                                                     returns=returns,
                                                                     eos_mask=eos_mask,
                                                                     cliprange_value=self.config.cliprange_value)
                if self.config.use_dynamic_bsz:
                    # relative to the dynamic bsz
                    loss = vf_loss * (len(data) / self.config.ppo_mini_batch_size)
                else:
                    loss = vf_loss / self.gradient_accumulation

                _debug(f"update_critic:backward_start idx={batch_idx}.{micro_idx}")
                loss.backward()
                _debug(f"update_critic:backward_end idx={batch_idx}.{micro_idx}")

                data = {
                    'critic/vf_loss': vf_loss.detach().item(),
                    'critic/vf_clipfrac': vf_clipfrac.detach().item(),
                    'critic/vpred_mean': masked_mean(vpreds, eos_mask).detach().item(),
                }

                append_to_dict(metrics, data)
                _debug(f"update_critic:microbatch_end idx={batch_idx}.{micro_idx}")

            _debug(f"update_critic:optimizer_step_start idx={batch_idx}")
            grad_norm = self._optimizer_step()
            _debug(f"update_critic:optimizer_step_end idx={batch_idx}")
            data = {'critic/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
            _debug(f"update_critic:minibatch_end idx={batch_idx}")
        self.critic_optimizer.zero_grad()
        _debug("update_critic:end")
        return metrics
