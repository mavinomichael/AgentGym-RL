# Multi-agent extension.
# Derived from: /Users/mavinomichael/PycharmProjects/AgentGym-RL/AgentGym-RL/verl/workers/agent_fsdp_workers.py
# Original file left untouched for comparison.

from __future__ import annotations

import copy
import os
from typing import Dict, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from tensordict import TensorDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig

from verl import DataProto
from verl.extend_multi_agent.role_training import (
    assert_module_has_no_gradients,
    clone_batch_with_role_weights,
    merge_role_log_probs,
    resolve_role_phase,
)
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fsdp_utils import (
    load_fsdp_optimizer,
    load_fsdp_param_and_grad,
    offload_fsdp_grad,
    offload_fsdp_optimizer,
    offload_fsdp_param_and_grad,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.agent_fsdp_workers import ActorRolloutRefWorker as BaseActorRolloutRefWorker
from verl.workers.agent_fsdp_workers import CriticWorker as BaseCriticWorker


def _mean_metric(values) -> float:
    if values is None:
        return 0.0
    if isinstance(values, (list, tuple)):
        if len(values) == 0:
            return 0.0
        return float(np.mean(values))
    return float(values)


class ActorRolloutRefWorker(BaseActorRolloutRefWorker):
    def _extend_cfg(self):
        return self.config.get("extend_multi_agent", None)

    def _resolve_phase(self):
        cfg = self._extend_cfg()
        if cfg is None:
            return resolve_role_phase("joint", False, False)
        return resolve_role_phase(
            train_role=cfg.get("train_role", "joint"),
            freeze_planner=cfg.get("freeze_planner", False),
            freeze_executor=cfg.get("freeze_executor", False),
        )

    def _role_model_path(self, role_name: str, *, ref: bool) -> str:
        cfg = self._extend_cfg()
        default_path = self.config.model.path
        if cfg is None:
            return default_path
        key = f"{role_name}_{'ref_' if ref else ''}model_path"
        candidate = cfg.get(key, None)
        if candidate in (None, ""):
            return default_path
        return candidate

    @staticmethod
    def _set_module_trainable(module, trainable: bool) -> None:
        for param in module.parameters():
            param.requires_grad_(trainable)

    @staticmethod
    def _clone_data_proto_with_role(data: DataProto, role: str) -> DataProto:
        batch_dict = {key: value.clone() for key, value in data.batch.items()}
        role_batch = clone_batch_with_role_weights(batch_dict, role)
        return DataProto(
            batch=TensorDict(source=role_batch, batch_size=data.batch.batch_size),
            non_tensor_batch=copy.deepcopy(data.non_tensor_batch),
            meta_info=copy.deepcopy(data.meta_info),
        )

    def _summarize_role_update_metrics(self, role_name: str, raw_metrics: Dict) -> Dict[str, float]:
        summary = {
            f"{role_name}_loss": _mean_metric(raw_metrics.get("actor/pg_loss", 0.0)),
            f"{role_name}_grad_norm": _mean_metric(raw_metrics.get("actor/grad_norm", 0.0)),
        }
        mapping = {
            "actor/entropy_loss": "entropy_loss",
            "actor/pg_loss": "pg_loss",
            "actor/pg_clipfrac": "pg_clipfrac",
            "actor/ppo_kl": "ppo_kl",
            "actor/grad_norm": "grad_norm",
            "actor/kl_loss": "kl_loss",
            "actor/kl_coef": "kl_coef",
            "actor/planner_kl_weight": "planner_kl_weight",
        }
        for raw_key, short_name in mapping.items():
            if raw_key in raw_metrics:
                summary[f"{role_name}/{short_name}"] = _mean_metric(raw_metrics[raw_key])
        return summary

    def _build_rollout_for_role(self, actor_module_fsdp, actor_model_config):
        from torch.distributed.device_mesh import init_device_mesh
        from verl.extend_multi_agent.workers.rollout.agent_vllm_rollout import vLLMRollout
        from verl.workers.sharding_manager import FSDPVLLMShardingManager

        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        )
        rollout_device_mesh = init_device_mesh(
            "cuda", mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )

        log_gpu_memory_usage("Before building extend_multi_agent vllm rollout", logger=None)
        rollout = vLLMRollout(
            actor_module=actor_module_fsdp,
            rollout_config=self.config.rollout,
            agentgym_config=self.config.agentgym,
            tokenizer=self.tokenizer,
            model_hf_config=actor_model_config,
            multi_agent_config=self.config.get("multi_agent", None),
        )
        log_gpu_memory_usage("After building extend_multi_agent vllm rollout", logger=None)
        if torch.distributed.get_world_size() == 1:
            self.config.rollout.load_format = "dummy_hf"
        rollout_sharding_manager = FSDPVLLMShardingManager(
            module=actor_module_fsdp,
            inference_engine=rollout.inference_engine,
            model_config=actor_model_config,
            full_params="hf" in self.config.rollout.load_format,
            device_mesh=rollout_device_mesh,
        )
        log_gpu_memory_usage("After building extend_multi_agent sharding manager", logger=None)
        return rollout, rollout_sharding_manager

    def _build_hf_role_generator(self, module_fsdp, role_name: str):
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id

        @torch.no_grad()
        def _generate(*, prompt_token_ids, sampling_overrides):
            if not prompt_token_ids:
                return torch.empty((0, 0), device=torch.cuda.current_device(), dtype=torch.long)

            max_prompt_len = max(len(ids) for ids in prompt_token_ids)
            input_rows = []
            attn_rows = []
            for ids in prompt_token_ids:
                pad_len = max_prompt_len - len(ids)
                input_rows.append(([pad_token_id] * pad_len) + list(ids))
                attn_rows.append(([0] * pad_len) + ([1] * len(ids)))

            input_ids = torch.tensor(input_rows, device=torch.cuda.current_device(), dtype=torch.long)
            attention_mask = torch.tensor(attn_rows, device=torch.cuda.current_device(), dtype=torch.long)
            position_ids = compute_position_id_with_mask(attention_mask)

            top_k = int(sampling_overrides.get("top_k", -1))
            if top_k < 0:
                top_k = 0
            temperature = float(sampling_overrides.get("temperature", 1.0))
            top_p = float(sampling_overrides.get("top_p", 1.0))
            do_sample = temperature > 0.0
            generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k)
            max_new_tokens = int(
                sampling_overrides.get("max_tokens", self.config.multi_agent.roles[role_name].max_tokens)
            )

            module_fsdp.eval()
            with FSDP.summon_full_params(module_fsdp, writeback=False, recurse=False):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = module_fsdp.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        do_sample=do_sample,
                        max_new_tokens=max_new_tokens,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                        generation_config=generation_config,
                        output_scores=False,
                        return_dict_in_generate=True,
                        use_cache=True,
                    )
            module_fsdp.train()
            return outputs.sequences[:, input_ids.size(1):]

        return _generate

    def _maybe_load_role_model(self, module_fsdp) -> None:
        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=module_fsdp,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )

    def _maybe_offload_role_model(self, module_fsdp) -> None:
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=module_fsdp, offload_grad=self._is_offload_grad)

    def _maybe_load_role_optimizer(self, optimizer) -> None:
        if self._is_offload_optimizer and optimizer is not None:
            load_fsdp_optimizer(optimizer=optimizer, device_id=torch.cuda.current_device())

    def _maybe_offload_role_optimizer(self, optimizer) -> None:
        if self._is_offload_optimizer and optimizer is not None:
            offload_fsdp_optimizer(optimizer=optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.agent_actor import DataParallelPPOActor

        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(
            self.config.model.get("override_config", OmegaConf.create())
        )
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        self.role_phase = self._resolve_phase()

        if self._is_actor or self._is_rollout:
            optim_config = self.config.actor.optim
            fsdp_config = self.config.actor.fsdp_config

            (
                self.planner_actor_module_fsdp,
                planner_optimizer,
                planner_lr_scheduler,
                self.planner_actor_model_config,
            ) = self._build_model_optimizer(
                model_path=self._role_model_path("planner", ref=False),
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
            )
            (
                self.executor_actor_module_fsdp,
                executor_optimizer,
                executor_lr_scheduler,
                self.executor_actor_model_config,
            ) = self._build_model_optimizer(
                model_path=self._role_model_path("executor", ref=False),
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
            )

            self.planner_actor_module = self.planner_actor_module_fsdp._fsdp_wrapped_module
            self.executor_actor_module = self.executor_actor_module_fsdp._fsdp_wrapped_module

            self._set_module_trainable(self.planner_actor_module_fsdp, self.role_phase.planner_trainable)
            self._set_module_trainable(self.executor_actor_module_fsdp, self.role_phase.executor_trainable)

            self.planner_actor_optimizer = planner_optimizer if self.role_phase.planner_trainable else None
            self.executor_actor_optimizer = executor_optimizer if self.role_phase.executor_trainable else None
            self.planner_actor_lr_scheduler = planner_lr_scheduler if self.role_phase.planner_trainable else None
            self.executor_actor_lr_scheduler = executor_lr_scheduler if self.role_phase.executor_trainable else None

            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding

            self.planner_actor = DataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.planner_actor_module_fsdp,
                actor_optimizer=self.planner_actor_optimizer,
            )
            self.executor_actor = DataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.executor_actor_module_fsdp,
                actor_optimizer=self.executor_actor_optimizer,
            )

            if self._is_offload_param:
                offload_fsdp_grad(module=self.planner_actor_module_fsdp)
                offload_fsdp_grad(module=self.executor_actor_module_fsdp)
                log_gpu_memory_usage("After offloading extend_multi_agent actor grads during init", logger=None)
            if self._is_offload_optimizer:
                self._maybe_offload_role_optimizer(self.planner_actor_optimizer)
                self._maybe_offload_role_optimizer(self.executor_actor_optimizer)
                log_gpu_memory_usage("After offloading extend_multi_agent actor optimizers during init", logger=None)

        if self._is_rollout:
            rollout_runtime = str(self._extend_cfg().get("rollout_runtime", "vllm")).lower()
            self.planner_rollout, self.planner_rollout_sharding_manager = self._build_rollout_for_role(
                self.planner_actor_module_fsdp,
                self.planner_actor_model_config,
            )
            self.rollout = self.planner_rollout
            if rollout_runtime == "hf":
                self.executor_rollout = None
                self.executor_rollout_sharding_manager = None
                self.rollout.bind_role_runtime(
                    "planner",
                    inference_engine=None,
                    generator_fn=self._build_hf_role_generator(self.planner_actor_module_fsdp, "planner"),
                )
                self.rollout.bind_role_runtime(
                    "executor",
                    inference_engine=None,
                    generator_fn=self._build_hf_role_generator(self.executor_actor_module_fsdp, "executor"),
                )
            else:
                self.executor_rollout, self.executor_rollout_sharding_manager = self._build_rollout_for_role(
                    self.executor_actor_module_fsdp,
                    self.executor_actor_model_config,
                )
                self.rollout.bind_role_runtime(
                    "planner",
                    inference_engine=self.planner_rollout.inference_engine,
                    sampling_params=self.planner_rollout.sampling_params,
                    sharding_manager=self.planner_rollout_sharding_manager,
                )
                self.rollout.bind_role_runtime(
                    "executor",
                    inference_engine=self.executor_rollout.inference_engine,
                    sampling_params=self.executor_rollout.sampling_params,
                    sharding_manager=self.executor_rollout_sharding_manager,
                )

        if self._is_ref:
            self.planner_ref_module_fsdp = self._build_model_optimizer(
                model_path=self._role_model_path("planner", ref=True),
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            self.executor_ref_module_fsdp = self._build_model_optimizer(
                model_path=self._role_model_path("executor", ref=True),
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
            self.planner_ref_policy = DataParallelPPOActor(
                config=self.config.ref,
                actor_module=self.planner_ref_module_fsdp,
            )
            self.executor_ref_policy = DataParallelPPOActor(
                config=self.config.ref,
                actor_module=self.executor_ref_module_fsdp,
            )

        if self._is_actor:
            self.planner_flops_counter = FlopsCounter(self.planner_actor_model_config)
            self.executor_flops_counter = FlopsCounter(self.executor_actor_model_config)
            self.planner_checkpoint_manager = FSDPCheckpointManager(
                model=self.planner_actor_module_fsdp,
                optimizer=self.planner_actor_optimizer,
                lr_scheduler=self.planner_actor_lr_scheduler,
                tokenizer=self.tokenizer,
            )
            self.executor_checkpoint_manager = FSDPCheckpointManager(
                model=self.executor_actor_module_fsdp,
                optimizer=self.executor_actor_optimizer,
                lr_scheduler=self.executor_actor_lr_scheduler,
                tokenizer=self.tokenizer,
            )

        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        assert self._is_actor
        data = data.to("cuda")
        data.batch = data.batch.cuda()

        if self.role_phase.planner_trainable:
            self._maybe_load_role_model(self.planner_actor_module_fsdp)
            self._maybe_load_role_optimizer(self.planner_actor_optimizer)
        if self.role_phase.executor_trainable:
            self._maybe_load_role_model(self.executor_actor_module_fsdp)
            self._maybe_load_role_optimizer(self.executor_actor_optimizer)

        metrics: Dict[str, float] = {
            "planner_loss": 0.0,
            "executor_loss": 0.0,
            "planner_grad_norm": 0.0,
            "executor_grad_norm": 0.0,
        }

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            if self.role_phase.planner_trainable:
                planner_batch = self._clone_data_proto_with_role(data, "planner")
                planner_metrics = self.planner_actor.update_policy(data=planner_batch)
                metrics.update(self._summarize_role_update_metrics("planner", planner_metrics))
                if self.planner_actor_lr_scheduler is not None:
                    self.planner_actor_lr_scheduler.step()
                    metrics["planner/lr"] = float(self.planner_actor_lr_scheduler.get_last_lr()[0])
            else:
                assert_module_has_no_gradients(self.planner_actor_module_fsdp)

            if self.role_phase.executor_trainable:
                executor_batch = self._clone_data_proto_with_role(data, "executor")
                executor_metrics = self.executor_actor.update_policy(data=executor_batch)
                metrics.update(self._summarize_role_update_metrics("executor", executor_metrics))
                if self.executor_actor_lr_scheduler is not None:
                    self.executor_actor_lr_scheduler.step()
                    metrics["executor/lr"] = float(self.executor_actor_lr_scheduler.get_last_lr()[0])
            else:
                assert_module_has_no_gradients(self.executor_actor_module_fsdp)

            output = DataProto(meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to("cpu")

        if self.role_phase.planner_trainable:
            self._maybe_offload_role_model(self.planner_actor_module_fsdp)
            self._maybe_offload_role_optimizer(self.planner_actor_optimizer)
        if self.role_phase.executor_trainable:
            self._maybe_offload_role_model(self.executor_actor_module_fsdp)
            self._maybe_offload_role_optimizer(self.executor_actor_optimizer)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        assert self._is_rollout
        prompts = prompts.to("cuda")
        prompts.batch = prompts.batch.cuda()

        self._maybe_load_role_model(self.planner_actor_module_fsdp)
        self._maybe_load_role_model(self.executor_actor_module_fsdp)

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        output = self.rollout.generate_sequences(prompts=prompts)
        output = output.to("cpu")

        self._maybe_offload_role_model(self.planner_actor_module_fsdp)
        self._maybe_offload_role_model(self.executor_actor_module_fsdp)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        self._maybe_load_role_model(self.planner_actor_module_fsdp)
        self._maybe_load_role_model(self.executor_actor_module_fsdp)

        data = data.to("cuda")
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            planner_log_probs = self.planner_actor.compute_log_prob(data=data)
            executor_log_probs = self.executor_actor.compute_log_prob(data=data)
            merged_log_probs = merge_role_log_probs(
                planner_log_probs=planner_log_probs,
                executor_log_probs=executor_log_probs,
                planner_mask=data.batch["planner_response_mask"],
                executor_mask=data.batch["executor_response_mask"],
            )
            output = DataProto.from_dict(
                tensors={
                    "old_log_probs": merged_log_probs,
                    "planner_old_log_probs": planner_log_probs,
                    "executor_old_log_probs": executor_log_probs,
                },
                meta_info={"temperature": self.config.rollout.temperature},
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        if self.world_size > 1:
            self.planner_actor.actor_module._handle.reshard(True)
            self.executor_actor.actor_module._handle.reshard(True)

        self._maybe_offload_role_model(self.planner_actor_module_fsdp)
        self._maybe_offload_role_model(self.executor_actor_module_fsdp)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref
        data = data.to("cuda")

        data.meta_info["micro_batch_size"] = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            planner_log_probs = self.planner_ref_policy.compute_log_prob(data=data)
            executor_log_probs = self.executor_ref_policy.compute_log_prob(data=data)
            merged_log_probs = merge_role_log_probs(
                planner_log_probs=planner_log_probs,
                executor_log_probs=executor_log_probs,
                planner_mask=data.batch["planner_response_mask"],
                executor_mask=data.batch["executor_response_mask"],
            )
            output = DataProto.from_dict(
                tensors={
                    "ref_log_prob": merged_log_probs,
                    "planner_ref_log_prob": planner_log_probs,
                    "executor_ref_log_prob": executor_log_probs,
                }
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        if self.world_size > 1:
            self.planner_ref_policy.actor_module._handle.reshard(True)
            self.executor_ref_policy.actor_module._handle.reshard(True)

        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, remove_previous_ckpt=False):
        assert self._is_actor
        self._maybe_load_role_model(self.planner_actor_module_fsdp)
        self._maybe_load_role_model(self.executor_actor_module_fsdp)
        self.planner_checkpoint_manager.save_checkpoint(
            local_path=os.path.join(local_path, "planner"),
            hdfs_path=None if hdfs_path is None else os.path.join(hdfs_path, "planner"),
            global_step=global_step,
            remove_previous_ckpt=remove_previous_ckpt,
        )
        self.executor_checkpoint_manager.save_checkpoint(
            local_path=os.path.join(local_path, "executor"),
            hdfs_path=None if hdfs_path is None else os.path.join(hdfs_path, "executor"),
            global_step=global_step,
            remove_previous_ckpt=remove_previous_ckpt,
        )
        torch.distributed.barrier()
        self._maybe_offload_role_model(self.planner_actor_module_fsdp)
        self._maybe_offload_role_model(self.executor_actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path, del_local_after_load=False):
        if path is None:
            return
        planner_path = os.path.join(path, "planner")
        executor_path = os.path.join(path, "executor")
        if self._is_actor:
            self._maybe_load_role_model(self.planner_actor_module_fsdp)
            self._maybe_load_role_model(self.executor_actor_module_fsdp)
            self.planner_checkpoint_manager.load_checkpoint(
                path=planner_path,
                del_local_after_load=del_local_after_load,
            )
            self.executor_checkpoint_manager.load_checkpoint(
                path=executor_path,
                del_local_after_load=del_local_after_load,
            )
            self._maybe_offload_role_model(self.planner_actor_module_fsdp)
            self._maybe_offload_role_model(self.executor_actor_module_fsdp)


class CriticWorker(BaseCriticWorker):
    pass
