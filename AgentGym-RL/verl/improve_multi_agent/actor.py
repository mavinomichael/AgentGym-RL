from __future__ import annotations

import itertools
from typing import Dict

import torch

from verl import DataProto
from verl.agent_trainer.ppo import core_algos
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import masked_mean
from verl.workers.agent_actor.dp_actor import DataParallelPPOActor


class RoleAwareDataParallelPPOActor(DataParallelPPOActor):
    """Actor update with role-local PPO clip and KL budgets.

    The underlying backbone is still shared, but planner and executor tokens are
    optimized with different budgets so they are no longer normalized as one
    homogeneous token population.
    """

    def _masked_role_loss(
        self,
        *,
        ratio: torch.Tensor,
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        entropy: torch.Tensor,
        role_mask: torch.Tensor,
        clip_ratio: float,
        entropy_coeff: float,
    ) -> Dict[str, torch.Tensor]:
        if torch.count_nonzero(role_mask).item() == 0:
            zero = torch.zeros((), device=ratio.device, dtype=ratio.dtype)
            return {
                "policy_loss": zero,
                "pg_loss": zero,
                "entropy_loss": zero,
                "pg_clipfrac": zero,
                "ppo_kl": zero,
            }
        weighted_mask = role_mask.float()
        denom = torch.clamp_min(weighted_mask.sum(), 1.0)
        negative_approx_kl = log_prob - old_log_prob
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        pg_loss = (torch.max(pg_losses, pg_losses2) * weighted_mask).sum() / denom
        pg_clipfrac = (torch.gt(pg_losses2, pg_losses).float() * weighted_mask).sum() / denom
        ppo_kl = ((-negative_approx_kl) * weighted_mask).sum() / denom
        entropy_loss = (entropy * weighted_mask).sum() / denom
        policy_loss = pg_loss - entropy_loss * entropy_coeff
        return {
            "policy_loss": policy_loss,
            "pg_loss": pg_loss,
            "entropy_loss": entropy_loss,
            "pg_clipfrac": pg_clipfrac,
            "ppo_kl": ppo_kl,
        }

    def update_policy(self, data: DataProto):
        self.actor_module.train()
        temperature = data.meta_info["temperature"]
        select_keys = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            "responses",
            "response_mask",
        ]
        optional_keys = [
            "planner_response_mask",
            "executor_response_mask",
            "ppo_loss_weights",
            "kl_loss_weights",
            "ref_log_prob",
        ]
        select_keys.extend([key for key in optional_keys if key in data.batch.keys()])
        batch = data.select(batch_keys=select_keys).batch
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        use_role_local_optimization = bool(getattr(self.config, "role_local_optimization", True))
        planner_clip = float(getattr(self.config, "planner_clip_ratio", self.config.clip_ratio))
        executor_clip = float(getattr(self.config, "executor_clip_ratio", self.config.clip_ratio))
        planner_entropy_coeff = float(getattr(self.config, "planner_entropy_coeff", self.config.entropy_coeff))
        executor_entropy_coeff = float(getattr(self.config, "executor_entropy_coeff", self.config.entropy_coeff))
        planner_kl_coef = float(getattr(self.config, "planner_kl_loss_coef", self.config.kl_loss_coef))
        executor_kl_coef = float(getattr(self.config, "executor_kl_loss_coef", self.config.kl_loss_coef))

        for _, mini_batch in enumerate(dataloader):
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                gradient_accumulation = 1
            else:
                gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

            self.actor_optimizer.zero_grad()
            for micro in micro_batches:
                micro = micro.cuda()
                response_mask = micro["response_mask"]
                old_log_prob = micro["old_log_probs"]
                advantages = micro["advantages"]
                entropy, log_prob = self._forward_micro_batch(micro_batch=micro, temperature=temperature)
                negative_approx_kl = log_prob - old_log_prob
                ratio = torch.exp(negative_approx_kl)

                if "ppo_loss_weights" in micro.keys():
                    weighted_mask = response_mask.float() * micro["ppo_loss_weights"].float()
                else:
                    weighted_mask = response_mask.float()

                planner_mask = micro.get("planner_response_mask", torch.zeros_like(response_mask)).float() * weighted_mask
                executor_mask = micro.get("executor_response_mask", torch.zeros_like(response_mask)).float() * weighted_mask
                remaining_mask = torch.clamp(weighted_mask - planner_mask - executor_mask, min=0.0)
                if not use_role_local_optimization:
                    remaining_mask = weighted_mask
                    planner_mask = torch.zeros_like(weighted_mask)
                    executor_mask = torch.zeros_like(weighted_mask)

                planner_terms = self._masked_role_loss(
                    ratio=ratio,
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    advantages=advantages,
                    entropy=entropy,
                    role_mask=planner_mask,
                    clip_ratio=planner_clip,
                    entropy_coeff=planner_entropy_coeff,
                )
                executor_terms = self._masked_role_loss(
                    ratio=ratio,
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    advantages=advantages,
                    entropy=entropy,
                    role_mask=executor_mask,
                    clip_ratio=executor_clip,
                    entropy_coeff=executor_entropy_coeff,
                )

                total_mask_denom = torch.clamp_min((planner_mask + executor_mask + remaining_mask).sum(), 1.0)
                if torch.count_nonzero(remaining_mask).item() > 0:
                    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        eos_mask=remaining_mask,
                        cliprange=self.config.clip_ratio,
                    )
                    entropy_loss = masked_mean(entropy, remaining_mask)
                    other_policy_loss = pg_loss - entropy_loss * self.config.entropy_coeff
                else:
                    zero = torch.zeros((), device=response_mask.device, dtype=log_prob.dtype)
                    pg_loss = zero
                    pg_clipfrac = zero
                    ppo_kl = zero
                    entropy_loss = zero
                    other_policy_loss = zero

                policy_loss = (
                    planner_terms["policy_loss"] * planner_mask.sum()
                    + executor_terms["policy_loss"] * executor_mask.sum()
                    + other_policy_loss * remaining_mask.sum()
                ) / total_mask_denom

                if self.config.use_kl_loss:
                    ref_log_prob = micro["ref_log_prob"]
                    kld = core_algos.kl_penalty(
                        logprob=log_prob,
                        ref_logprob=ref_log_prob,
                        kl_penalty=self.config.kl_loss_type,
                    )
                    planner_kl = (
                        (kld * planner_mask).sum() / torch.clamp_min(planner_mask.sum(), 1.0)
                        if torch.count_nonzero(planner_mask).item() > 0
                        else torch.zeros((), device=kld.device, dtype=kld.dtype)
                    )
                    executor_kl = (
                        (kld * executor_mask).sum() / torch.clamp_min(executor_mask.sum(), 1.0)
                        if torch.count_nonzero(executor_mask).item() > 0
                        else torch.zeros((), device=kld.device, dtype=kld.dtype)
                    )
                    other_kl = (
                        (kld * remaining_mask).sum() / torch.clamp_min(remaining_mask.sum(), 1.0)
                        if torch.count_nonzero(remaining_mask).item() > 0
                        else torch.zeros((), device=kld.device, dtype=kld.dtype)
                    )
                    role_weighted_kl = (
                        planner_kl * planner_mask.sum() * planner_kl_coef
                        + executor_kl * executor_mask.sum() * executor_kl_coef
                        + other_kl * remaining_mask.sum() * self.config.kl_loss_coef
                    ) / total_mask_denom
                    policy_loss = policy_loss + role_weighted_kl
                    metrics["actor/planner_kl_loss"] = planner_kl.detach().item()
                    metrics["actor/executor_kl_loss"] = executor_kl.detach().item()
                    metrics["actor/planner_kl_coef"] = planner_kl_coef
                    metrics["actor/executor_kl_coef"] = executor_kl_coef

                if self.config.use_dynamic_bsz:
                    loss = policy_loss * (len(micro) / self.config.ppo_mini_batch_size)
                else:
                    loss = policy_loss / gradient_accumulation
                loss.backward()

                data = {
                    "actor/planner_pg_loss": planner_terms["pg_loss"].detach().item(),
                    "actor/planner_entropy_loss": planner_terms["entropy_loss"].detach().item(),
                    "actor/planner_pg_clipfrac": planner_terms["pg_clipfrac"].detach().item(),
                    "actor/planner_ppo_kl": planner_terms["ppo_kl"].detach().item(),
                    "actor/executor_pg_loss": executor_terms["pg_loss"].detach().item(),
                    "actor/executor_entropy_loss": executor_terms["entropy_loss"].detach().item(),
                    "actor/executor_pg_clipfrac": executor_terms["pg_clipfrac"].detach().item(),
                    "actor/executor_ppo_kl": executor_terms["ppo_kl"].detach().item(),
                    "actor/other_pg_loss": pg_loss.detach().item(),
                    "actor/other_entropy_loss": entropy_loss.detach().item(),
                    "actor/other_pg_clipfrac": pg_clipfrac.detach().item(),
                    "actor/other_ppo_kl": ppo_kl.detach().item(),
                }
                append_to_dict(metrics, data)

            grad_norm = self._optimizer_step()
            append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})
        self.actor_optimizer.zero_grad()
        return metrics
