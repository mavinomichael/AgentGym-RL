from __future__ import annotations

import json
import os
from copy import deepcopy
import time
import uuid

import numpy as np
import torch
from verl import DataProto
from verl.agent_trainer.ppo.ray_trainer import (
    FixedRoundsScheduler,
    ResourcePoolManager,
    Role,
    StepRoundsScheduler,
    _timer,
    apply_kl_penalty,
    compute_advantage,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.agent_trainer.ppo.ray_trainer import compute_data_metrics as compute_single_agent_metrics
from verl.multi_agent.ppo.ray_trainer import _infer_task_name
from verl.multi_agent.ppo.ray_trainer import RayPPOTrainer as BaseRayPPOTrainer
from verl.utils.tracking import Tracking

from ..advantage import RoleRewardNormalizer, compute_rolewise_gae_advantage_return
from ..dataset import ImproveRLHFDataset, collate_fn
from ..monitoring import CollapseMonitor


def _parse_save_steps(raw_value):
    if raw_value is None:
        return set()
    if isinstance(raw_value, str):
        import re

        return {int(token) for token in re.findall(r"\d+", raw_value)}
    save_steps = set()
    try:
        for item in raw_value:
            if item is None:
                continue
            save_steps.add(int(item))
    except TypeError:
        return set()
    return save_steps


def compute_data_metrics(batch, use_critic=True):
    if "task_scores" not in batch.batch.keys():
        if "scores" in batch.batch.keys():
            batch.batch["task_scores"] = batch.batch["scores"]
        elif "token_level_scores" in batch.batch.keys():
            batch.batch["task_scores"] = batch.batch["token_level_scores"]
    if "task_rounds" not in batch.batch.keys():
        if "team_env_rounds" in batch.batch.keys():
            batch.batch["task_rounds"] = batch.batch["team_env_rounds"]
        elif "planner_window_id" in batch.batch.keys():
            batch.batch["task_rounds"] = batch.batch["planner_window_id"]

    metrics = compute_single_agent_metrics(batch=batch, use_critic=use_critic)
    task_name = _infer_task_name(batch)
    if "planner_response_mask" in batch.batch.keys():
        planner_tokens = batch.batch["planner_response_mask"].sum(-1).float()
        executor_tokens = batch.batch["executor_response_mask"].sum(-1).float()
        metrics["planner_tokens/mean"] = torch.mean(planner_tokens).detach().item()
        metrics["executor_tokens/mean"] = torch.mean(executor_tokens).detach().item()
    for key, metric_name in (
        ("planner_json_validity", f"planner_json_validity/{task_name}"),
        ("executor_legal_action_rate", f"executor_legal_action_rate/{task_name}"),
        ("milestone_hit_rate", f"milestone_hit_rate/{task_name}"),
        ("subgoal_success_rate", f"subgoal_success_rate/{task_name}"),
    ):
        if key in batch.batch.keys():
            metrics[metric_name] = torch.mean(batch.batch[key].float()).detach().item()
    invalid_rate = 0.0
    if "executor_native_format_valid" in batch.batch.keys():
        invalid_rate = (1.0 - torch.mean(batch.batch["executor_native_format_valid"].float())).detach().item()
        metrics[f"executor_invalid_output_rate/{task_name}"] = invalid_rate
    metrics["planner_json_validity"] = metrics.get(f"planner_json_validity/{task_name}", 1.0)
    metrics["executor_legal_action_rate"] = metrics.get(f"executor_legal_action_rate/{task_name}", 1.0)
    metrics["milestone_hit_rate"] = metrics.get(f"milestone_hit_rate/{task_name}", 0.0)
    metrics["planner_fallback_rate"] = metrics.get(f"planner_fallback_rate/{task_name}", 0.0)
    metrics["planner_tag_only_rate"] = metrics.get(f"planner_tag_only_rate/{task_name}", 0.0)
    metrics["executor_invalid_output_rate"] = invalid_rate
    return metrics


def compute_rolewise_advantage(data: DataProto, gamma=1.0, lam=1.0, reward_normalizer: RoleRewardNormalizer | None = None):
    values = data.batch["values"]
    response_mask = data.batch["response_mask"]
    token_level_rewards = data.batch["token_level_rewards"]
    planner_mask = data.batch.get("planner_response_mask", torch.zeros_like(response_mask)).float()
    executor_mask = data.batch.get("executor_response_mask", torch.zeros_like(response_mask)).float()
    if values.shape[-1] != token_level_rewards.shape[-1]:
        target_len = min(values.shape[-1], response_mask.shape[-1], token_level_rewards.shape[-1])
        values = values[:, -target_len:]
        response_mask = response_mask[:, -target_len:]
        token_level_rewards = token_level_rewards[:, -target_len:]
        planner_mask = planner_mask[:, -target_len:]
        executor_mask = executor_mask[:, -target_len:]
    if reward_normalizer is not None:
        token_level_rewards = reward_normalizer.normalize(token_level_rewards, planner_mask, executor_mask)
        data.batch["token_level_rewards"] = token_level_rewards
    advantages, returns = compute_rolewise_gae_advantage_return(
        token_level_rewards=token_level_rewards,
        values=values,
        response_mask=response_mask,
        planner_mask=planner_mask,
        executor_mask=executor_mask,
        gamma=gamma,
        lam=lam,
    )
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


def _assert_training_batch_ready(batch: DataProto, use_reference_policy: bool) -> None:
    missing_batch_keys = [
        key for key in ("old_log_probs", "advantages", "returns") if key not in batch.batch.keys()
    ]
    if use_reference_policy and "ref_log_prob" not in batch.batch.keys():
        missing_batch_keys.append("ref_log_prob")
    if "transition_outcome_recorded" not in batch.batch.keys():
        missing_batch_keys.append("transition_outcome_recorded")
    missing_meta_keys = [key for key in ("global_token_num",) if key not in batch.meta_info]
    if missing_batch_keys or missing_meta_keys:
        raise RuntimeError(
            "Improve multi-agent smoke guard failed: "
            f"missing batch keys={missing_batch_keys} missing meta keys={missing_meta_keys}"
        )
    unresolved_mask = batch.batch["transition_outcome_recorded"] < 0.5
    if torch.any(unresolved_mask):
        unresolved_indices = torch.nonzero(unresolved_mask, as_tuple=False).view(-1).tolist()
        raise RuntimeError(
            "Improve multi-agent rollout invariant failed: "
            f"unresolved transitions at batch indices={unresolved_indices}"
        )


class RayPPOTrainer(BaseRayPPOTrainer):
    @staticmethod
    def _debug_enabled() -> bool:
        return os.getenv("VERL_IMPROVE_DEBUG_PROGRESS", "0") == "1"

    def _debug(self, message: str) -> None:
        if self._debug_enabled():
            print(f"[improve-trainer][{time.strftime('%H:%M:%S')}] {message}", flush=True)

    def _create_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

        self.train_dataset = ImproveRLHFDataset(
            data_file=self.config.data.train_file,
            tokenizer=self.tokenizer,
            data_config=self.config.data,
            agentgym_config=self.config.actor_rollout_ref.agentgym,
        )
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get("seed", 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps
        if self.config.algorithm.rounds_ctrl.type == "fixed":
            self.rounds_scheduler = FixedRoundsScheduler(rounds=self.config.algorithm.rounds_ctrl.rounds)
        else:
            self.rounds_scheduler = StepRoundsScheduler(
                steps_scaling_inter=self.config.algorithm.rounds_ctrl.steps_scaling_inter,
                rounds_ls=self.config.algorithm.rounds_ctrl.rounds,
            )
        from omegaconf import OmegaConf, open_dict

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def fit(self):
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=self.config,
        )
        self.global_steps = 0
        self._load_checkpoint()
        explicit_save_steps = _parse_save_steps(self.config.trainer.get("save_steps", []))
        use_role_local_advantage = bool(self.config.improve_multi_agent.features.get("role_local_advantage", True))
        reward_normalizer = RoleRewardNormalizer() if use_role_local_advantage else None
        collapse_monitor = CollapseMonitor()
        collapse_dir = os.path.join(self.config.trainer.default_local_dir, "collapse_monitor")
        os.makedirs(collapse_dir, exist_ok=True)

        def should_save_checkpoint(step: int) -> bool:
            if step in explicit_save_steps:
                return True
            return self.config.trainer.save_freq > 0 and step % self.config.trainer.save_freq == 0

        self.global_steps += 1
        for _epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["item_id", "raw_prompt"],
                )
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch.meta_info["max_rounds"] = self.rounds_scheduler.get_rounds()
                metrics["max_rounds"] = self.rounds_scheduler.get_rounds()

                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        self._debug(
                            f"step={self.global_steps} gen:start max_rounds={gen_batch.meta_info['max_rounds']} batch_size={len(gen_batch.batch)}"
                        )
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        self._debug(
                            "step="
                            f"{self.global_steps} gen:end response_batch={len(gen_batch_output.batch)} "
                            f"transition_recorded_mean={float(gen_batch_output.batch['transition_outcome_recorded'].float().mean().item()):.3f}"
                        )

                    if self.config.algorithm.adv_estimator == "remax":
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = batch.batch["rewards"].sum(dim=-1)
                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            batch.batch["reward_baselines"] = reward_baseline_tensor
                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        dtype=object,
                    )
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    self._balance_batch(batch, metrics=metrics)
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("old_log_prob", timing_raw):
                        self._debug(f"step={self.global_steps} old_log_prob:start")
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)
                        self._debug(f"step={self.global_steps} old_log_prob:end")

                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            self._debug(f"step={self.global_steps} ref_log_prob:start")
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                            self._debug(f"step={self.global_steps} ref_log_prob:end")

                    if self.use_critic:
                        with _timer("values", timing_raw):
                            self._debug(f"step={self.global_steps} values:start")
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)
                            self._debug(f"step={self.global_steps} values:end")
                    with _timer("adv", timing_raw):
                        self._debug(f"step={self.global_steps} adv:start")
                        reward_tensor = batch.batch["scores"]
                        batch.batch["token_level_scores"] = reward_tensor
                        if not self.config.actor_rollout_ref.actor.get("use_kl_loss", False):
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                        if use_role_local_advantage:
                            batch = compute_rolewise_advantage(
                                batch,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                reward_normalizer=reward_normalizer,
                            )
                        else:
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                            )
                        self._debug(f"step={self.global_steps} adv:end")
                    _assert_training_batch_ready(batch, use_reference_policy=self.use_reference_policy)
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            self._debug(f"step={self.global_steps} update_critic:start")
                            critic_output = self.critic_wg.update_critic(batch)
                            self._debug(f"step={self.global_steps} update_critic:end")
                        metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            self._debug(f"step={self.global_steps} update_actor:start")
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                            self._debug(f"step={self.global_steps} update_actor:end")
                        metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
                    if should_save_checkpoint(self.global_steps):
                        with _timer("save_checkpoint", timing_raw):
                            self._debug(f"step={self.global_steps} save_checkpoint:start")
                            self._save_checkpoint()
                            self._debug(f"step={self.global_steps} save_checkpoint:end")

                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                collapse_state = collapse_monitor.update(metrics)
                metrics["collapse_state"] = collapse_state.status
                metrics["collapse_reasons"] = ",".join(collapse_state.reasons)
                logger.log(data=metrics, step=self.global_steps)
                self._debug(f"step:{self.global_steps} logged collapse_state={collapse_state.status}")

                if collapse_state.status == "collapse":
                    summary_path = os.path.join(collapse_dir, f"step_{self.global_steps:06d}.json")
                    with open(summary_path, "w", encoding="utf-8") as handle:
                        json.dump(
                            {
                                "step": self.global_steps,
                                "reasons": collapse_state.reasons,
                                "metrics": {key: float(value) if isinstance(value, (int, float)) else value for key, value in metrics.items()},
                            },
                            handle,
                            ensure_ascii=True,
                            indent=2,
                        )
                    if should_save_checkpoint(self.global_steps) and not should_save_checkpoint(self.global_steps - 1):
                        self._save_checkpoint()
                    raise RuntimeError(f"Collapse detected at step {self.global_steps}: {collapse_state.reasons}")

                self.global_steps += 1
                self.rounds_scheduler.step()
                if self.global_steps >= self.total_training_steps:
                    if should_save_checkpoint(self.global_steps) and not should_save_checkpoint(self.global_steps - 1):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                    return


__all__ = ["RayPPOTrainer", "ResourcePoolManager", "Role"]
