# Multi-agent extension.
# Derived from: /Users/mavinomichael/PycharmProjects/AgentGym-RL/AgentGym-RL/verl/agent_trainer/ppo/ray_trainer.py
# Original file left untouched for comparison.

from copy import deepcopy
import uuid

import numpy as np
import torch
from verl import DataProto
from verl.agent_trainer.ppo.ray_trainer import (
    RayPPOTrainer as BaseRayPPOTrainer,
    ResourcePoolManager,
    Role,
    FixedRoundsScheduler,
    StepRoundsScheduler,
    _timer,
    apply_kl_penalty,
    compute_advantage,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.agent_trainer.ppo.ray_trainer import compute_data_metrics as compute_single_agent_metrics
from verl.multi_agent.utils.agent_dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path


def compute_data_metrics(batch, use_critic=True):
    metrics = compute_single_agent_metrics(batch=batch, use_critic=use_critic)
    if "planner_response_mask" not in batch.batch.keys():
        return metrics

    planner_tokens = batch.batch["planner_response_mask"].sum(-1).float()
    executor_tokens = batch.batch["executor_response_mask"].sum(-1).float()
    reward_events = batch.batch["reward_event_mask"].sum(-1).float()
    env_rounds = batch.batch.get("team_env_rounds", batch.batch["task_rounds"]).float()
    valid_actions = batch.batch.get(
        "executor_action_valid",
        torch.ones_like(env_rounds, dtype=torch.float32),
    ).float()

    metrics.update(
        {
            "planner_tokens/mean": torch.mean(planner_tokens).detach().item(),
            "executor_tokens/mean": torch.mean(executor_tokens).detach().item(),
            "planner_to_executor_ratio": (
                (torch.mean(planner_tokens) / (torch.mean(executor_tokens) + 1e-6)).detach().item()
            ),
            "executor_invalid_action_rate": (1.0 - torch.mean(valid_actions)).detach().item(),
            "env_rounds/mean": torch.mean(env_rounds).detach().item(),
            "reward_events/mean": torch.mean(reward_events).detach().item(),
        }
    )
    return metrics


class RayPPOTrainer(BaseRayPPOTrainer):
    def _create_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

        self.train_dataset = RLHFDataset(
            data_file=self.config.data.train_file,
            tokenizer=self.tokenizer,
            data_config=self.config.data,
            agentgym_config=self.config.actor_rollout_ref.agentgym,
            multi_agent_config=self.config.get("multi_agent", None),
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
        assert len(self.train_dataloader) >= 1
        print(f"Size of train dataloader: {len(self.train_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        if self.config.algorithm.rounds_ctrl.type == "fixed":
            self.rounds_scheduler = FixedRoundsScheduler(rounds=self.config.algorithm.rounds_ctrl.rounds)
        elif self.config.algorithm.rounds_ctrl.type == "scaling_inter_stepwise":
            self.rounds_scheduler = StepRoundsScheduler(
                steps_scaling_inter=self.config.algorithm.rounds_ctrl.steps_scaling_inter,
                rounds_ls=self.config.algorithm.rounds_ctrl.rounds,
            )
        else:
            raise NotImplementedError
        print(f"Total training steps: {self.total_training_steps}")

        from omegaconf import OmegaConf, open_dict

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def fit(self):
        # ORIGINAL FLOW DIFFERENCE:
        # single-agent path logged only aggregate trajectory metrics.
        # multi-agent path adds planner/executor token and reward-event metrics.
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.config.trainer.storage_mode == "aistudio":
            self._save_checkpoint()

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
                metrics.update({"max_rounds": self.rounds_scheduler.get_rounds()})

                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

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

                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    self._balance_batch(batch, metrics=metrics)
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
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

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                        )

                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1
                self.rounds_scheduler.step()

                if self.global_steps >= self.total_training_steps:
                    if self.config.trainer.save_freq > 0 and (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                    return


__all__ = ["RayPPOTrainer", "ResourcePoolManager", "Role"]
