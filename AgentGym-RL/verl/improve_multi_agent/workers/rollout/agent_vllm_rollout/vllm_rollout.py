from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torch import nn
from tqdm import tqdm

from verl import DataProto
from verl.multi_agent.workers.rollout.agent_vllm_rollout.vllm_rollout import vLLMRollout as LegacyVLLMRollout
from verl.multi_agent.workers.rollout.schemas import (
    EXECUTOR_SPEAKER_ID,
    PLANNER_SPEAKER_ID,
    RolloutHandler,
)
from verl.utils.agentgym.client import init_env_client
from verl.utils.model import compute_position_id_with_mask

from ....protocol import (
    PlannerMessage,
    build_executor_prompt,
    build_planner_prompt,
    extract_available_actions,
    render_babyai_action_payload,
    safe_json_dumps,
    validate_executor_json,
    validate_planner_json,
)
from ....rewarding import (
    compute_executor_reward,
    compute_planner_reward,
    detect_babyai_milestones,
    detect_subgoal_completion,
)


class vLLMRollout(LegacyVLLMRollout):
    def __init__(
        self,
        actor_module: nn.Module,
        rollout_config: DictConfig,
        agentgym_config: DictConfig,
        tokenizer,
        model_hf_config,
        improve_config=None,
        **kwargs,
    ):
        self.improve_config = improve_config
        super().__init__(
            actor_module=actor_module,
            rollout_config=rollout_config,
            agentgym_config=agentgym_config,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            multi_agent_config=improve_config,
            **kwargs,
        )
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.trace_executor_payload = True
        if self.trace_executor_payload:
            self.trace_dir = str(self._multi_agent_get("debug.trace_dir", "/mnt/data/logs"))
            self.trace_file = os.path.join(self.trace_dir, f"executor_payload_trace_rank{self.rank}.jsonl")
            os.makedirs(self.trace_dir, exist_ok=True)
        self.topology = "planner_executor"
        self.invalid_output_max_retries = 0
        self.planner_max_retries = 0
        self.planner_interval = int(self._multi_agent_get("planner_interval", 3))
        self.debug_progress = os.getenv("VERL_IMPROVE_DEBUG_PROGRESS", "0") == "1"
        self.debug_all_ranks = os.getenv("VERL_IMPROVE_DEBUG_ALL_RANKS", "0") == "1"

    def _debug(self, message: str) -> None:
        if self.debug_progress and (self.debug_all_ranks or self.rank == 0):
            print(f"[improve-rollout][{time.strftime('%H:%M:%S')}] {message}", flush=True)

    def _planner_message_to_prompt(self, message: PlannerMessage) -> str:
        return message.to_json()

    def _build_turn_prompt_token_ids(self, prompt: str) -> List[int]:
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        except Exception:
            encoded = self.tokenizer(prompt, add_special_tokens=False)
            return list(encoded["input_ids"])

    def _feature_enabled(self, key: str, default: bool = True) -> bool:
        if self.improve_config is None:
            return default
        value = OmegaConf.select(self.improve_config, f"features.{key}", default=default)
        return bool(default if value is None else value)

    @staticmethod
    def _is_punctuation_only(text: str) -> bool:
        stripped = re.sub(r"[\s_]+", "", text or "")
        return bool(stripped) and not any(ch.isalnum() for ch in stripped)

    @staticmethod
    def _align_non_tensor_entries(entries: List[Any], target_size: int) -> np.ndarray:
        def _to_object_vector(values: List[Any]) -> np.ndarray:
            array = np.empty((len(values),), dtype=object)
            array[:] = values
            return array

        if len(entries) == target_size:
            return _to_object_vector(entries)
        if not entries:
            return np.empty((target_size,), dtype=object)
        if target_size % len(entries) == 0:
            repeat_factor = target_size // len(entries)
            expanded = [entry for entry in entries for _ in range(repeat_factor)]
            return _to_object_vector(expanded)
        if len(entries) == 1:
            return _to_object_vector(entries * target_size)
        raise AssertionError(
            f"Cannot align non-tensor entries of length {len(entries)} to batch size {target_size}"
        )

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        global_steps = prompts.meta_info.get("global_steps", None)
        max_rounds = prompts.meta_info.get("max_rounds", self.task_profile.default_max_rounds)
        cur_device = prompts.batch["input_ids"].device
        do_sample = prompts.meta_info.get("do_sample", True)

        batch_size = prompts.batch["input_ids"].size(0) * self.config.n
        self._debug(
            f"generate_sequences:start global_steps={global_steps} max_rounds={max_rounds} batch_size={batch_size}"
        )
        rollout_handler_ls = self.preprocess_prompt_to_rollout_handler(prompts, n=self.config.n)
        env_clients = [init_env_client(self.agentgym_config) for _ in range(batch_size)]
        previous_scores = [0.0] * batch_size
        current_plans: List[Optional[PlannerMessage]] = [None] * batch_size
        planner_window_ids = [0] * batch_size
        planner_steps_remaining = [0] * batch_size
        force_replan_next = [True] * batch_size
        valid_action_streaks = [0] * batch_size
        milestone_counts = [0.0] * batch_size
        subgoal_success_counts = [0.0] * batch_size
        planner_json_validity = [0.0] * batch_size
        executor_legal_action_rate = [0.0] * batch_size
        for idx, handler in enumerate(rollout_handler_ls):
            handler.planner_window_id = 0
            handler.planner_json_validity = 1.0
            handler.executor_legal_action_rate = 1.0
            handler.executor_invalid_output_rate = 0.0
            handler.milestone_hit_rate = 0.0
            handler.subgoal_success_rate = 0.0
            handler.collapse_state = "ok"
            handler.transition_outcome_recorded = 0.0
            handler.transition_terminal_failure = 0.0
            try:
                env_clients[idx].reset(handler.item_id)
                observation = env_clients[idx].observe()
                handler.latest_observation = observation
            except Exception as exc:
                handler.done = True
                handler.env_step_failed = 1.0
                handler.executor_action_valid = 0.0
                handler.latest_observation = str(exc)
                handler.transition_outcome_recorded = 1.0
                handler.transition_terminal_failure = 1.0

        rounds = 0
        rollout_bar = tqdm(total=max_rounds, desc="Improve multi-agent rounds", disable=torch.distributed.get_rank() != 0)

        while rounds < max_rounds:
            active_indices = [idx for idx, handler in enumerate(rollout_handler_ls) if not handler.done]
            if not active_indices:
                self._debug(f"round={rounds + 1} no_active_indices -> rollout_complete")
                break

            replanning_indices = [idx for idx in active_indices if force_replan_next[idx] or planner_steps_remaining[idx] <= 0]
            self._debug(
                f"round={rounds + 1} start active={len(active_indices)} replanning={len(replanning_indices)}"
            )
            if replanning_indices:
                planner_prompt_ids = []
                planner_prompt_texts = []
                planner_legal_actions: List[List[str]] = []
                for idx in replanning_indices:
                    handler = rollout_handler_ls[idx]
                    legal_actions = extract_available_actions(handler.latest_observation)
                    planner_legal_actions.append(legal_actions)
                    planner_prompt = build_planner_prompt(handler.latest_observation, legal_actions=legal_actions)
                    handler.latest_planner_prompt = planner_prompt
                    planner_prompt_texts.append(planner_prompt)
                    planner_prompt_ids.append(self._build_turn_prompt_token_ids(planner_prompt))
                with self.update_sampling_params(**self._sampling_overrides("planner", do_sample, kwargs.copy())):
                    self._debug(f"round={rounds + 1} planner_generate:start prompts={len(planner_prompt_ids)}")
                    planner_output = self.inference_engine.generate(
                        prompts=None,
                        prompt_token_ids=planner_prompt_ids,
                        sampling_params=self.sampling_params,
                        use_tqdm=False,
                    )
                    self._debug(f"round={rounds + 1} planner_generate:end prompts={len(planner_prompt_ids)}")
                planner_response_ids = self._extract_response_token_ids(planner_output)
                for local_i, idx in enumerate(replanning_indices):
                    handler = rollout_handler_ls[idx]
                    raw_text = self.tokenizer.decode(planner_response_ids[local_i], skip_special_tokens=True)
                    legal_actions = planner_legal_actions[local_i]
                    handler.add_user_message(self.tokenizer, planner_prompt_texts[local_i])
                    validation = validate_planner_json(raw_text, legal_actions)
                    planner_message = validation.message
                    current_plans[idx] = planner_message
                    planner_window_ids[idx] += 1
                    planner_steps_remaining[idx] = self.planner_interval
                    force_replan_next[idx] = False
                    handler.planner_window_id = planner_window_ids[idx]
                    handler.latest_planner_raw_output = raw_text
                    handler.latest_planner_message = planner_message.to_json()
                    handler.latest_planner_context_used = planner_message.to_json()
                    handler.latest_planner_context_source = "original" if validation.valid else "fallback"
                    handler.latest_planner_validation_reason = validation.reason
                    handler.planner_output_valid = 1.0 if validation.valid else 0.0
                    handler.planner_fallback_used = 0.0 if validation.valid else 1.0
                    handler.planner_tag_only = 1.0 if self._is_punctuation_only(raw_text) else 0.0
                    handler.planner_exact_action = 0.0
                    handler.planner_degenerate_fragment = 0.0
                    handler.planner_message_token_count = float(len(handler.latest_planner_message.split()))
                    handler.planner_json_validity = handler.planner_output_valid
                    handler.add_assistant_message(
                        self.tokenizer,
                        planner_message.to_json(),
                        PLANNER_SPEAKER_ID,
                        prompt_content=planner_message.to_json(),
                        trainable=validation.valid,
                        ppo_weight=float(self.role_training["planner"]["ppo_weight"]),
                        kl_weight=float(self.role_training["planner"]["kl_weight"]),
                    )
                    planner_reward = compute_planner_reward(
                        valid_json=validation.valid,
                        executable_hint=validation.executable_hint,
                        subgoal_success=False,
                        contradiction=validation.contradiction,
                    )
                    if self._feature_enabled("role_aux_rewards", True):
                        handler.mark_last_planner_reward(planner_reward.total)
                    planner_json_validity[idx] = handler.planner_json_validity

            executor_prompt_ids = []
            executor_prompt_texts = []
            legal_actions_by_idx: Dict[int, List[str]] = {}
            for idx in active_indices:
                handler = rollout_handler_ls[idx]
                legal_actions = extract_available_actions(handler.latest_observation)
                legal_actions_by_idx[idx] = legal_actions
                planner_message = current_plans[idx]
                if planner_message is None:
                    planner_message = validate_planner_json("{}", legal_actions).message
                    current_plans[idx] = planner_message
                executor_prompt = build_executor_prompt(
                    observation=handler.latest_observation,
                    planner_message=planner_message,
                    legal_actions=legal_actions,
                )
                executor_prompt_texts.append(executor_prompt)
                executor_prompt_ids.append(self._build_turn_prompt_token_ids(executor_prompt))

            with self.update_sampling_params(**self._sampling_overrides("executor", do_sample, kwargs.copy())):
                self._debug(f"round={rounds + 1} executor_generate:start prompts={len(executor_prompt_ids)}")
                executor_output = self.inference_engine.generate(
                    prompts=None,
                    prompt_token_ids=executor_prompt_ids,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )
                self._debug(f"round={rounds + 1} executor_generate:end prompts={len(executor_prompt_ids)}")
            executor_response_ids = self._extract_response_token_ids(executor_output)
            step_trace_events = []
            for local_i, idx in enumerate(active_indices):
                handler = rollout_handler_ls[idx]
                raw_text = self.tokenizer.decode(executor_response_ids[local_i], skip_special_tokens=True)
                legal_actions = legal_actions_by_idx[idx]
                handler.add_user_message(self.tokenizer, executor_prompt_texts[local_i])
                validation = validate_executor_json(raw_text, legal_actions)
                handler.executor_first_pass_valid = 1.0 if validation.valid else 0.0
                handler.executor_retry_used = 0.0
                handler.executor_retry_resolved = 0.0
                handler.executor_retry_count = 0.0
                handler.executor_action_changed_after_retry = 0.0
                handler.team_env_rounds += 1
                planner_message = current_plans[idx]
                trace_event = {
                    "training_step": global_steps,
                    "item_id": handler.item_id,
                    "task_name": handler.task_name,
                    "round": rounds + 1,
                    "rank": self.rank,
                    "observation_excerpt": self._clip(handler.latest_observation),
                    "planner_prompt": self._clip(handler.latest_planner_prompt),
                    "planner_message": self._clip(handler.latest_planner_message),
                    "planner_raw_output": self._clip(handler.latest_planner_raw_output),
                    "planner_validation_valid": bool(handler.planner_output_valid),
                    "planner_validation_reason": handler.latest_planner_validation_reason,
                    "executor_prompt": self._clip(executor_prompt_texts[local_i]),
                    "executor_raw_output": self._clip(raw_text),
                    "validation_valid": bool(validation.valid),
                    "validation_reason": validation.reason,
                    "available_actions": legal_actions,
                    "extracted_action": validation.action,
                    "env_step_called": False,
                    "env_step_payload": "",
                    "env_reward": None,
                    "env_done": None,
                    "env_state_excerpt": "",
                    "trace_reasons": [],
                }

                if validation.valid and validation.decision is not None:
                    assistant_json = validation.decision.to_json()
                    handler.add_assistant_message(
                        self.tokenizer,
                        assistant_json,
                        EXECUTOR_SPEAKER_ID,
                        prompt_content=assistant_json,
                        trainable=True,
                        ppo_weight=float(self.role_training["executor"]["ppo_weight"]),
                        kl_weight=float(self.role_training["executor"]["kl_weight"]),
                    )
                    payload = render_babyai_action_payload(validation.decision, legal_actions)
                    try:
                        previous_observation = handler.latest_observation
                        trace_event["env_step_called"] = True
                        trace_event["env_step_payload"] = payload
                        step_output = env_clients[idx].step(payload)
                        state = step_output.state
                        reward = float(step_output.reward)
                        done = bool(step_output.done)
                        milestones = detect_babyai_milestones(
                            previous_observation,
                            state,
                            planner_message,
                            previous_score=previous_scores[idx],
                            current_score=reward,
                            valid_action_streak=valid_action_streaks[idx] + 1,
                        )
                        milestone_hits = sum(milestones.values())
                        milestone_counts[idx] += milestone_hits
                        subgoal_success = detect_subgoal_completion(
                            planner_message,
                            previous_observation,
                            state,
                            validation.action or "",
                            milestones,
                        )
                        if subgoal_success:
                            subgoal_success_counts[idx] += 1.0
                        planner_reward = compute_planner_reward(
                            valid_json=True,
                            executable_hint=True,
                            subgoal_success=subgoal_success,
                            contradiction=False,
                        )
                        if subgoal_success and self._feature_enabled("role_aux_rewards", True):
                            handler.mark_last_planner_reward(planner_reward.subgoal_success)

                        executor_reward = compute_executor_reward(
                            valid_json=True,
                            legal_action=True,
                            observation_changed=previous_observation != state,
                            milestone_hits=milestones,
                            subgoal_success=subgoal_success,
                        )
                        if not self._feature_enabled("milestone_rewards", True):
                            milestones = {key: 0.0 for key in milestones}
                            executor_reward = compute_executor_reward(
                                valid_json=True,
                                legal_action=True,
                                observation_changed=previous_observation != state,
                                milestone_hits=milestones,
                                subgoal_success=subgoal_success,
                            )
                        delta_reward = 0.0
                        if self.reward_mode == "delta_per_executor_step":
                            delta_reward = reward - previous_scores[idx]
                        executor_aux = executor_reward.total if self._feature_enabled("role_aux_rewards", True) else 0.0
                        handler.mark_last_executor_reward(delta_reward + executor_aux)
                        previous_scores[idx] = reward
                        handler.score = reward
                        handler.done = done
                        handler.latest_observation = state
                        handler.executor_action_valid = 1.0
                        handler.executor_native_format_valid = 1.0
                        handler.transition_outcome_recorded = 1.0
                        handler.transition_terminal_failure = 0.0
                        valid_action_streaks[idx] += 1
                        executor_legal_action_rate[idx] = 1.0
                        handler.executor_legal_action_rate = 1.0
                        handler.milestone_hit_rate = milestone_counts[idx] / max(handler.team_env_rounds, 1)
                        handler.subgoal_success_rate = subgoal_success_counts[idx] / max(handler.planner_window_id, 1)
                        trace_event["env_reward"] = reward
                        trace_event["env_done"] = done
                        trace_event["env_state_excerpt"] = self._clip(state)
                        if subgoal_success or milestone_hits > 0 or milestones["inventory_changed"] or milestones["door_opened"]:
                            force_replan_next[idx] = True
                        else:
                            planner_steps_remaining[idx] -= 1
                    except Exception as exc:
                        handler.env_step_failed = 1.0
                        handler.executor_action_valid = 0.0
                        handler.executor_native_format_valid = 0.0
                        handler.done = True
                        handler.latest_observation = str(exc)
                        handler.mark_last_executor_reward(-0.2)
                        valid_action_streaks[idx] = 0
                        executor_legal_action_rate[idx] = 0.0
                        handler.executor_legal_action_rate = 0.0
                        handler.transition_outcome_recorded = 1.0
                        handler.transition_terminal_failure = 1.0
                        trace_event["env_state_excerpt"] = self._clip(exc)
                        trace_event["validation_reason"] = "env_exception"
                        force_replan_next[idx] = True
                else:
                    handler.add_assistant_message(
                        self.tokenizer,
                        raw_text,
                        EXECUTOR_SPEAKER_ID,
                        prompt_content=raw_text,
                        trainable=False,
                        ppo_weight=float(self.role_training["executor"]["ppo_weight"]),
                        kl_weight=float(self.role_training["executor"]["kl_weight"]),
                    )
                    handler.executor_action_valid = 0.0
                    handler.executor_native_format_valid = 0.0
                    handler.executor_invalid_output_rate = 1.0
                    handler.invalid_format_terminated = 1.0
                    handler.done = True
                    handler.mark_last_executor_reward(-0.2)
                    valid_action_streaks[idx] = 0
                    executor_legal_action_rate[idx] = 0.0
                    handler.executor_legal_action_rate = 0.0
                    handler.latest_observation = f"Terminated due to invalid executor output ({validation.reason})."
                    handler.transition_outcome_recorded = 1.0
                    handler.transition_terminal_failure = 1.0
                    trace_event["env_state_excerpt"] = self._clip(handler.latest_observation)
                    force_replan_next[idx] = True

                trace_event["executor_action_valid"] = bool(handler.executor_action_valid)
                trace_event["executor_native_format_valid"] = bool(handler.executor_native_format_valid)
                step_trace_events.append(trace_event)

            valid_count = sum(1 for event in step_trace_events if event["validation_valid"])
            env_step_count = sum(1 for event in step_trace_events if event["env_step_called"])
            done_count = sum(1 for idx in active_indices if rollout_handler_ls[idx].done)
            self._debug(
                "round="
                f"{rounds + 1} executor_process:end valid={valid_count}/{len(active_indices)} "
                f"env_step_called={env_step_count}/{len(active_indices)} done_now={done_count}/{len(active_indices)}"
            )
            trace_reasons = self._trace_reasons_for_step(step_trace_events, global_steps)
            if trace_reasons:
                for event in step_trace_events:
                    event["trace_reasons"] = trace_reasons
                self._append_trace_events(step_trace_events)

            rounds += 1
            rollout_bar.update(1)

        rollout_bar.close()
        self._debug(
            f"generate_sequences:end rounds_completed={rounds} completed_handlers={sum(handler.done for handler in rollout_handler_ls)}/{len(rollout_handler_ls)}"
        )
        if self.reward_mode == "final_only":
            for handler in rollout_handler_ls:
                handler.mark_last_executor_reward(handler.score)

        response_ids = []
        response_attention_mask = []
        response_loss_mask = []
        response_speaker_ids = []
        response_planner_mask = []
        response_executor_mask = []
        response_scores = []
        response_reward_event_mask = []
        team_env_rounds = []
        executor_action_valid = []
        executor_native_format_valid = []
        env_step_failed = []
        timeout_occurred = []
        invalid_format_terminated = []
        invalid_action_terminated = []
        planner_output_valid = []
        planner_fallback_used = []
        planner_rewrite_used = []
        planner_tag_only = []
        planner_exact_action = []
        planner_degenerate_fragment = []
        planner_message_token_count = []
        executor_first_pass_valid = []
        executor_retry_used = []
        executor_retry_resolved = []
        executor_retry_count = []
        executor_action_changed_after_retry = []
        milestone_hit_rate = []
        planner_window_id_tensor = []
        planner_json_validity_tensor = []
        executor_legal_action_rate_tensor = []
        subgoal_success_rate = []
        transition_outcome_recorded = []
        transition_terminal_failure = []
        messages = []
        for handler in rollout_handler_ls:
            handler.truncate_output_ids()
            response_ids.append(torch.tensor(handler.response_ids, dtype=torch.int, device=cur_device))
            response_attention_mask.append(torch.tensor(handler.response_attention_mask, dtype=torch.int, device=cur_device))
            response_loss_mask.append(torch.tensor(handler.response_loss_mask, dtype=torch.int, device=cur_device))
            response_speaker_ids.append(torch.tensor(handler.response_speaker_ids, dtype=torch.int, device=cur_device))
            response_planner_mask.append(torch.tensor(handler.response_planner_response_mask, dtype=torch.int, device=cur_device))
            response_executor_mask.append(torch.tensor(handler.response_executor_response_mask, dtype=torch.int, device=cur_device))
            response_scores.append(torch.tensor(handler.response_score_values, dtype=torch.float32, device=cur_device))
            response_reward_event_mask.append(torch.tensor(handler.response_reward_event_mask, dtype=torch.int, device=cur_device))
            team_env_rounds.append(float(handler.team_env_rounds))
            executor_action_valid.append(float(handler.executor_action_valid))
            executor_native_format_valid.append(float(handler.executor_native_format_valid))
            env_step_failed.append(float(handler.env_step_failed))
            timeout_occurred.append(float(handler.timeout_occurred))
            invalid_format_terminated.append(float(handler.invalid_format_terminated))
            invalid_action_terminated.append(float(handler.invalid_action_terminated))
            planner_output_valid.append(float(handler.planner_output_valid))
            planner_fallback_used.append(float(handler.planner_fallback_used))
            planner_rewrite_used.append(float(handler.planner_rewrite_used))
            planner_tag_only.append(float(handler.planner_tag_only))
            planner_exact_action.append(float(handler.planner_exact_action))
            planner_degenerate_fragment.append(float(handler.planner_degenerate_fragment))
            planner_message_token_count.append(float(handler.planner_message_token_count))
            executor_first_pass_valid.append(float(handler.executor_first_pass_valid))
            executor_retry_used.append(float(handler.executor_retry_used))
            executor_retry_resolved.append(float(handler.executor_retry_resolved))
            executor_retry_count.append(float(handler.executor_retry_count))
            executor_action_changed_after_retry.append(float(handler.executor_action_changed_after_retry))
            milestone_hit_rate.append(float(getattr(handler, "milestone_hit_rate", 0.0)))
            planner_window_id_tensor.append(float(getattr(handler, "planner_window_id", 0)))
            planner_json_validity_tensor.append(float(getattr(handler, "planner_json_validity", 0.0)))
            executor_legal_action_rate_tensor.append(float(getattr(handler, "executor_legal_action_rate", 0.0)))
            subgoal_success_rate.append(float(getattr(handler, "subgoal_success_rate", 0.0)))
            transition_outcome_recorded.append(float(getattr(handler, "transition_outcome_recorded", 0.0)))
            transition_terminal_failure.append(float(getattr(handler, "transition_terminal_failure", 0.0)))
            messages.append(handler.messages)

        response_ids = self._pad_tensor_list(response_ids, self.tokenizer.pad_token_id, self.config.response_length)
        response_attention_mask = self._pad_tensor_list(response_attention_mask, 0, self.config.response_length)
        response_loss_mask = self._pad_tensor_list(response_loss_mask, 0, self.config.response_length)
        response_speaker_ids = self._pad_tensor_list(response_speaker_ids, 0, self.config.response_length)
        response_planner_mask = self._pad_tensor_list(response_planner_mask, 0, self.config.response_length)
        response_executor_mask = self._pad_tensor_list(response_executor_mask, 0, self.config.response_length)
        response_scores = self._pad_tensor_list(response_scores, 0.0, self.config.response_length, dtype=torch.float32)
        response_reward_event_mask = self._pad_tensor_list(response_reward_event_mask, 0, self.config.response_length)
        prompt_input_ids = prompts.batch["input_ids"].repeat_interleave(self.config.n, dim=0)
        prompt_attention_mask = prompts.batch["attention_mask"].repeat_interleave(self.config.n, dim=0)
        prompt_position_ids = prompts.batch["position_ids"].repeat_interleave(self.config.n, dim=0)
        input_ids = torch.cat((prompt_input_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        position_ids = compute_position_id_with_mask(attention_mask)
        response_mask = response_loss_mask
        speaker_ids = torch.cat((torch.zeros_like(prompt_input_ids, dtype=torch.int, device=cur_device), response_speaker_ids), dim=-1)
        batch = TensorDict(
            {
                "prompts": prompt_input_ids,
                "responses": response_ids,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "response_mask": response_mask,
                "loss_mask": response_mask,
                "speaker_ids": speaker_ids,
                "response_speaker_ids": response_speaker_ids,
                "planner_response_mask": response_planner_mask,
                "executor_response_mask": response_executor_mask,
                "scores": response_scores,
                "token_level_scores": response_scores,
                "reward_event_mask": response_reward_event_mask,
                "team_env_rounds": torch.tensor(team_env_rounds, dtype=torch.float32, device=cur_device),
                "executor_action_valid": torch.tensor(executor_action_valid, dtype=torch.float32, device=cur_device),
                "executor_native_format_valid": torch.tensor(executor_native_format_valid, dtype=torch.float32, device=cur_device),
                "env_step_failed": torch.tensor(env_step_failed, dtype=torch.float32, device=cur_device),
                "timeout_occurred": torch.tensor(timeout_occurred, dtype=torch.float32, device=cur_device),
                "invalid_format_terminated": torch.tensor(invalid_format_terminated, dtype=torch.float32, device=cur_device),
                "invalid_action_terminated": torch.tensor(invalid_action_terminated, dtype=torch.float32, device=cur_device),
                "planner_output_valid": torch.tensor(planner_output_valid, dtype=torch.float32, device=cur_device),
                "planner_fallback_used": torch.tensor(planner_fallback_used, dtype=torch.float32, device=cur_device),
                "planner_rewrite_used": torch.tensor(planner_rewrite_used, dtype=torch.float32, device=cur_device),
                "planner_tag_only": torch.tensor(planner_tag_only, dtype=torch.float32, device=cur_device),
                "planner_exact_action": torch.tensor(planner_exact_action, dtype=torch.float32, device=cur_device),
                "planner_degenerate_fragment": torch.tensor(planner_degenerate_fragment, dtype=torch.float32, device=cur_device),
                "planner_message_token_count": torch.tensor(planner_message_token_count, dtype=torch.float32, device=cur_device),
                "executor_first_pass_valid": torch.tensor(executor_first_pass_valid, dtype=torch.float32, device=cur_device),
                "executor_retry_used": torch.tensor(executor_retry_used, dtype=torch.float32, device=cur_device),
                "executor_retry_resolved": torch.tensor(executor_retry_resolved, dtype=torch.float32, device=cur_device),
                "executor_retry_count": torch.tensor(executor_retry_count, dtype=torch.float32, device=cur_device),
                "executor_action_changed_after_retry": torch.tensor(executor_action_changed_after_retry, dtype=torch.float32, device=cur_device),
                "milestone_hit_rate": torch.tensor(milestone_hit_rate, dtype=torch.float32, device=cur_device),
                "planner_window_id": torch.tensor(planner_window_id_tensor, dtype=torch.float32, device=cur_device),
                "planner_json_validity": torch.tensor(planner_json_validity_tensor, dtype=torch.float32, device=cur_device),
                "executor_legal_action_rate": torch.tensor(executor_legal_action_rate_tensor, dtype=torch.float32, device=cur_device),
                "subgoal_success_rate": torch.tensor(subgoal_success_rate, dtype=torch.float32, device=cur_device),
                "transition_outcome_recorded": torch.tensor(transition_outcome_recorded, dtype=torch.float32, device=cur_device),
                "transition_terminal_failure": torch.tensor(transition_terminal_failure, dtype=torch.float32, device=cur_device),
            },
            batch_size=(batch_size,),
        )
        non_tensor_batch = {
            "item_id": self._align_non_tensor_entries([handler.item_id for handler in rollout_handler_ls], batch_size),
            "raw_prompt": self._align_non_tensor_entries(list(prompts.non_tensor_batch["raw_prompt"]), batch_size),
            "messages": self._align_non_tensor_entries(messages, batch_size),
        }
        meta_info = {
            "micro_batch_size": prompts.meta_info.get("micro_batch_size", None),
            "temperature": prompts.meta_info.get("temperature", self.config.temperature),
            "use_dynamic_bsz": prompts.meta_info.get("use_dynamic_bsz", False),
            "max_token_len": prompts.meta_info.get("max_token_len", self.config.max_tokens),
        }
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)
