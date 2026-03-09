# Multi-agent extension.
# Derived from: /Users/mavinomichael/PycharmProjects/AgentGym-RL/AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py
# Original file left untouched for comparison.

from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from verl import DataProto
from verl.multi_agent.envs import (
    CONTROL_SPEAKER_ID,
    EXECUTOR_SPEAKER_ID,
    PLANNER_SPEAKER_ID,
    build_executor_retry_prompt,
    build_executor_turn_prompt,
    build_planner_turn_prompt,
    compute_reward_delta,
    detect_invalid_action,
    extract_available_actions_from_observation,
    get_task_profile,
    normalize_executor_payload,
    validate_executor_payload,
)
from verl.multi_agent.workers.rollout.schemas import Message, RolloutHandler, _pre_process_inputs
from verl.utils.agentgym.client import init_env_client
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import pad_sequence_to_length
from verl.workers.rollout.agent_vllm_rollout.vllm_rollout import vLLMRollout as BaseVLLMRollout

logger = logging.getLogger(__name__)


class vLLMRollout(BaseVLLMRollout):
    def __init__(
        self,
        actor_module: nn.Module,
        rollout_config: DictConfig,
        agentgym_config: DictConfig,
        tokenizer,
        model_hf_config,
        multi_agent_config=None,
        **kwargs,
    ):
        self.multi_agent_config = multi_agent_config
        self.task_profile = get_task_profile(agentgym_config.task_name)
        super().__init__(
            actor_module=actor_module,
            rollout_config=rollout_config,
            agentgym_config=agentgym_config,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            **kwargs,
        )
        self.role_defaults = {
            "planner": {
                "max_tokens": int(self._multi_agent_get("roles.planner.max_tokens", self.task_profile.planner_max_tokens)),
                "temperature": float(self._multi_agent_get("roles.planner.temperature", self.config.temperature)),
            },
            "executor": {
                "max_tokens": int(self._multi_agent_get("roles.executor.max_tokens", self.task_profile.executor_max_tokens)),
                "temperature": float(self._multi_agent_get("roles.executor.temperature", self.config.temperature)),
            },
        }
        self.reward_mode = self._multi_agent_get("reward_mode", "delta_per_executor_step")
        self.invalid_output_policy = str(
            self._multi_agent_get("invalid_output.policy", "terminate_with_penalty")
        ).lower()
        self.invalid_output_penalty = float(self._multi_agent_get("invalid_output.penalty", -0.2))
        self.invalid_output_max_retries = max(0, int(self._multi_agent_get("invalid_output.max_retries", 2)))
        self.invalid_output_retry_temperature = float(self._multi_agent_get("invalid_output.retry_temperature", 0.2))
        self.invalid_output_retry_max_tokens = int(self._multi_agent_get("invalid_output.retry_max_tokens", 80))
        self.trace_executor_payload = self._as_bool(self._multi_agent_get("debug.trace_executor_payload", False))
        self.trace_dir = str(self._multi_agent_get("debug.trace_dir", "/mnt/data/logs"))
        self.trace_max_chars = int(self._multi_agent_get("debug.trace_max_chars", 800))
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.trace_file = (
            os.path.join(self.trace_dir, f"executor_payload_trace_rank{self.rank}.jsonl")
            if self.trace_executor_payload
            else None
        )
        if self.trace_executor_payload:
            os.makedirs(self.trace_dir, exist_ok=True)

    def _multi_agent_get(self, dotted_key: str, default=None):
        config = self.multi_agent_config
        if config is None:
            return default
        if isinstance(config, dict):
            current = config
            for key in dotted_key.split("."):
                if key not in current:
                    return default
                current = current[key]
            return current
        current = config
        for key in dotted_key.split("."):
            if not hasattr(current, key):
                return default
            current = getattr(current, key)
        return current

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _clip(self, value: Any) -> str:
        text = "" if value is None else str(value)
        if len(text) <= self.trace_max_chars:
            return text
        return text[: self.trace_max_chars] + "...[truncated]"

    def _append_trace_events(self, events: List[Dict[str, Any]]) -> None:
        if not self.trace_executor_payload or not events or self.trace_file is None:
            return
        try:
            with open(self.trace_file, "a", encoding="utf-8") as file_obj:
                for event in events:
                    file_obj.write(json.dumps(event, ensure_ascii=True) + "\n")
        except Exception:
            pass

    def _parse_item_id(self, raw_item_id: str) -> Tuple[str, int]:
        task_name, item_id = raw_item_id.split("_", 1)
        return task_name, int(item_id)

    def preprocess_prompt_to_rollout_handler(self, prompts: DataProto, n: int) -> List[RolloutHandler]:
        assert "raw_prompt" in prompts.non_tensor_batch.keys(), "raw_prompt is not in non_tensor_batch, need to set data.return_raw_chat=True"
        handler_list = []
        for i, raw_prompt in enumerate(prompts.non_tensor_batch["raw_prompt"]):
            task_name, item_id = self._parse_item_id(prompts.non_tensor_batch["item_id"][i])
            for _ in range(n):
                input_ids = _pre_process_inputs(self.pad_token_id, prompts.batch["input_ids"][i])
                attention_mask = _pre_process_inputs(0, prompts.batch["attention_mask"][i])
                position_ids = compute_position_id_with_mask(torch.tensor(attention_mask)).tolist()
                prompt_len = len(input_ids)
                handler = RolloutHandler(
                    messages=[Message(role=prompt["role"], content=prompt["content"]) for prompt in raw_prompt],
                    task_name=task_name,
                    item_id=item_id,
                    score=0.0,
                    done=False,
                    input_ids=list(input_ids),
                    prompt_ids=list(input_ids),
                    response_ids=[],
                    attention_mask=list(attention_mask),
                    prompt_attention_mask=list(attention_mask),
                    response_attention_mask=[],
                    position_ids=list(position_ids),
                    prompt_position_ids=list(position_ids),
                    response_position_ids=[],
                    loss_mask=[0] * prompt_len,
                    prompt_loss_mask=[0] * prompt_len,
                    response_loss_mask=[],
                    speaker_ids=[CONTROL_SPEAKER_ID] * prompt_len,
                    prompt_speaker_ids=[CONTROL_SPEAKER_ID] * prompt_len,
                    response_speaker_ids=[],
                    planner_response_mask=[0] * prompt_len,
                    prompt_planner_response_mask=[0] * prompt_len,
                    response_planner_response_mask=[],
                    executor_response_mask=[0] * prompt_len,
                    prompt_executor_response_mask=[0] * prompt_len,
                    response_executor_response_mask=[],
                    score_values=[0.0] * prompt_len,
                    prompt_score_values=[0.0] * prompt_len,
                    response_score_values=[],
                    reward_event_mask=[0] * prompt_len,
                    prompt_reward_event_mask=[0] * prompt_len,
                    response_reward_event_mask=[],
                    max_response_len=self.config.response_length,
                    max_model_len=min(self.config.max_model_len, self.config.prompt_length + self.config.response_length),
                )
                handler_list.append(handler)
        return handler_list

    def _extract_response_token_ids(self, output) -> List[List[int]]:
        token_ids = output[0].tolist()
        if len(token_ids) == 0:
            return []
        if isinstance(token_ids[0], int):
            return [token_ids]
        return token_ids

    def _sampling_overrides(self, role: str, do_sample: bool, extra_kwargs: dict) -> dict:
        overrides = {
            "max_tokens": self.role_defaults[role]["max_tokens"],
            "temperature": self.role_defaults[role]["temperature"],
        }
        if not do_sample:
            overrides.update(
                {
                    "best_of": 1,
                    "top_p": 1.0,
                    "top_k": -1,
                    "min_p": 0.0,
                    "temperature": 0.0,
                    "n": 1,
                }
            )
        overrides.update(extra_kwargs)
        return overrides

    def _retry_sampling_overrides(self, do_sample: bool, extra_kwargs: dict) -> dict:
        overrides = {
            "max_tokens": self.invalid_output_retry_max_tokens,
            "temperature": self.invalid_output_retry_temperature,
            "n": 1,
        }
        if not do_sample:
            overrides.update(
                {
                    "best_of": 1,
                    "top_p": 1.0,
                    "top_k": -1,
                    "min_p": 0.0,
                    "temperature": 0.0,
                    "n": 1,
                }
            )
        overrides.update(extra_kwargs)
        return overrides

    def _pad_tensor_list(self, tensors, pad_value, total_length, dtype=None):
        padded = pad_sequence(tensors, batch_first=True, padding_value=pad_value)
        if padded.shape[1] < total_length:
            padded = pad_sequence_to_length(padded, total_length, pad_value)
        if dtype is not None:
            padded = padded.to(dtype=dtype)
        return padded

    def _mark_timeout(self, handler: RolloutHandler) -> None:
        handler.timeout_occurred = 1.0
        handler.env_step_failed = 1.0
        handler.executor_action_valid = 0.0

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # ORIGINAL FLOW DIFFERENCE:
        # single-agent path generated once per env round.
        # multi-agent path generates planner then executor before env.step.
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        global_steps = prompts.meta_info.get("global_steps", None)
        max_rounds = prompts.meta_info.get("max_rounds", self.task_profile.default_max_rounds)
        cur_device = prompts.batch["input_ids"].device
        do_sample = prompts.meta_info.get("do_sample", True)

        base_batch_size = prompts.batch["input_ids"].size(0)
        batch_size = base_batch_size * self.config.n
        rollout_handler_ls = self.preprocess_prompt_to_rollout_handler(prompts, n=self.config.n)
        env_clients = [init_env_client(self.agentgym_config) for _ in range(batch_size)]
        previous_scores = [0.0] * batch_size
        time.sleep(self.config.send_interval)

        for idx, rollout_handler in enumerate(rollout_handler_ls):
            try:
                env_clients[idx].reset(rollout_handler.item_id)
                observation = env_clients[idx].observe()
                rollout_handler.latest_observation = observation
                rollout_handler.add_user_message(self.tokenizer, observation)
            except TimeoutError:
                rollout_handler.done = True
                self._mark_timeout(rollout_handler)
            except Exception as exc:
                rollout_handler.done = True
                rollout_handler.env_step_failed = 1.0
                rollout_handler.executor_action_valid = 0.0
                rollout_handler.latest_observation = str(exc)

        rounds = 0
        rollout_bar = tqdm(total=max_rounds, desc="Multi-agent rounds", disable=torch.distributed.get_rank() != 0)

        while rounds < max_rounds:
            active_indices = [idx for idx, handler in enumerate(rollout_handler_ls) if not handler.done]
            if not active_indices:
                break

            planner_prompt_ids = []
            for idx in active_indices:
                planner_turn_prompt = build_planner_turn_prompt(
                    rollout_handler_ls[idx].latest_observation,
                    self.task_profile,
                )
                rollout_handler_ls[idx].add_user_message(self.tokenizer, planner_turn_prompt)
                planner_prompt_ids.append(rollout_handler_ls[idx].get_generation_prompt(self.tokenizer))

            rollout_bar.set_description(
                f"Multi-agent rounds {rounds + 1}/{max_rounds} | Active episodes per gpu: {len(active_indices)}"
            )
            with self.update_sampling_params(**self._sampling_overrides("planner", do_sample, kwargs.copy())):
                planner_output = self.inference_engine.generate(
                    prompts=None,
                    prompt_token_ids=planner_prompt_ids,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )
            planner_response_ids = self._extract_response_token_ids(planner_output)

            executor_prompt_ids = []
            executor_prompt_texts = []
            for local_i, idx in enumerate(active_indices):
                planner_text = self.tokenizer.decode(planner_response_ids[local_i], skip_special_tokens=True).strip()
                planner_context_text = f"[PLANNER]\n{planner_text}"
                rollout_handler_ls[idx].latest_planner_message = planner_context_text
                rollout_handler_ls[idx].add_assistant_message(self.tokenizer, planner_context_text, PLANNER_SPEAKER_ID)
                executor_turn_prompt = build_executor_turn_prompt(
                    rollout_handler_ls[idx].latest_observation,
                    planner_context_text,
                    self.task_profile,
                )
                rollout_handler_ls[idx].add_user_message(self.tokenizer, executor_turn_prompt)
                executor_prompt_texts.append(executor_turn_prompt)
                executor_prompt_ids.append(rollout_handler_ls[idx].get_generation_prompt(self.tokenizer))

            with self.update_sampling_params(**self._sampling_overrides("executor", do_sample, kwargs.copy())):
                executor_output = self.inference_engine.generate(
                    prompts=None,
                    prompt_token_ids=executor_prompt_ids,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )
            executor_response_ids = self._extract_response_token_ids(executor_output)
            time.sleep(self.config.send_interval)

            executor_results: List[Dict[str, Any]] = []
            for local_i, idx in enumerate(active_indices):
                handler = rollout_handler_ls[idx]
                raw_content = self.tokenizer.decode(executor_response_ids[local_i], skip_special_tokens=True)
                content = normalize_executor_payload(raw_content, self.task_profile)
                validation = validate_executor_payload(content, handler.latest_observation, self.task_profile)
                available_actions = extract_available_actions_from_observation(handler.latest_observation)
                initial_prompt_text = executor_prompt_texts[local_i]
                last_retry_prompt = ""
                retry_count_total = 0
                resolved_after_retry = False

                executor_context_text = f"[EXECUTOR]\n{content}"
                handler.add_assistant_message(self.tokenizer, executor_context_text, EXECUTOR_SPEAKER_ID)

                should_retry = (
                    self.task_profile.task_name == "babyai"
                    and not validation.valid
                    and self.invalid_output_policy == "terminate_with_penalty"
                    and self.invalid_output_max_retries > 0
                )
                while should_retry and retry_count_total < self.invalid_output_max_retries:
                    retry_count_total += 1
                    last_retry_prompt = build_executor_retry_prompt(
                        observation=handler.latest_observation,
                        planner_message=handler.latest_planner_message,
                        invalid_executor_output=content,
                        validation_reason=validation.reason,
                        task_profile=self.task_profile,
                    )
                    handler.add_user_message(self.tokenizer, last_retry_prompt)
                    retry_prompt_ids = [handler.get_generation_prompt(self.tokenizer)]
                    with self.update_sampling_params(**self._retry_sampling_overrides(do_sample, kwargs.copy())):
                        retry_output = self.inference_engine.generate(
                            prompts=None,
                            prompt_token_ids=retry_prompt_ids,
                            sampling_params=self.sampling_params,
                            use_tqdm=False,
                        )
                    retry_response_ids = self._extract_response_token_ids(retry_output)
                    if len(retry_response_ids) == 0:
                        raw_content = ""
                    else:
                        raw_content = self.tokenizer.decode(retry_response_ids[0], skip_special_tokens=True)
                    content = normalize_executor_payload(raw_content, self.task_profile)
                    validation = validate_executor_payload(content, handler.latest_observation, self.task_profile)
                    available_actions = extract_available_actions_from_observation(handler.latest_observation)
                    executor_context_text = f"[EXECUTOR]\n{content}"
                    handler.add_assistant_message(self.tokenizer, executor_context_text, EXECUTOR_SPEAKER_ID)
                    if validation.valid:
                        resolved_after_retry = True
                        break

                if not validation.valid and validation.reason == "invalid_format":
                    handler.executor_native_format_valid = 0.0
                    handler.executor_action_valid = 0.0
                elif not validation.valid and validation.reason == "action_not_in_available":
                    handler.executor_action_valid = 0.0
                handler.team_env_rounds += 1

                executor_results.append(
                    {
                        "idx": idx,
                        "raw_content": raw_content,
                        "content": content,
                        "validation": validation,
                        "available_actions": available_actions,
                        "initial_prompt_text": initial_prompt_text,
                        "last_retry_prompt": last_retry_prompt,
                        "retry_count_total": retry_count_total,
                        "resolved_after_retry": resolved_after_retry,
                    }
                )

            def agent_step(executor_result: Dict[str, Any]):
                idx = executor_result["idx"]
                handler = rollout_handler_ls[idx]
                raw_content = executor_result["raw_content"]
                content = executor_result["content"]
                validation = executor_result["validation"]
                trace_event: Dict[str, Any] = {
                    "item_id": handler.item_id,
                    "task_name": handler.task_name,
                    "round": rounds + 1,
                    "rank": self.rank,
                    "observation_excerpt": self._clip(handler.latest_observation),
                    "planner_message": self._clip(handler.latest_planner_message),
                    "executor_prompt": self._clip(executor_result["initial_prompt_text"]),
                    "executor_retry_prompt": self._clip(executor_result["last_retry_prompt"]),
                    "executor_raw_output": self._clip(raw_content),
                    "executor_normalized_output": self._clip(content),
                    "validation_valid": bool(validation.valid),
                    "validation_reason": validation.reason,
                    "extracted_action": validation.action,
                    "available_actions": executor_result["available_actions"],
                    "retry_attempt": executor_result["retry_count_total"],
                    "resolved_after_retry": bool(executor_result["resolved_after_retry"]),
                    "retry_count_total": executor_result["retry_count_total"],
                    "env_step_called": False,
                    "env_step_payload": "",
                    "env_reward": None,
                    "env_done": None,
                    "env_state_excerpt": "",
                    "exception": "",
                }

                terminate_for_invalid = (
                    self.task_profile.task_name == "babyai"
                    and not validation.valid
                    and self.invalid_output_policy == "terminate_with_penalty"
                )
                if terminate_for_invalid:
                    handler.mark_last_executor_reward(self.invalid_output_penalty)
                    handler.done = True
                    if validation.reason == "invalid_format":
                        handler.invalid_format_terminated = 1.0
                    else:
                        handler.invalid_action_terminated = 1.0
                    reason_msg = f"Terminated due to invalid executor output ({validation.reason})."
                    handler.latest_observation = reason_msg
                    trace_event["env_state_excerpt"] = self._clip(reason_msg)
                    try:
                        handler.add_user_message(self.tokenizer, reason_msg)
                    except Exception:
                        pass
                    return True, trace_event

                try:
                    trace_event["env_step_called"] = True
                    trace_event["env_step_payload"] = content
                    if self.trace_executor_payload:
                        logger.warning(
                            "[EXECUTOR TRACE] item_id=%s round=%s retries=%s raw=%s normalized=%s payload=%s parsed_action=%s",
                            handler.item_id,
                            rounds + 1,
                            executor_result["retry_count_total"],
                            self._clip(raw_content),
                            self._clip(content),
                            content,
                            validation.action,
                        )
                    step_output = env_clients[idx].step(content)
                    state = step_output.state
                    reward = float(step_output.reward)
                    done = bool(step_output.done)
                    trace_event["env_reward"] = reward
                    trace_event["env_done"] = done
                    trace_event["env_state_excerpt"] = self._clip(state)
                    if detect_invalid_action(state, self.task_profile):
                        handler.executor_action_valid = 0.0
                    delta_reward = 0.0
                    if self.reward_mode == "delta_per_executor_step":
                        delta_reward = compute_reward_delta(previous_scores[idx], reward)
                    handler.mark_last_executor_reward(delta_reward)
                    previous_scores[idx] = reward
                    handler.score = reward
                    handler.done = done
                    handler.latest_observation = state
                    handler.add_user_message(self.tokenizer, state)
                    return done, trace_event
                except TimeoutError:
                    self._mark_timeout(handler)
                    handler.done = True
                    trace_event["exception"] = "TimeoutError"
                    return True, trace_event
                except Exception as exc:
                    handler.env_step_failed = 1.0
                    handler.executor_action_valid = 0.0
                    handler.done = True
                    handler.latest_observation = str(exc)
                    trace_event["exception"] = self._clip(exc)
                    trace_event["env_state_excerpt"] = self._clip(exc)
                    try:
                        handler.add_user_message(self.tokenizer, str(exc))
                    except Exception:
                        pass
                    return True, trace_event

            with ThreadPoolExecutor(max_workers=len(active_indices)) as executor:
                step_results = list(executor.map(agent_step, executor_results))
            step_dones = [result[0] for result in step_results]
            step_trace_events = [result[1] for result in step_results]
            self._append_trace_events(step_trace_events)

            rounds += 1
            rollout_bar.update(1)
            if all(step_dones):
                break

        rollout_bar.close()

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
            team_env_rounds.append(handler.team_env_rounds)
            executor_action_valid.append(handler.executor_action_valid)
            executor_native_format_valid.append(handler.executor_native_format_valid)
            env_step_failed.append(handler.env_step_failed)
            timeout_occurred.append(handler.timeout_occurred)
            invalid_format_terminated.append(handler.invalid_format_terminated)
            invalid_action_terminated.append(handler.invalid_action_terminated)
            messages.append(handler.messages)

        response_ids = self._pad_tensor_list(response_ids, self.pad_token_id, self.config.response_length)
        response_attention_mask = self._pad_tensor_list(response_attention_mask, 0, self.config.response_length)
        response_loss_mask = self._pad_tensor_list(response_loss_mask, 0, self.config.response_length)
        response_speaker_ids = self._pad_tensor_list(response_speaker_ids, CONTROL_SPEAKER_ID, self.config.response_length)
        response_planner_mask = self._pad_tensor_list(response_planner_mask, 0, self.config.response_length)
        response_executor_mask = self._pad_tensor_list(response_executor_mask, 0, self.config.response_length)
        response_scores = self._pad_tensor_list(response_scores, 0.0, self.config.response_length, dtype=torch.float32)
        response_reward_event_mask = self._pad_tensor_list(response_reward_event_mask, 0, self.config.response_length)

        prompt_input_ids = prompts.batch["input_ids"].repeat_interleave(self.config.n, dim=0)
        prompt_attention_mask = prompts.batch["attention_mask"].repeat_interleave(self.config.n, dim=0)
        prompt_position_ids = prompts.batch["position_ids"].repeat_interleave(self.config.n, dim=0)
        prompt_speaker_ids = torch.zeros_like(prompt_input_ids, dtype=torch.int, device=cur_device)

        response_length = response_ids.size(1)
        delta_position_ids = torch.arange(1, response_length + 1, device=cur_device).unsqueeze(0).repeat(batch_size, 1)
        response_position_ids = prompt_position_ids[:, -1:] + delta_position_ids

        seq = torch.cat((prompt_input_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)
        speaker_ids = torch.cat((prompt_speaker_ids, response_speaker_ids), dim=-1)

        if global_steps and self.config.rollout_log_dir is not None:
            try:
                log_dir = os.path.join(self.config.rollout_log_dir, f"step{global_steps}")
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, f"{torch.distributed.get_rank()}.json")
                with open(log_path, "w", encoding="utf-8") as file_obj:
                    json.dump(
                        [
                            {
                                "item_id": rollout_handler_ls[idx].item_id,
                                "task_name": rollout_handler_ls[idx].task_name,
                                "conversations": [msg.to_dict() for msg in msgs],
                                "reward": rollout_handler_ls[idx].score,
                            }
                            for idx, msgs in enumerate(messages)
                        ],
                        file_obj,
                        ensure_ascii=True,
                        indent=2,
                    )
            except Exception:
                pass

        for client in env_clients:
            try:
                client.close()
            except Exception:
                pass

        batch = TensorDict(
            {
                "prompts": prompt_input_ids,
                "responses": response_ids,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "response_mask": response_loss_mask,
                "scores": response_scores,
                "task_rounds": torch.tensor(team_env_rounds, dtype=torch.float32, device=cur_device),
                "task_scores": response_scores,
                "speaker_ids": speaker_ids,
                "planner_response_mask": response_planner_mask,
                "executor_response_mask": response_executor_mask,
                "reward_event_mask": response_reward_event_mask,
                "team_env_rounds": torch.tensor(team_env_rounds, dtype=torch.float32, device=cur_device),
                "executor_action_valid": torch.tensor(executor_action_valid, dtype=torch.float32, device=cur_device),
                "executor_native_format_valid": torch.tensor(executor_native_format_valid, dtype=torch.float32, device=cur_device),
                "env_step_failed": torch.tensor(env_step_failed, dtype=torch.float32, device=cur_device),
                "timeout_occurred": torch.tensor(timeout_occurred, dtype=torch.float32, device=cur_device),
                "invalid_format_terminated": torch.tensor(invalid_format_terminated, dtype=torch.float32, device=cur_device),
                "invalid_action_terminated": torch.tensor(invalid_action_terminated, dtype=torch.float32, device=cur_device),
            },
            batch_size=batch_size,
        )

        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)
