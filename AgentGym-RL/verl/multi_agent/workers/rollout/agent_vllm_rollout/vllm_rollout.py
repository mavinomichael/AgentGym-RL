# Multi-agent extension.
# Derived from: /Users/mavinomichael/PycharmProjects/AgentGym-RL/AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py
# Original file left untouched for comparison.

from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
from typing import List

import torch
import torch.distributed
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from verl import DataProto
from verl.multi_agent.workers.rollout.agent_vllm_rollout.planner_executor import (
    CONTROL_SPEAKER_ID,
    EXECUTOR_SPEAKER_ID,
    PLANNER_SPEAKER_ID,
    build_executor_turn_prompt,
    build_planner_turn_prompt,
    compute_reward_delta,
    strip_speaker_prefix,
)
from verl.multi_agent.workers.rollout.schemas import Message, RolloutHandler, _pre_process_inputs
from verl.utils.agentgym.client import init_env_client
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import pad_sequence_to_length
from verl.workers.rollout.agent_vllm_rollout.vllm_rollout import vLLMRollout as BaseVLLMRollout


class vLLMRollout(BaseVLLMRollout):
    def __init__(self, actor_module: nn.Module, rollout_config: DictConfig, agentgym_config: DictConfig, tokenizer, model_hf_config, multi_agent_config=None, **kwargs):
        self.multi_agent_config = multi_agent_config
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
                "max_tokens": int(self._multi_agent_get("roles.planner.max_tokens", min(256, self.config.max_tokens))),
                "temperature": float(self._multi_agent_get("roles.planner.temperature", self.config.temperature)),
            },
            "executor": {
                "max_tokens": int(self._multi_agent_get("roles.executor.max_tokens", min(256, self.config.max_tokens))),
                "temperature": float(self._multi_agent_get("roles.executor.temperature", self.config.temperature)),
            },
        }
        self.reward_mode = self._multi_agent_get("reward_mode", "delta_per_executor_step")

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

    def preprocess_prompt_to_rollout_handler(self, prompts: DataProto, n: int) -> List[RolloutHandler]:
        assert "raw_prompt" in prompts.non_tensor_batch.keys(), "raw_prompt is not in non_tensor_batch, need to set data.return_raw_chat=True"
        handler_list = []
        for i, raw_prompt in enumerate(prompts.non_tensor_batch["raw_prompt"]):
            for _ in range(n):
                input_ids = _pre_process_inputs(self.pad_token_id, prompts.batch["input_ids"][i])
                attention_mask = _pre_process_inputs(0, prompts.batch["attention_mask"][i])
                position_ids = compute_position_id_with_mask(torch.tensor(attention_mask)).tolist()
                prompt_len = len(input_ids)
                handler = RolloutHandler(
                    messages=[Message(role=prompt["role"], content=prompt["content"]) for prompt in raw_prompt],
                    task_name=prompts.non_tensor_batch["item_id"][i].split("_")[0],
                    item_id=int(prompts.non_tensor_batch["item_id"][i].split("_")[-1]),
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
            overrides.update({
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0.0,
                "n": 1,
            })
        overrides.update(extra_kwargs)
        return overrides

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # ORIGINAL FLOW DIFFERENCE:
        # single-agent path generated once per env round.
        # multi-agent path generates planner then executor before env.step.
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        global_steps = prompts.meta_info.get("global_steps", None)
        max_rounds = prompts.meta_info.get("max_rounds", 10)
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
                task = env_clients[idx].observe()
                rollout_handler.latest_observation = task
                rollout_handler.add_user_message(self.tokenizer, task)
            except TimeoutError:
                rollout_handler.done = True
                rollout_handler.score = 0.0
                rollout_handler.executor_action_valid = 0.0

        task_rounds = [0] * batch_size
        rounds = 0
        rollout_bar = tqdm(total=max_rounds, desc="Multi-agent rounds", disable=torch.distributed.get_rank() != 0)

        while rounds < max_rounds:
            active_indices = [idx for idx, handler in enumerate(rollout_handler_ls) if not handler.done]
            if not active_indices:
                break

            planner_prompt_ids = []
            for idx in active_indices:
                planner_turn_prompt = build_planner_turn_prompt(rollout_handler_ls[idx].latest_observation)
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
            for local_i, idx in enumerate(active_indices):
                planner_text = self.tokenizer.decode(planner_response_ids[local_i], skip_special_tokens=True).strip()
                if not planner_text.startswith("Planner:"):
                    planner_text = f"Planner: {planner_text}".strip()
                rollout_handler_ls[idx].latest_planner_message = planner_text
                rollout_handler_ls[idx].add_assistant_message(self.tokenizer, planner_text, PLANNER_SPEAKER_ID)
                executor_turn_prompt = build_executor_turn_prompt(
                    rollout_handler_ls[idx].latest_observation,
                    planner_text,
                )
                rollout_handler_ls[idx].add_user_message(self.tokenizer, executor_turn_prompt)
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

            def agent_step(local_i, idx):
                raw_content = self.tokenizer.decode(executor_response_ids[local_i], skip_special_tokens=True).strip()
                content = raw_content if raw_content.startswith("Executor:") else f"Executor: {raw_content}".strip()
                rollout_handler_ls[idx].add_assistant_message(self.tokenizer, content, EXECUTOR_SPEAKER_ID)
                action_content = strip_speaker_prefix(content, "Executor")
                task_rounds[idx] += 1
                rollout_handler_ls[idx].team_env_rounds += 1
                try:
                    step_output = env_clients[idx].step(action_content)
                    if isinstance(step_output.state, str) and step_output.state.startswith("Invalid Action."):
                        rollout_handler_ls[idx].executor_action_valid = 0.0
                    delta_reward = 0.0
                    if self.reward_mode == "delta_per_executor_step":
                        delta_reward = compute_reward_delta(previous_scores[idx], step_output.reward)
                    rollout_handler_ls[idx].mark_last_executor_reward(delta_reward)
                    previous_scores[idx] = step_output.reward
                    rollout_handler_ls[idx].score = step_output.reward
                    rollout_handler_ls[idx].done = step_output.done
                    rollout_handler_ls[idx].latest_observation = step_output.state
                    rollout_handler_ls[idx].add_user_message(self.tokenizer, step_output.state)
                    return step_output.done
                except Exception:
                    rollout_handler_ls[idx].executor_action_valid = 0.0
                    rollout_handler_ls[idx].done = True
                    return True

            with ThreadPoolExecutor(max_workers=len(active_indices)) as executor:
                step_dones = list(executor.map(lambda args: agent_step(*args), [(i, idx) for i, idx in enumerate(active_indices)]))
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
        messages = []
        prompt_speaker_ids = []

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
            messages.append(handler.messages)
            prompt_speaker_ids.append(torch.zeros_like(prompts.batch["input_ids"][0], device=cur_device, dtype=torch.int))

        def _pad_tensor_list(tensors, pad_value, total_length, dtype=None):
            padded = pad_sequence(tensors, batch_first=True, padding_value=pad_value)
            if padded.shape[1] < total_length:
                padded = pad_sequence_to_length(padded, total_length, pad_value)
            if dtype is not None:
                padded = padded.to(dtype=dtype)
            return padded

        response_ids = _pad_tensor_list(response_ids, self.pad_token_id, self.config.response_length)
        response_attention_mask = _pad_tensor_list(response_attention_mask, 0, self.config.response_length)
        response_loss_mask = _pad_tensor_list(response_loss_mask, 0, self.config.response_length)
        response_speaker_ids = _pad_tensor_list(response_speaker_ids, CONTROL_SPEAKER_ID, self.config.response_length)
        response_planner_mask = _pad_tensor_list(response_planner_mask, 0, self.config.response_length)
        response_executor_mask = _pad_tensor_list(response_executor_mask, 0, self.config.response_length)
        response_scores = _pad_tensor_list(response_scores, 0.0, self.config.response_length, dtype=torch.float32)
        response_reward_event_mask = _pad_tensor_list(response_reward_event_mask, 0, self.config.response_length)

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
                os.makedirs(os.path.join(self.config.rollout_log_dir, f"step{global_steps}"), exist_ok=True)
                with open(os.path.join(self.config.rollout_log_dir, f"step{global_steps}/{torch.distributed.get_rank()}.json"), "w") as file_obj:
                    json_msg = []
                    for idx, msgs in enumerate(messages):
                        json_msg.append(
                            {
                                "item_id": rollout_handler_ls[idx].item_id,
                                "conversations": [msg.to_dict() for msg in msgs],
                                "reward": rollout_handler_ls[idx].score,
                            }
                        )
                    json.dump(json_msg, file_obj, ensure_ascii=True, indent=4)
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
            },
            batch_size=batch_size,
        )

        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)
