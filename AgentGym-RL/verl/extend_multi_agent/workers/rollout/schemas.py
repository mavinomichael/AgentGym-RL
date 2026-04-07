# Multi-agent extension.
# Derived from: /Users/mavinomichael/PycharmProjects/AgentGym-RL/AgentGym-RL/verl/workers/rollout/schemas.py
# Original file left untouched for comparison.

from typing import Any, List, Literal, Optional

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - fallback for lightweight test environments
    torch = None

try:
    from transformers import PreTrainedTokenizer
except ModuleNotFoundError:  # pragma: no cover - fallback for lightweight test environments
    PreTrainedTokenizer = Any

from .agent_vllm_rollout.planner_executor import (
    CONTROL_SPEAKER_ID,
    EXECUTOR_SPEAKER_ID,
    EXECUTOR_REVIEWER_SPEAKER_ID,
    PLANNER_SPEAKER_ID,
    PLANNER_REVIEWER_SPEAKER_ID,
    SPEAKER_ID_TO_ROLE,
)


def _pre_process_inputs(pad_token_id, prompt_token_ids: Any) -> List[int]:
    if torch is None:
        raise ModuleNotFoundError("torch is required to preprocess inputs")
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    return prompt_token_ids[non_pad_index:].tolist()


class Message:
    def __init__(
        self,
        role: str,
        content: str,
        speaker: Optional[str] = None,
        prompt_content: Optional[str] = None,
    ):
        self.role = role
        self.content = content
        self.speaker = speaker
        self.prompt_content = prompt_content if prompt_content is not None else content

    def to_dict(self):
        data = {"role": self.role, "content": self.content}
        if self.speaker is not None:
            data["speaker"] = self.speaker
        return data

    def to_prompt_dict(self):
        return {"role": self.role, "content": self.prompt_content}

    def __repr__(self):
        return str(self.to_dict())


class RolloutHandler:
    def __init__(
        self,
        messages: List[Message],
        task_name: str,
        item_id: int,
        score: float,
        done: bool,
        input_ids: List[int],
        prompt_ids: List[int],
        response_ids: List[int],
        attention_mask: List[int],
        prompt_attention_mask: List[int],
        response_attention_mask: List[int],
        position_ids: List[int],
        prompt_position_ids: List[int],
        response_position_ids: List[int],
        loss_mask: List[int],
        prompt_loss_mask: List[int],
        response_loss_mask: List[int],
        speaker_ids: List[int],
        prompt_speaker_ids: List[int],
        response_speaker_ids: List[int],
        planner_response_mask: List[int],
        prompt_planner_response_mask: List[int],
        response_planner_response_mask: List[int],
        executor_response_mask: List[int],
        prompt_executor_response_mask: List[int],
        response_executor_response_mask: List[int],
        score_values: List[float],
        prompt_score_values: List[float],
        response_score_values: List[float],
        reward_event_mask: List[int],
        prompt_reward_event_mask: List[int],
        response_reward_event_mask: List[int],
        planner_reviewer_response_mask: Optional[List[int]] = None,
        prompt_planner_reviewer_response_mask: Optional[List[int]] = None,
        response_planner_reviewer_response_mask: Optional[List[int]] = None,
        executor_reviewer_response_mask: Optional[List[int]] = None,
        prompt_executor_reviewer_response_mask: Optional[List[int]] = None,
        response_executor_reviewer_response_mask: Optional[List[int]] = None,
        ppo_loss_weights: Optional[List[float]] = None,
        prompt_ppo_loss_weights: Optional[List[float]] = None,
        response_ppo_loss_weights: Optional[List[float]] = None,
        kl_loss_weights: Optional[List[float]] = None,
        prompt_kl_loss_weights: Optional[List[float]] = None,
        response_kl_loss_weights: Optional[List[float]] = None,
        max_response_len: int = 8192,
        max_model_len: int = 32768,
    ):
        self.messages = messages
        self.task_name = task_name
        self.item_id = item_id
        self.score = score
        self.done = done
        self.input_ids = input_ids
        self.prompt_ids = prompt_ids
        self.response_ids = response_ids
        self.attention_mask = attention_mask
        self.prompt_attention_mask = prompt_attention_mask
        self.response_attention_mask = response_attention_mask
        self.position_ids = position_ids
        self.prompt_position_ids = prompt_position_ids
        self.response_position_ids = response_position_ids
        self.loss_mask = loss_mask
        self.prompt_loss_mask = prompt_loss_mask
        self.response_loss_mask = response_loss_mask
        self.speaker_ids = speaker_ids
        self.prompt_speaker_ids = prompt_speaker_ids
        self.response_speaker_ids = response_speaker_ids
        self.planner_response_mask = planner_response_mask
        self.prompt_planner_response_mask = prompt_planner_response_mask
        self.response_planner_response_mask = response_planner_response_mask
        self.planner_reviewer_response_mask = planner_reviewer_response_mask or [0] * len(input_ids)
        self.prompt_planner_reviewer_response_mask = (
            prompt_planner_reviewer_response_mask or [0] * len(prompt_ids)
        )
        self.response_planner_reviewer_response_mask = response_planner_reviewer_response_mask or []
        self.executor_response_mask = executor_response_mask
        self.prompt_executor_response_mask = prompt_executor_response_mask
        self.response_executor_response_mask = response_executor_response_mask
        self.executor_reviewer_response_mask = executor_reviewer_response_mask or [0] * len(input_ids)
        self.prompt_executor_reviewer_response_mask = (
            prompt_executor_reviewer_response_mask or [0] * len(prompt_ids)
        )
        self.response_executor_reviewer_response_mask = response_executor_reviewer_response_mask or []
        self.ppo_loss_weights = ppo_loss_weights or [0.0] * len(input_ids)
        self.prompt_ppo_loss_weights = prompt_ppo_loss_weights or [0.0] * len(prompt_ids)
        self.response_ppo_loss_weights = response_ppo_loss_weights or []
        self.kl_loss_weights = kl_loss_weights or [0.0] * len(input_ids)
        self.prompt_kl_loss_weights = prompt_kl_loss_weights or [0.0] * len(prompt_ids)
        self.response_kl_loss_weights = response_kl_loss_weights or []
        self.score_values = score_values
        self.prompt_score_values = prompt_score_values
        self.response_score_values = response_score_values
        self.reward_event_mask = reward_event_mask
        self.prompt_reward_event_mask = prompt_reward_event_mask
        self.response_reward_event_mask = response_reward_event_mask
        self.max_response_len = max_response_len
        self.max_model_len = max_model_len

        self.team_env_rounds = 0
        self.executor_action_valid = 1.0
        self.executor_native_format_valid = 1.0
        self.executor_first_pass_valid = 1.0
        self.executor_retry_used = 0.0
        self.executor_retry_resolved = 0.0
        self.executor_retry_count = 0.0
        self.executor_action_changed_after_retry = 0.0
        self.planner_output_valid = 1.0
        self.planner_fallback_used = 0.0
        self.planner_rewrite_used = 0.0
        self.planner_tag_only = 0.0
        self.planner_exact_action = 0.0
        self.planner_degenerate_fragment = 0.0
        self.planner_message_token_count = 0.0
        self.env_step_failed = 0.0
        self.timeout_occurred = 0.0
        self.invalid_format_terminated = 0.0
        self.invalid_action_terminated = 0.0
        self.last_executor_score_index = None
        self.last_planner_score_index = None
        self.latest_observation = ""
        self.latest_planner_message = ""
        self.latest_planner_context_used = ""
        self.latest_planner_context_source = "original"
        self.latest_planner_validation_reason = "ok"
        self.latest_planner_raw_output = ""
        self.latest_planner_normalized_output = ""
        self.latest_planner_prompt = ""
        self.planner_review_retry_count = 0.0
        self.planner_review_repair_count = 0.0
        self.executor_review_retry_count = 0.0
        self.executor_review_passed = 1.0
        self.format_config = {
            "qwen": {
                "assistant_prefix_msg": "\n<|im_start|>assistant\n",
                "assistant_suffix_msg": "<|im_end|>",
                "user_prefix_msg": "\n<|im_start|>user\n",
                "user_suffix_msg": "<|im_end|>",
            }
        }

    def get_generation_prompt(self, tokenizer: PreTrainedTokenizer) -> List[int]:
        conversations = [message.to_prompt_dict() for message in self.messages]
        return tokenizer.apply_chat_template(conversations, add_generation_prompt=True, tokenize=True)

    def _append_segment(
        self,
        append_token_ids: List[int],
        loss_mask: List[int],
        speaker_id: int,
        planner_mask: List[int],
        planner_reviewer_mask: List[int],
        executor_mask: List[int],
        executor_reviewer_mask: List[int],
        ppo_weight_mask: List[float],
        kl_weight_mask: List[float],
    ) -> None:
        self.input_ids += append_token_ids
        self.attention_mask += [1] * len(append_token_ids)
        last_position_id = self.position_ids[-1] if self.position_ids else -1
        self.position_ids += [last_position_id + idx for idx in range(1, len(append_token_ids) + 1)]
        self.loss_mask += loss_mask
        self.speaker_ids += [speaker_id] * len(append_token_ids)
        self.planner_response_mask += planner_mask
        self.planner_reviewer_response_mask += planner_reviewer_mask
        self.executor_response_mask += executor_mask
        self.executor_reviewer_response_mask += executor_reviewer_mask
        self.ppo_loss_weights += ppo_weight_mask
        self.kl_loss_weights += kl_weight_mask
        self.score_values += [0.0] * len(append_token_ids)
        self.reward_event_mask += [0] * len(append_token_ids)

    def add_user_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        prompt_content: Optional[str] = None,
        format: Literal["qwen"] = "qwen",
    ) -> None:
        prompt_content = prompt_content if prompt_content is not None else content
        self.messages.append(Message(role="user", content=content, prompt_content=prompt_content))
        prefix_msg = self.format_config[format]["user_prefix_msg"]
        suffix_msg = self.format_config[format]["user_suffix_msg"]
        prefix_token_ids = tokenizer.encode(prefix_msg, add_special_tokens=False)
        suffix_token_ids = tokenizer.encode(suffix_msg, add_special_tokens=False)
        content_token_ids = tokenizer.encode(prompt_content, add_special_tokens=False)
        if self.input_ids[-len(prefix_token_ids):] == prefix_token_ids:
            append_token_ids = content_token_ids
            local_loss_mask = [0] * len(content_token_ids)
        elif self.input_ids[-len(suffix_token_ids):] == suffix_token_ids:
            append_token_ids = prefix_token_ids + content_token_ids
            local_loss_mask = [0] * (len(prefix_token_ids) + len(content_token_ids))
        else:
            max_len = max(len(prefix_token_ids), len(suffix_token_ids))
            raise ValueError(f"Unsupported end of message format: {tokenizer.decode(self.input_ids[-max_len:])}")
        append_token_ids += suffix_token_ids
        local_loss_mask += [0] * len(suffix_token_ids)
        self._append_segment(
            append_token_ids=append_token_ids,
            loss_mask=local_loss_mask,
            speaker_id=CONTROL_SPEAKER_ID,
            planner_mask=[0] * len(append_token_ids),
            planner_reviewer_mask=[0] * len(append_token_ids),
            executor_mask=[0] * len(append_token_ids),
            executor_reviewer_mask=[0] * len(append_token_ids),
            ppo_weight_mask=[0.0] * len(append_token_ids),
            kl_weight_mask=[0.0] * len(append_token_ids),
        )

    def add_assistant_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        speaker_id: int,
        prompt_content: Optional[str] = None,
        trainable: bool = True,
        ppo_weight: float = 1.0,
        kl_weight: float = 1.0,
        format: Literal["qwen"] = "qwen",
    ) -> None:
        speaker = SPEAKER_ID_TO_ROLE.get(speaker_id, "assistant")
        prompt_content = prompt_content if prompt_content is not None else content
        self.messages.append(
            Message(role="assistant", content=content, speaker=speaker, prompt_content=prompt_content)
        )
        prefix_msg = self.format_config[format]["assistant_prefix_msg"]
        suffix_msg = self.format_config[format]["assistant_suffix_msg"]
        prefix_token_ids = tokenizer.encode(prefix_msg, add_special_tokens=False)
        suffix_token_ids = tokenizer.encode(suffix_msg, add_special_tokens=False)
        content_token_ids = tokenizer.encode(prompt_content, add_special_tokens=False)
        raw_content_token_ids = tokenizer.encode(content, add_special_tokens=False)
        prompt_prefix_token_len = 0
        if prompt_content != content:
            if content and prompt_content.endswith(content):
                prompt_prefix_token_len = max(0, len(content_token_ids) - len(raw_content_token_ids))
            elif not content:
                prompt_prefix_token_len = len(content_token_ids)
        if self.input_ids[-len(prefix_token_ids):] == prefix_token_ids:
            append_token_ids = content_token_ids
            target_mask = [0] * prompt_prefix_token_len + [1] * (len(content_token_ids) - prompt_prefix_token_len)
            local_loss_mask = target_mask if trainable else [0] * len(content_token_ids)
            role_mask = target_mask if trainable else [0] * len(content_token_ids)
        elif self.input_ids[-len(suffix_token_ids):] == suffix_token_ids:
            append_token_ids = prefix_token_ids + content_token_ids
            prefix_zeroes = [0] * (len(prefix_token_ids) + prompt_prefix_token_len)
            target_mask = prefix_zeroes + [1] * (len(content_token_ids) - prompt_prefix_token_len)
            local_loss_mask = target_mask if trainable else [0] * len(append_token_ids)
            role_mask = target_mask if trainable else [0] * len(append_token_ids)
        else:
            max_len = max(len(prefix_token_ids), len(suffix_token_ids))
            raise ValueError(f"Unsupported end of message format: {tokenizer.decode(self.input_ids[-max_len:])}")
        append_token_ids += suffix_token_ids
        suffix_mask = [1] * len(suffix_token_ids) if trainable else [0] * len(suffix_token_ids)
        local_loss_mask += suffix_mask
        role_mask += suffix_mask
        planner_mask = role_mask if speaker_id == PLANNER_SPEAKER_ID else [0] * len(role_mask)
        planner_reviewer_mask = role_mask if speaker_id == PLANNER_REVIEWER_SPEAKER_ID else [0] * len(role_mask)
        executor_mask = role_mask if speaker_id == EXECUTOR_SPEAKER_ID else [0] * len(role_mask)
        executor_reviewer_mask = role_mask if speaker_id == EXECUTOR_REVIEWER_SPEAKER_ID else [0] * len(role_mask)
        if trainable:
            ppo_weight_mask = [float(ppo_weight) if token else 0.0 for token in role_mask]
            kl_weight_mask = [float(kl_weight) if token else 0.0 for token in role_mask]
        else:
            ppo_weight_mask = [0.0] * len(role_mask)
            kl_weight_mask = [0.0] * len(role_mask)
        self._append_segment(
            append_token_ids=append_token_ids,
            loss_mask=local_loss_mask,
            speaker_id=speaker_id,
            planner_mask=planner_mask,
            planner_reviewer_mask=planner_reviewer_mask,
            executor_mask=executor_mask,
            executor_reviewer_mask=executor_reviewer_mask,
            ppo_weight_mask=ppo_weight_mask,
            kl_weight_mask=kl_weight_mask,
        )
        if speaker_id == PLANNER_SPEAKER_ID and trainable:
            self.last_planner_score_index = len(self.input_ids) - 1
        if speaker_id == EXECUTOR_SPEAKER_ID and trainable:
            self.last_executor_score_index = len(self.input_ids) - 1

    def mark_last_executor_reward(self, reward: float) -> None:
        if self.last_executor_score_index is None:
            return
        self.score_values[self.last_executor_score_index] += float(reward)
        self.reward_event_mask[self.last_executor_score_index] = 1

    def mark_last_planner_reward(self, reward: float) -> None:
        if self.last_planner_score_index is None:
            return
        self.score_values[self.last_planner_score_index] += float(reward)
        self.reward_event_mask[self.last_planner_score_index] = 1

    def truncate_output_ids(self) -> None:
        self.input_ids = self.input_ids[: self.max_model_len]
        self.attention_mask = self.attention_mask[: self.max_model_len]
        self.position_ids = self.position_ids[: self.max_model_len]
        self.loss_mask = self.loss_mask[: self.max_model_len]
        self.speaker_ids = self.speaker_ids[: self.max_model_len]
        self.planner_response_mask = self.planner_response_mask[: self.max_model_len]
        self.planner_reviewer_response_mask = self.planner_reviewer_response_mask[: self.max_model_len]
        self.executor_response_mask = self.executor_response_mask[: self.max_model_len]
        self.executor_reviewer_response_mask = self.executor_reviewer_response_mask[: self.max_model_len]
        self.ppo_loss_weights = self.ppo_loss_weights[: self.max_model_len]
        self.kl_loss_weights = self.kl_loss_weights[: self.max_model_len]
        self.score_values = self.score_values[: self.max_model_len]
        self.reward_event_mask = self.reward_event_mask[: self.max_model_len]

        prompt_len = len(self.prompt_ids)
        response_slice = slice(prompt_len, min(len(self.input_ids), prompt_len + self.max_response_len))
        self.response_ids = self.input_ids[response_slice]
        self.response_attention_mask = self.attention_mask[response_slice]
        self.response_position_ids = self.position_ids[response_slice]
        self.response_loss_mask = self.loss_mask[response_slice]
        self.response_speaker_ids = self.speaker_ids[response_slice]
        self.response_planner_response_mask = self.planner_response_mask[response_slice]
        self.response_planner_reviewer_response_mask = self.planner_reviewer_response_mask[response_slice]
        self.response_executor_response_mask = self.executor_response_mask[response_slice]
        self.response_executor_reviewer_response_mask = self.executor_reviewer_response_mask[response_slice]
        self.response_ppo_loss_weights = self.ppo_loss_weights[response_slice]
        self.response_kl_loss_weights = self.kl_loss_weights[response_slice]
        self.response_score_values = self.score_values[response_slice]
        self.response_reward_event_mask = self.reward_event_mask[response_slice]
