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
    PLANNER_SPEAKER_ID,
)


def _pre_process_inputs(pad_token_id, prompt_token_ids: Any) -> List[int]:
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class Message:
    def __init__(self, role: str, content: str, speaker: Optional[str] = None):
        self.role = role
        self.content = content
        self.speaker = speaker

    def to_dict(self):
        data = {"role": self.role, "content": self.content}
        if self.speaker is not None:
            data["speaker"] = self.speaker
        return data

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
        self.executor_response_mask = executor_response_mask
        self.prompt_executor_response_mask = prompt_executor_response_mask
        self.response_executor_response_mask = response_executor_response_mask
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
        self.last_executor_score_index = None
        self.latest_observation = ""
        self.latest_planner_message = ""
        self.format_config = {
            "qwen": {
                "assistat_prefix_msg": "\n<|im_start|>assistant\n",
                "assistat_suffix_msg": "<|im_end|>",
                "user_prefix_msg": "\n<|im_start|>user\n",
                "user_suffix_msg": "<|im_end|>",
            }
        }

    def get_generation_prompt(self, tokenizer: PreTrainedTokenizer) -> List[int]:
        conversations = [msg.to_dict() for msg in self.messages]
        cleaned = [{"role": msg["role"], "content": msg["content"]} for msg in conversations]
        return tokenizer.apply_chat_template(cleaned, add_generation_prompt=True, tokenize=True)

    def _append_segment(
        self,
        append_token_ids: List[int],
        loss_mask: List[int],
        speaker_id: int,
        planner_mask: List[int],
        executor_mask: List[int],
    ) -> None:
        self.input_ids += append_token_ids
        self.attention_mask += [1] * len(append_token_ids)
        last_position_id = self.position_ids[-1] if self.position_ids else -1
        self.position_ids += [last_position_id + idx for idx in range(1, len(append_token_ids) + 1)]
        self.loss_mask += loss_mask
        self.speaker_ids += [speaker_id] * len(append_token_ids)
        self.planner_response_mask += planner_mask
        self.executor_response_mask += executor_mask
        self.score_values += [0.0] * len(append_token_ids)
        self.reward_event_mask += [0] * len(append_token_ids)
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask)
        assert len(self.speaker_ids) == len(self.input_ids)
        assert len(self.planner_response_mask) == len(self.input_ids)
        assert len(self.executor_response_mask) == len(self.input_ids)
        assert len(self.score_values) == len(self.input_ids)
        assert len(self.reward_event_mask) == len(self.input_ids)

    def add_user_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        format: Literal["qwen"] = "qwen",
    ) -> None:
        self.messages.append(Message(role="user", content=content))
        prefix_msg = self.format_config[format]["user_prefix_msg"]
        suffix_msg = self.format_config[format]["user_suffix_msg"]
        prefix_token_ids = tokenizer.encode(prefix_msg, add_special_tokens=False)
        suffix_token_ids = tokenizer.encode(suffix_msg, add_special_tokens=False)
        content_token_ids = tokenizer.encode(content, add_special_tokens=False)
        if self.input_ids[-len(prefix_token_ids):] == prefix_token_ids:
            append_token_ids = content_token_ids
            local_loss_mask = [0] * len(content_token_ids)
        elif self.input_ids[-len(suffix_token_ids):] == suffix_token_ids:
            append_token_ids = prefix_token_ids + content_token_ids
            local_loss_mask = [0] * (len(prefix_token_ids) + len(content_token_ids))
        else:
            max_len = max(len(prefix_token_ids), len(suffix_token_ids))
            raise ValueError(
                f"Unsupported end of message format: {tokenizer.decode(self.input_ids[-max_len:])}"
            )
        append_token_ids += suffix_token_ids
        local_loss_mask += [0] * len(suffix_token_ids)
        self._append_segment(
            append_token_ids=append_token_ids,
            loss_mask=local_loss_mask,
            speaker_id=CONTROL_SPEAKER_ID,
            planner_mask=[0] * len(append_token_ids),
            executor_mask=[0] * len(append_token_ids),
        )

    def add_assistant_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        speaker_id: int,
        format: Literal["qwen"] = "qwen",
    ) -> None:
        speaker = "planner" if speaker_id == PLANNER_SPEAKER_ID else "executor"
        self.messages.append(Message(role="assistant", content=content, speaker=speaker))
        prefix_msg = self.format_config[format]["assistat_prefix_msg"]
        suffix_msg = self.format_config[format]["assistat_suffix_msg"]
        prefix_token_ids = tokenizer.encode(prefix_msg, add_special_tokens=False)
        suffix_token_ids = tokenizer.encode(suffix_msg, add_special_tokens=False)
        response = tokenizer.encode(content, add_special_tokens=False)
        if self.input_ids[-len(prefix_token_ids):] == prefix_token_ids:
            append_token_ids = response
            local_loss_mask = [1] * len(response)
            role_mask = [1] * len(response)
        elif self.input_ids[-len(suffix_token_ids):] == suffix_token_ids:
            append_token_ids = prefix_token_ids + response
            local_loss_mask = [0] * len(prefix_token_ids) + [1] * len(response)
            role_mask = [0] * len(prefix_token_ids) + [1] * len(response)
        else:
            max_len = max(len(prefix_token_ids), len(suffix_token_ids))
            raise ValueError(
                f"Unsupported end of message format: {tokenizer.decode(self.input_ids[-max_len:])}"
            )
        append_token_ids += suffix_token_ids
        local_loss_mask += [1] * len(suffix_token_ids)
        role_mask += [1] * len(suffix_token_ids)
        planner_mask = role_mask if speaker_id == PLANNER_SPEAKER_ID else [0] * len(role_mask)
        executor_mask = role_mask if speaker_id == EXECUTOR_SPEAKER_ID else [0] * len(role_mask)
        self._append_segment(
            append_token_ids=append_token_ids,
            loss_mask=local_loss_mask,
            speaker_id=speaker_id,
            planner_mask=planner_mask,
            executor_mask=executor_mask,
        )
        if speaker_id == EXECUTOR_SPEAKER_ID:
            self.last_executor_score_index = len(self.input_ids) - 1

    def mark_last_executor_reward(self, reward: float) -> None:
        if self.last_executor_score_index is None:
            return
        self.score_values[self.last_executor_score_index] += float(reward)
        self.reward_event_mask[self.last_executor_score_index] = 1

    def truncate_output_ids(self) -> None:
        self.input_ids = self.input_ids[: self.max_model_len]
        self.attention_mask = self.attention_mask[: self.max_model_len]
        self.position_ids = self.position_ids[: self.max_model_len]
        self.loss_mask = self.loss_mask[: self.max_model_len]
        self.speaker_ids = self.speaker_ids[: self.max_model_len]
        self.planner_response_mask = self.planner_response_mask[: self.max_model_len]
        self.executor_response_mask = self.executor_response_mask[: self.max_model_len]
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
        self.response_executor_response_mask = self.executor_response_mask[response_slice]
        self.response_score_values = self.score_values[response_slice]
        self.response_reward_event_mask = self.reward_event_mask[response_slice]
