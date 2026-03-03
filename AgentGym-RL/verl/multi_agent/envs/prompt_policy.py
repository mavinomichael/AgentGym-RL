# Multi-agent extension.
# Derived from: /Users/mavinomichael/PycharmProjects/AgentGym-RL/AgentGym-RL/verl/utils/agent_dataset/rl_dataset.py
# Original file left untouched for comparison.

import re
from typing import List, Optional, Tuple

from .task_registry import TaskProfile

CONTROL_SPEAKER_ID = 0
PLANNER_SPEAKER_ID = 1
EXECUTOR_SPEAKER_ID = 2
QWEN_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def build_multi_agent_instruction(base_instruction: str) -> str:
    return (
        "You are a two-agent team with a shared objective and shared reward.\n\n"
        "Roles:\n"
        "- Planner: analyze the latest environment state, reason about long-horizon progress, and send concise guidance to the Executor. Never issue an environment action.\n"
        "- Executor: follow the Planner's guidance and produce exactly one next environment response in the original task format.\n\n"
        "Rules:\n"
        "- Only the Executor may interact with the environment.\n"
        "- Planner and Executor cooperate through explicit natural-language communication.\n"
        "- The Executor must respond in the original task format exactly.\n"
        "- Do not prepend role labels like Planner: or Executor: to environment-facing outputs.\n\n"
        "Original task instructions:\n"
        f"{base_instruction}"
    )


def build_multi_agent_bootstrap(env_client, task_profile: Optional[TaskProfile] = None) -> Tuple[List[dict], str]:
    base_instruction = env_client.conversation_start[0]["value"]
    assistant_ack = env_client.conversation_start[1]["value"]
    wrapped_instruction = build_multi_agent_instruction(base_instruction)
    messages = [
        {"role": "user", "content": wrapped_instruction},
        {"role": "assistant", "content": assistant_ack},
    ]
    prompt_with_chat_template = (
        f"<|im_start|>system\n{QWEN_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{wrapped_instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_ack}<|im_end|>"
    )
    return messages, prompt_with_chat_template


def build_planner_turn_prompt(observation: str, task_profile: Optional[TaskProfile] = None) -> str:
    return (
        "[Planner Turn]\n"
        "[Environment Observation]\n"
        f"{observation}\n\n"
        "Planner: provide concise strategic guidance for the next environment interaction.\n"
        "Do not emit an environment action.\n"
        "Focus on the next step while preserving long-horizon progress."
    )


def build_executor_turn_prompt(
    observation: str, planner_message: str, task_profile: Optional[TaskProfile] = None
) -> str:
    format_hint = (
        task_profile.executor_native_format_hint
        if task_profile is not None
        else "Use the original task-native format exactly."
    )
    return (
        "[Executor Turn]\n"
        "[Environment Observation]\n"
        f"{observation}\n\n"
        "[Latest Planner Message]\n"
        f"{planner_message}\n\n"
        "Produce exactly one next environment response in the original task-native format.\n"
        f"Format requirement: {format_hint}\n"
        "Do not prepend role labels like Planner: or Executor:."
    )


def normalize_executor_payload(raw_text: str, task_profile: TaskProfile) -> str:
    text = raw_text.strip()
    for suffix in ("</s>", "<|im_end|>", "<|endoftext|>"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].rstrip()
    return text


def _format_validators(task_name: str):
    validators = {
        "babyai": lambda text: bool(re.search(r"Thought:\s*.*Action:\s*.+", text, re.DOTALL)),
        "textcraft": lambda text: bool(re.search(r"Thought:\s*.*Action:\s*.+", text, re.DOTALL)),
        "searchqa": lambda text: bool(re.match(r"\s*<(think|search|information|answer)>.*", text, re.DOTALL)),
        "sciworld": lambda text: bool(text.strip()),
        "webarena": lambda text: "```" in text,
    }
    return validators.get(task_name, lambda text: bool(text.strip()))


def is_executor_payload_valid(raw_text: str, task_profile: TaskProfile) -> bool:
    validator = _format_validators(task_profile.task_name)
    return validator(normalize_executor_payload(raw_text, task_profile))


def detect_invalid_action(observation: str, task_profile: TaskProfile) -> bool:
    text = observation if isinstance(observation, str) else str(observation)
    return any(pattern in text for pattern in task_profile.invalid_action_patterns)


def compute_reward_delta(previous_score: float, current_score: float) -> float:
    return float(current_score) - float(previous_score)
