# Multi-agent extension.
# Derived from: /Users/mavinomichael/PycharmProjects/AgentGym-RL/AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py
# Original file left untouched for comparison.

import re
from typing import List, Tuple

CONTROL_SPEAKER_ID = 0
PLANNER_SPEAKER_ID = 1
EXECUTOR_SPEAKER_ID = 2
TEAM_ACK = "Understood. Planner and Executor will cooperate under a shared objective."


def build_multi_agent_instruction(base_instruction: str) -> str:
    return (
        "You are a two-agent team with a shared objective and shared reward.\n\n"
        "Roles:\n"
        "- Planner: analyze the latest environment state and send concise guidance. Never issue an environment action.\n"
        "- Executor: use the planner guidance and produce the single next environment action in the original task format.\n\n"
        "Rules:\n"
        "- Only the Executor may act on the environment.\n"
        "- Planner and Executor cooperate through explicit natural-language communication.\n"
        "- Optimize final task success.\n\n"
        "Original task instructions:\n"
        f"{base_instruction}"
    )


def build_multi_agent_bootstrap(env_client) -> Tuple[List[dict], str]:
    base_instruction = env_client.conversation_start[0]["value"]
    wrapped_instruction = build_multi_agent_instruction(base_instruction)
    messages = [
        {"role": "user", "content": wrapped_instruction},
        {"role": "assistant", "content": TEAM_ACK},
    ]
    prompt_with_chat_template = (
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{wrapped_instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{TEAM_ACK}<|im_end|>"
    )
    return messages, prompt_with_chat_template


def build_planner_turn_prompt(observation: str) -> str:
    return (
        "[Planner Turn]\n"
        "[Environment Observation]\n"
        f"{observation}\n\n"
        "Planner: give concise guidance to the Executor.\n"
        "Do not output an environment action.\n"
        'Prefix your reply with "Planner:".'
    )


def build_executor_turn_prompt(observation: str, planner_message: str) -> str:
    return (
        "[Executor Turn]\n"
        "[Environment Observation]\n"
        f"{observation}\n\n"
        "[Latest Planner Message]\n"
        f"{planner_message}\n\n"
        "Executor: produce exactly one next environment action using the original task format.\n"
        'Prefix your reply with "Executor:".'
    )


def strip_speaker_prefix(content: str, speaker: str) -> str:
    pattern = rf"^\s*{re.escape(speaker)}\s*:\s*"
    return re.sub(pattern, "", content, count=1).strip()


def compute_reward_delta(previous_score: float, current_score: float) -> float:
    return float(current_score) - float(previous_score)
