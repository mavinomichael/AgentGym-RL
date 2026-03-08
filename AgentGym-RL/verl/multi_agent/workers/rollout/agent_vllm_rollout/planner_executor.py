# Multi-agent extension.
# Derived from: /Users/mavinomichael/PycharmProjects/AgentGym-RL/AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py
# Original file left untouched for comparison.

from verl.multi_agent.envs.prompt_policy import (
    CONTROL_SPEAKER_ID,
    EXECUTOR_SPEAKER_ID,
    PLANNER_SPEAKER_ID,
    ExecutorPayloadValidation,
    build_executor_turn_prompt,
    build_multi_agent_bootstrap,
    build_multi_agent_instruction,
    build_planner_turn_prompt,
    compute_reward_delta,
    detect_invalid_action,
    extract_available_actions_from_observation,
    extract_executor_action,
    is_executor_payload_valid,
    normalize_executor_payload,
    validate_executor_payload,
)
from verl.multi_agent.envs.task_registry import get_task_profile

__all__ = [
    "CONTROL_SPEAKER_ID",
    "EXECUTOR_SPEAKER_ID",
    "ExecutorPayloadValidation",
    "PLANNER_SPEAKER_ID",
    "build_executor_turn_prompt",
    "build_multi_agent_bootstrap",
    "build_multi_agent_instruction",
    "build_planner_turn_prompt",
    "compute_reward_delta",
    "detect_invalid_action",
    "extract_available_actions_from_observation",
    "extract_executor_action",
    "get_task_profile",
    "is_executor_payload_valid",
    "normalize_executor_payload",
    "validate_executor_payload",
]
