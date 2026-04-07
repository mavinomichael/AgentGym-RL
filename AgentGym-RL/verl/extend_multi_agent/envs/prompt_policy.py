from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..json_protocol import (
    build_executor_schema_example,
    build_planner_schema_example,
    canonical_json,
    extract_available_actions,
    normalize_json_candidate,
    render_executor_env_action,
    validate_executor_json,
    validate_planner_json,
)
from .task_registry import TaskProfile

CONTROL_SPEAKER_ID = 0
PLANNER_SPEAKER_ID = 1
EXECUTOR_SPEAKER_ID = 2
PLANNER_REVIEWER_SPEAKER_ID = 3
EXECUTOR_REVIEWER_SPEAKER_ID = 4
SPEAKER_ID_TO_ROLE = {
    CONTROL_SPEAKER_ID: "control",
    PLANNER_SPEAKER_ID: "planner",
    EXECUTOR_SPEAKER_ID: "executor",
    PLANNER_REVIEWER_SPEAKER_ID: "planner_reviewer",
    EXECUTOR_REVIEWER_SPEAKER_ID: "executor_reviewer",
}
ROLE_TO_SPEAKER_ID = {role: speaker_id for speaker_id, role in SPEAKER_ID_TO_ROLE.items()}
QWEN_SYSTEM_PROMPT = (
    "You are a BabyAI planner-executor assistant. "
    "Follow the requested JSON schema exactly. "
    "Do not emit prose outside the required JSON object."
)


@dataclass(frozen=True)
class ExecutorPayloadValidation:
    valid: bool
    reason: str
    action: Optional[str] = None


@dataclass(frozen=True)
class PlannerPayloadValidation:
    valid: bool
    reason: str
    message: Optional[str] = None
    exact_action: bool = False
    degenerate_fragment: bool = False
    token_count: int = 0


@dataclass(frozen=True)
class PlannerReviewerDecision:
    valid: bool
    verdict: str
    reason: str
    reviewed_plan: Optional[str] = None


@dataclass(frozen=True)
class ExecutorReviewerDecision:
    valid: bool
    verdict: str
    reason: str


def build_multi_agent_instruction(base_instruction: str, task_profile: Optional[TaskProfile] = None) -> str:
    return (
        "You are part of a two-role BabyAI team.\n"
        "Planner outputs strict planner JSON only.\n"
        "Executor outputs strict executor JSON only.\n"
        "Only the executor action is adapted into the environment action.\n"
        "Do not emit Thought/Action prose.\n"
        "Do not explain your answer.\n"
        "Emit one JSON object only when asked for Planner or Executor output."
    )


def build_multi_agent_bootstrap(env_client, task_profile: Optional[TaskProfile] = None) -> Tuple[List[dict], str]:
    base_instruction = ""
    assistant_ack = "Understood. I will return JSON only."
    wrapped_instruction = build_multi_agent_instruction(base_instruction, task_profile=task_profile)
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


def _format_available_actions_block(observation: str) -> str:
    actions = extract_available_actions_from_observation(observation)
    if not actions:
        return "- unavailable"
    return "\n".join(f"- {action}" for action in actions)


def build_planner_turn_prompt(observation: str, task_profile: Optional[TaskProfile] = None) -> str:
    available_actions_block = _format_available_actions_block(observation)
    return (
        "Planner role. Return one JSON object only.\n"
        f"Schema: {build_planner_schema_example()}\n"
        "Use exactly the keys subgoal, constraints, done.\n"
        "Keep subgoal short, keep constraints short, and stay grounded.\n\n"
        f"Observation:\n{observation}\n\n"
        f"Available actions:\n{available_actions_block}\n"
    )


def build_planner_retry_prompt(
    observation: str,
    invalid_planner_output: str,
    validation_reason: str,
    task_profile: Optional[TaskProfile] = None,
) -> str:
    return (
        "Your previous planner JSON was invalid. Return JSON only.\n"
        f"Failure reason: {validation_reason}\n\n"
        f"Observation:\n{observation}\n\n"
        f"Previous invalid planner output:\n{invalid_planner_output}\n\n"
        "Use the exact planner schema again."
    )


def build_long_planner_turn_prompt(observation: str, task_profile: Optional[TaskProfile] = None) -> str:
    return build_planner_turn_prompt(observation, task_profile)


def build_long_planner_retry_prompt(
    observation: str,
    invalid_planner_output: str,
    reviewer_reason: str,
    retry_count: int,
    task_profile: Optional[TaskProfile] = None,
) -> str:
    return build_planner_retry_prompt(observation, invalid_planner_output, reviewer_reason, task_profile)


def build_planner_reviewer_prompt(
    observation: str,
    planner_draft: str,
    retry_count: int = 0,
    allow_repair: bool = False,
    review_reason: Optional[str] = None,
    task_profile: Optional[TaskProfile] = None,
) -> str:
    return "Planner reviewers are disabled in extend_multi_agent BabyAI v1."


def build_executor_turn_prompt(observation: str, planner_message: str, task_profile: Optional[TaskProfile] = None) -> str:
    available_actions_block = _format_available_actions_block(observation)
    default_action = extract_available_actions_from_observation(observation)
    example_action = default_action[0] if default_action else "turn left"
    return (
        "Executor role. Return one JSON object only.\n"
        f"Schema: {build_executor_schema_example(example_action)}\n"
        "Use exactly the keys action, action_args, done.\n"
        "action must exactly match one available action.\n"
        "action_args must be null or {}.\n"
        "Use planner JSON only as advisory context.\n\n"
        f"Observation:\n{observation}\n\n"
        f"Planner JSON:\n{planner_message}\n\n"
        f"Available actions:\n{available_actions_block}\n"
    )


def build_executor_retry_prompt(
    observation: str,
    planner_message: str,
    invalid_executor_output: str,
    validation_reason: str,
    task_profile: Optional[TaskProfile] = None,
) -> str:
    return (
        "Your previous executor JSON was invalid. Return JSON only.\n"
        f"Failure reason: {validation_reason}\n\n"
        f"Observation:\n{observation}\n\n"
        f"Planner JSON:\n{planner_message}\n\n"
        f"Previous invalid executor output:\n{invalid_executor_output}\n\n"
        "Use the exact executor schema again."
    )


def build_executor_reviewer_prompt(
    observation: str,
    reviewed_plan: str,
    executor_output: str,
    retry_count: int = 0,
    review_reason: Optional[str] = None,
    task_profile: Optional[TaskProfile] = None,
) -> str:
    return "Executor reviewers are disabled in extend_multi_agent BabyAI v1."


def planner_fallback_message() -> str:
    return canonical_json({"subgoal": "explore nearby state", "constraints": ["use one available action"], "done": False})


def normalize_executor_payload(raw_text: str, task_profile: TaskProfile) -> str:
    return normalize_json_candidate(raw_text)


def normalize_planner_payload(raw_text: str) -> str:
    return normalize_json_candidate(raw_text)


def normalize_reviewer_payload(raw_text: str) -> str:
    return (raw_text or "").strip()


def rewrite_planner_payload(raw_text: str, observation: str, task_profile: Optional[TaskProfile] = None) -> Optional[str]:
    validation = validate_planner_payload(raw_text, observation=observation, task_profile=task_profile)
    return validation.message if validation.valid else None


def rewrite_reviewed_planner_plan(raw_text: str, observation: str, task_profile: Optional[TaskProfile] = None) -> Optional[str]:
    return rewrite_planner_payload(raw_text, observation, task_profile)


def validate_reviewed_planner_plan(raw_text: str, observation: str, task_profile: Optional[TaskProfile] = None) -> PlannerPayloadValidation:
    return validate_planner_payload(raw_text, observation=observation, task_profile=task_profile)


def validate_planner_payload(raw_text: str, observation: str = "", task_profile: Optional[TaskProfile] = None) -> PlannerPayloadValidation:
    result = validate_planner_json(raw_text)
    if not result.valid or result.payload is None:
        return PlannerPayloadValidation(valid=False, reason=result.reason, message=None, token_count=0)
    message = canonical_json(result.payload)
    return PlannerPayloadValidation(
        valid=True,
        reason=result.reason,
        message=message,
        exact_action=False,
        degenerate_fragment=False,
        token_count=len(message.split()),
    )


def extract_available_actions_from_observation(observation: str) -> List[str]:
    return extract_available_actions(observation)


def extract_executor_action(raw_text: str, task_profile: TaskProfile) -> Optional[str]:
    validation = validate_executor_json(raw_text, [])
    return validation.action if validation.valid else None


def is_executor_payload_valid(raw_text: str, task_profile: TaskProfile) -> bool:
    payload = normalize_executor_payload(raw_text, task_profile)
    validation = validate_executor_json(payload, [])
    return validation.valid


def validate_executor_payload(
    raw_text: str,
    observation: str,
    task_profile: TaskProfile,
    planner_message: Optional[str] = None,
) -> ExecutorPayloadValidation:
    legal_actions = extract_available_actions_from_observation(observation)
    validation = validate_executor_json(raw_text, legal_actions)
    return ExecutorPayloadValidation(valid=validation.valid, reason=validation.reason, action=validation.action)


def parse_planner_reviewer_output(raw_text: str, observation: str = "", task_profile: Optional[TaskProfile] = None) -> PlannerReviewerDecision:
    return PlannerReviewerDecision(valid=False, verdict="RETRY", reason="reviewers_disabled", reviewed_plan=None)


def parse_executor_reviewer_output(raw_text: str) -> ExecutorReviewerDecision:
    return ExecutorReviewerDecision(valid=False, verdict="RETRY", reason="reviewers_disabled")


def adapt_executor_payload_to_env_action(raw_text: str, observation: str) -> str:
    legal_actions = extract_available_actions_from_observation(observation)
    validation = validate_executor_json(raw_text, legal_actions)
    if not validation.valid or validation.payload is None:
        raise ValueError(f"Cannot adapt invalid executor JSON: {validation.reason}")
    return render_executor_env_action(validation.payload, legal_actions)


def detect_invalid_action(observation: str, task_profile: TaskProfile) -> bool:
    text = observation if isinstance(observation, str) else str(observation)
    return any(pattern in text for pattern in task_profile.invalid_action_patterns)


def compute_reward_delta(previous_score: float, current_score: float) -> float:
    return float(current_score) - float(previous_score)
