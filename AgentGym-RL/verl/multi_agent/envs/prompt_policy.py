# Multi-agent extension.
# Derived from: /Users/mavinomichael/PycharmProjects/AgentGym-RL/AgentGym-RL/verl/utils/agent_dataset/rl_dataset.py
# Original file left untouched for comparison.

import ast
from dataclasses import dataclass
import re
from typing import List, Optional, Tuple

from .task_registry import TaskProfile

CONTROL_SPEAKER_ID = 0
PLANNER_SPEAKER_ID = 1
EXECUTOR_SPEAKER_ID = 2
QWEN_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
PLANNER_MAX_WORDS = 12
PLANNER_MAX_CHARS = 96


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
        "Planner: provide exactly one short guidance sentence for the Executor.\n"
        "Do not emit an environment action.\n"
        "Do not output role labels or sections such as [PLANNER], [EXECUTOR], Planner:, Executor:, Thought:, or Action:.\n"
        "Output only the guidance sentence.\n"
        f"Keep it under {PLANNER_MAX_WORDS} words.\n"
        "Use an imperative sentence such as 'Turn left toward the red ball.'\n"
        "Focus only on the next step while preserving long-horizon progress."
    )


def build_planner_retry_prompt(
    observation: str,
    invalid_planner_output: str,
    validation_reason: str,
    task_profile: Optional[TaskProfile] = None,
) -> str:
    return (
        "[Planner Retry]\n"
        "Your previous planner message was invalid and will not be shown to the Executor.\n"
        f"Failure reason: {validation_reason}\n\n"
        "[Environment Observation]\n"
        f"{observation}\n\n"
        "[Previous Invalid Planner Output]\n"
        f"{invalid_planner_output}\n\n"
        "Respond again with exactly one short guidance sentence.\n"
        "Do not emit an environment action.\n"
        "Do not output role labels or sections such as [PLANNER], [EXECUTOR], Planner:, Executor:, Thought:, or Action:.\n"
        f"Keep it under {PLANNER_MAX_WORDS} words.\n"
        "Output only the guidance sentence."
    )


def build_executor_turn_prompt(
    observation: str, planner_message: str, task_profile: Optional[TaskProfile] = None
) -> str:
    format_hint = (
        task_profile.executor_native_format_hint
        if task_profile is not None
        else "Use the original task-native format exactly."
    )
    strict_executor_rules = ""
    if task_profile is not None and task_profile.task_name == "babyai":
        available_actions = extract_available_actions_from_observation(observation)
        action_list = ", ".join(available_actions) if available_actions else "(read from observation)"
        strict_executor_rules = (
            "\nBabyAI strict rules:\n"
            "- Your response must be exactly this structure:\n"
            "Thought:\n"
            "<one short sentence>\n\n"
            "Action:\n"
            "<exactly one action>\n"
            f"- Action must be exactly one of: {action_list}\n"
            "- Do not output bare words like Go, Up, Left, or Right.\n"
            "- Do not output multiple Action lines.\n"
            "- Do not copy prompt headers such as [Executor Response] or [Planner Message].\n"
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
        f"{strict_executor_rules}"
    )


def build_executor_retry_prompt(
    observation: str,
    planner_message: str,
    invalid_executor_output: str,
    validation_reason: str,
    task_profile: Optional[TaskProfile] = None,
) -> str:
    format_hint = (
        task_profile.executor_native_format_hint
        if task_profile is not None
        else "Use the original task-native format exactly."
    )
    if task_profile is not None and task_profile.task_name == "babyai":
        available_actions = extract_available_actions_from_observation(observation)
        action_list = ", ".join(available_actions) if available_actions else "(read from observation)"
        return (
            "[Executor Retry]\n"
            "Your previous response was invalid and was not sent to the environment.\n"
            f"Failure reason: {validation_reason}\n\n"
            "[Environment Observation]\n"
            f"{observation}\n\n"
            "[Latest Planner Message]\n"
            f"{planner_message}\n\n"
            "[Previous Invalid Executor Output]\n"
            f"{invalid_executor_output}\n\n"
            "Respond again with exactly one valid BabyAI response.\n"
            "Your response must be exactly this structure:\n"
            "Thought:\n"
            "<one short sentence>\n\n"
            "Action:\n"
            "<exactly one action>\n"
            f"Action must be exactly one of: {action_list}\n"
            "Do not output bare words like Go, Up, Left, or Right.\n"
            "Do not output multiple Action lines.\n"
            "Do not include role labels.\n"
            "Do not copy prompt headers such as [Executor Response] or [Planner Message]."
        )

    return (
        "[Executor Retry]\n"
        "Your previous response was invalid and was not sent to the environment.\n"
        f"Failure reason: {validation_reason}\n\n"
        "[Environment Observation]\n"
        f"{observation}\n\n"
        "[Latest Planner Message]\n"
        f"{planner_message}\n\n"
        "[Previous Invalid Executor Output]\n"
        f"{invalid_executor_output}\n\n"
        "Respond again with exactly one valid environment response.\n"
        f"Format requirement: {format_hint}\n"
        "Do not include role labels."
    )


def planner_fallback_message() -> str:
    return "Planner guidance unavailable. Infer the next step from the observation only."


def normalize_executor_payload(raw_text: str, task_profile: TaskProfile) -> str:
    text = raw_text.strip()
    for suffix in ("</s>", "<|im_end|>", "<|endoftext|>"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].rstrip()
    return _strip_control_headers(text)


def normalize_planner_payload(raw_text: str) -> str:
    text = raw_text.strip()
    for suffix in ("</s>", "<|im_end|>", "<|endoftext|>"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].rstrip()
    return _strip_control_headers(text)


def _strip_control_headers(text: str) -> str:
    header_pattern = re.compile(
        r"^(?:\[(?:executor response|planner message|executor|planner)\]\s*)+",
        re.IGNORECASE,
    )
    while True:
        updated = header_pattern.sub("", text).lstrip()
        if updated == text:
            break
        text = updated
    return text


def _normalize_action_text(action: str) -> str:
    action = re.sub(r"[^A-Za-z0-9, ]+", "", action)
    return " ".join(action.lower().split()).strip()


def _contains_disallowed_planner_tokens(text: str) -> bool:
    return bool(
        re.search(
            r"\[(planner|executor|pl|ex)\]|(?:^|\s)(planner|executor)\s*:|Thought:|Action:",
            text,
            re.IGNORECASE,
        )
    )


def _is_tag_only_planner_text(text: str) -> bool:
    stripped = re.sub(r"\[(planner|executor|pl|ex)\]", " ", text, flags=re.IGNORECASE)
    stripped = re.sub(r"[\s\W_]+", "", stripped)
    return stripped == ""


def validate_planner_payload(raw_text: str) -> PlannerPayloadValidation:
    normalized = normalize_planner_payload(raw_text)
    if not normalized:
        return PlannerPayloadValidation(valid=False, reason="empty", message=None)
    if _is_tag_only_planner_text(normalized):
        return PlannerPayloadValidation(valid=False, reason="tag_only", message=None)
    if _contains_disallowed_planner_tokens(normalized):
        return PlannerPayloadValidation(valid=False, reason="contains_role_or_schema_tokens", message=None)
    collapsed = " ".join(normalized.split())
    if len(collapsed) > PLANNER_MAX_CHARS or len(collapsed.split()) > PLANNER_MAX_WORDS:
        return PlannerPayloadValidation(valid=False, reason="too_long", message=None)
    return PlannerPayloadValidation(valid=True, reason="ok", message=collapsed)


def extract_available_actions_from_observation(observation: str) -> List[str]:
    text = observation if isinstance(observation, str) else str(observation)
    match = re.search(r"Available actions:\s*(\[[^\]]*\])", text, re.DOTALL)
    if not match:
        return []

    payload = match.group(1)
    actions: List[str] = []
    try:
        parsed = ast.literal_eval(payload)
        if isinstance(parsed, list):
            actions = [_normalize_action_text(str(item)) for item in parsed]
    except Exception:
        actions = [_normalize_action_text(item) for item in re.findall(r'"([^"]+)"', payload)]

    return [item for item in actions if item]


def extract_executor_action(raw_text: str, task_profile: TaskProfile) -> Optional[str]:
    text = normalize_executor_payload(raw_text, task_profile)
    action_matches = re.findall(r"Action:\s*(.*?)(?=\n|$)", text, re.DOTALL)
    if len(action_matches) != 1:
        return None
    action = _normalize_action_text(action_matches[0])
    return action or None


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


def validate_executor_payload(
    raw_text: str,
    observation: str,
    task_profile: TaskProfile,
) -> ExecutorPayloadValidation:
    normalized = normalize_executor_payload(raw_text, task_profile)
    valid_format = is_executor_payload_valid(normalized, task_profile)

    if task_profile.task_name != "babyai":
        if valid_format:
            return ExecutorPayloadValidation(valid=True, reason="ok")
        return ExecutorPayloadValidation(valid=False, reason="invalid_format")

    if not valid_format:
        return ExecutorPayloadValidation(valid=False, reason="invalid_format")

    action = extract_executor_action(normalized, task_profile)
    if not action:
        return ExecutorPayloadValidation(valid=False, reason="invalid_format")

    available_actions = extract_available_actions_from_observation(observation)
    if available_actions and action not in available_actions:
        return ExecutorPayloadValidation(
            valid=False,
            reason="action_not_in_available",
            action=action,
        )
    return ExecutorPayloadValidation(valid=True, reason="ok", action=action)


def detect_invalid_action(observation: str, task_profile: TaskProfile) -> bool:
    text = observation if isinstance(observation, str) else str(observation)
    return any(pattern in text for pattern in task_profile.invalid_action_patterns)


def compute_reward_delta(previous_score: float, current_score: float) -> float:
    return float(current_score) - float(previous_score)
