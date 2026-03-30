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
QWEN_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
PLANNER_MIN_WORDS = 2
PLANNER_MAX_WORDS = 6
PLANNER_MAX_CHARS = 48
PLANNER_FILLER_OPENERS = (
    "given that",
    "given",
    "since",
    "in this environment",
    "it seems",
    "you should",
    "you might",
    "because",
)
PLANNER_NON_IMPERATIVE_STARTERS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "there",
    "it",
    "you",
    "we",
    "i",
    "because",
    "given",
    "since",
    "in",
}
PLANNER_STATE_WORDS = {"closed", "locked", "open", "opened"}
PLANNER_DIRECTIONS = ("left", "right", "forward", "ahead", "front")
PLANNER_REVIEW_MAX_WORDS = 32
PLANNER_REVIEW_MAX_CHARS = 220
REVIEWER_JUNK_RE = re.compile(r"^[\W_!?.:,;|`~^=-]+$")


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


def build_multi_agent_instruction(
        base_instruction: str,
        task_profile: Optional[TaskProfile] = None,
) -> str:
    if task_profile is not None and task_profile.task_name == "babyai":
        return (
            "You are part of an exploration team that works together to finish every goal it is given.\n"
            "Work with the team and follow the role-specific instructions in the current prompt."
        )

    return (
        f"{base_instruction}\n\n"
        "You will now solve the same task as a cooperative multi-agent team with shared reward.\n"
        "Keep the original task instruction, action surface, and environment-facing response format unchanged.\n\n"
        "Team protocol:\n"
        "- Planner: send short non-environment-facing guidance to help with the next response.\n"
        "- Executor: produce the actual environment-facing response in the original single-agent format.\n"
        "- Reviewers, if present, only check drafts and do not change the task definition.\n\n"
        "Rules:\n"
        "- Only the Executor may emit the final environment-facing response.\n"
        "- Planner and reviewers must not emit the final environment response.\n"
        "- Do not prepend role labels like Planner: or Executor: to the environment-facing response."
    )


def build_multi_agent_bootstrap(env_client, task_profile: Optional[TaskProfile] = None) -> Tuple[List[dict], str]:
    base_instruction = env_client.conversation_start[0]["value"]
    assistant_ack = env_client.conversation_start[1]["value"]
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


def build_planner_turn_prompt(observation: str, task_profile: Optional[TaskProfile] = None) -> str:
    if task_profile is not None and task_profile.task_name == "babyai":
        available_actions_block = _format_available_actions_block(observation)
        return (
            "You are part of an exploration team, and you are the planner that wants to finish every goal you are given.\n\n"
            "Every round you will be given an observation, and you must reason about the situation and provide your reasoning, plan, hint, or suggestion based on the observation to help another agent finish the task.\n\n"
            "Another agent will be responsible for taking the final action. Your job is to help that agent choose the next useful move.\n\n"
            f"You can use the following actions:\n\n{available_actions_block}\n\n"
            "Guidance:\n"
            "- Stay grounded in the observation only.\n"
            "- Use the available actions to reason about what can be done next.\n"
            "- You may describe one action, several actions, or a short sequence if that is the best guidance.\n"
            "- You may mention exact environment actions if they are the clearest guidance.\n"
            "- Do not use the final response format.\n"
            "- Do not write brackets or role labels.\n"
            "- Keep the reasoning clear and useful for the executor.\n\n"
            f"Observation:\n{observation}\n\n"
            "Output only your reasoning, plan, hint, or suggestion."
        )

    return (
        f"{observation}\n\n"
        "You are the Planner.\n\n"
        "The original task instruction, action surface, and final response format remain unchanged.\n"
        "Help the Executor produce the next valid single-agent response.\n\n"
        "Output rules:\n"
        "- Output exactly one short guidance phrase.\n"
        f"- Output {PLANNER_MIN_WORDS} to {PLANNER_MAX_WORDS} words only.\n"
        "- Start with a verb.\n"
        "- Focus on the next useful direction, object, or constraint.\n"
        "- Stay grounded in the observation only.\n"
        "- Do not emit the final environment response.\n"
        "- Do not output the exact environment action.\n"
        "- Do not explain.\n"
        "- Do not use role labels, Thought:, or Action:.\n\n"
        "Output only the guidance phrase."
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
        "Respond again.\n\n"
        "Rules:\n"
        "- The original task instruction and action surface remain unchanged.\n"
        "- Output exactly one phrase.\n"
        f"- Output {PLANNER_MIN_WORDS} to {PLANNER_MAX_WORDS} words only.\n"
        "- Start with a verb.\n"
        "- Help the Executor produce the next valid single-agent response.\n"
        "- Give intent-level guidance, not the exact environment action.\n"
        "- No explanation.\n"
        "- No filler openers.\n"
        "- No role labels.\n"
        "- No Thought: or Action:.\n\n"
        "Good examples:\n"
        "- Approach the red key\n"
        "- Face the blue door\n"
        "- Search left side\n\n"
        "Bad examples:\n"
        "- turn right\n"
        "- move forward\n"
        "- Given that the key is nearby, you should move right first.\n"
        "- It seems you should explore.\n"
        "- Thought: turn right\n"
        "- Action: move forward\n\n"
        "Output only the phrase."
    )


def build_long_planner_turn_prompt(observation: str, task_profile: Optional[TaskProfile] = None) -> str:
    return (
        "[Planner Turn]\n"
        "[Environment Observation]\n"
        f"{observation}\n\n"
        "You are the Planner.\n\n"
        "The original task instruction, action surface, and final response format remain unchanged.\n"
        "Think through the next useful move so the Executor can produce the next valid single-agent response.\n"
        "You may use grounded natural-language reasoning, but keep it concise.\n\n"
        "Rules:\n"
        "- Stay grounded in the observation only.\n"
        "- Output at most 64 tokens.\n"
        "- Explain the next useful direction, focus, or target.\n"
        "- Do not emit the final environment response.\n"
        "- Do not emit exact BabyAI environment syntax.\n"
        "- Do not output role labels or sections such as [PLANNER], [EXECUTOR], Planner:, Executor:, Thought:, Action:.\n"
        "- Do not emit garbage characters or punctuation spam.\n"
        "- Mention only objects, doors, directions, or actions that are supported by the observation.\n\n"
        "Output only the planner draft."
    )


def build_long_planner_retry_prompt(
        observation: str,
        invalid_planner_output: str,
        reviewer_reason: str,
        retry_count: int,
        task_profile: Optional[TaskProfile] = None,
) -> str:
    return (
        "[Planner Retry]\n"
        f"Your previous planner draft was rejected by the Planner Reviewer.\n"
        f"Reviewer reason: {reviewer_reason}\n"
        f"Retry attempt: {retry_count}\n\n"
        "[Environment Observation]\n"
        f"{observation}\n\n"
        "[Previous Planner Draft]\n"
        f"{invalid_planner_output}\n\n"
        "Respond again.\n\n"
        "Rules:\n"
        "- The original task instruction and action surface remain unchanged.\n"
        "- Stay grounded in the observation only.\n"
        "- Output at most 64 tokens.\n"
        "- Give useful planner-level guidance for the next step.\n"
        "- Help the Executor produce the next valid single-agent response.\n"
        "- Do not emit exact BabyAI env actions.\n"
        "- Do not emit role labels, Thought:, or Action:.\n"
        "- Do not emit garbage characters or punctuation spam.\n"
        "- Prefer concrete, grounded guidance over generic filler.\n\n"
        "Output only the planner draft."
    )


def build_planner_reviewer_prompt(
        observation: str,
        planner_draft: str,
        retry_count: int = 0,
        allow_repair: bool = False,
        review_reason: Optional[str] = None,
        task_profile: Optional[TaskProfile] = None,
) -> str:
    retry_note = f"Planner retries already used: {retry_count}\n" if retry_count else ""
    repair_rule = (
        "- Use Verdict: REPAIR only when retries are exhausted and you can safely rewrite the planner draft.\n"
        if allow_repair
        else "- Do not use Verdict: REPAIR yet. Use PASS or RETRY only.\n"
    )
    review_reason_block = f"Previous review reason: {review_reason}\n\n" if review_reason else ""
    return (
        "[Planner Reviewer Turn]\n"
        "[Environment Observation]\n"
        f"{observation}\n\n"
        "[Planner Draft]\n"
        f"{planner_draft}\n\n"
        f"{review_reason_block}"
        "You are the Planner Reviewer.\n"
        "Judge whether the planner draft is grounded, non-garbage, and useful for helping the Executor produce the next valid single-agent response.\n\n"
        "Output exactly this schema:\n"
        "Verdict: PASS | RETRY | REPAIR\n"
        "Reason: <short reason>\n"
        "ReviewedPlan: <clean reviewed plan>\n\n"
        "Rules:\n"
        f"{retry_note}"
        "- ReviewedPlan is required for PASS and REPAIR.\n"
        "- ReviewedPlan must be concise and executor-oriented.\n"
        "- ReviewedPlan must be at most 32 words.\n"
        "- ReviewedPlan must not be empty, tag-only, garbage, Thought:, or Action:.\n"
        "- ReviewedPlan should not simply copy an exact BabyAI environment action.\n"
        "- If the planner draft contains garbage, planner tags, unsupported claims, or is not useful, use RETRY.\n"
        f"{repair_rule}"
        "Output only the schema."
    )


def build_executor_reviewer_prompt(
        observation: str,
        reviewed_plan: str,
        executor_output: str,
        retry_count: int = 0,
        review_reason: Optional[str] = None,
        task_profile: Optional[TaskProfile] = None,
) -> str:
    review_reason_block = f"Previous review reason: {review_reason}\n\n" if review_reason else ""
    return (
        "[Executor Reviewer Turn]\n"
        "[Environment Observation]\n"
        f"{observation}\n\n"
        "[Reviewed Planner Plan]\n"
        f"{reviewed_plan}\n\n"
        "[Executor Output]\n"
        f"{executor_output}\n\n"
        f"{review_reason_block}"
        "You are the Executor Reviewer.\n"
        "Judge whether the executor output matches the original single-agent task-native format and can be checked by deterministic validation.\n\n"
        "Output exactly this schema:\n"
        "Verdict: PASS | RETRY\n"
        "Reason: <short reason>\n\n"
        "Rules:\n"
        f"- Executor retries already used: {retry_count}\n"
        "- PASS only if the output looks properly structured and grounded in the observation.\n"
        "- Use RETRY when the output contains garbage, wrong format, multiple actions, role labels, or unsupported action text.\n"
        "- Do not repair or rewrite the executor output yourself.\n"
        "Output only the schema."
    )


def build_executor_turn_prompt(
        observation: str, planner_message: str, task_profile: Optional[TaskProfile] = None
) -> str:
    if task_profile is not None and task_profile.task_name == "babyai":
        available_actions_block = _format_available_actions_block(observation)
        return (
            "You are part of an exploration team, and you are the executor that wants to finish every goal you are given.\n\n"
            "Every round you will be given an observation, and you have to respond with an action and your thought based on the observation to finish the given task.\n\n"
            f"You can use the following actions:\n\n{available_actions_block}\n\n"
            "A planner agent has already reasoned about the task for you and provided this suggestion:\n\n"
            f"{planner_message}\n\n"
            "Use the planner suggestion if it is helpful. If it conflicts with the current observation or with the available actions, ignore it and follow the observation instead.\n\n"
            "Do not copy the planner suggestion verbatim as your final answer.\n\n"
            "Your response should use the following format:\n"
            "Thought:\n"
            "<Your Thought>\n\n"
            "Action:\n"
            "<Your Action>\n\n"
            f"Observation:\n{observation}\n"
            "Output exactly one Action line."
        )

    format_hint = (
        task_profile.executor_native_format_hint
        if task_profile is not None
        else "Use the original task-native format exactly."
    )
    executor_rules = ""
    if task_profile is not None and task_profile.task_name == "babyai":
        executor_rules = (
            "\nBabyAI reminder:\n"
            "- Keep the original single-agent structure:\n"
            "Thought:\n"
            "<brief thought>\n\n"
            "Action:\n"
            "<one action>\n"
            "- Output exactly one Action line.\n"
        )

    return (
        "[Executor Turn]\n"
        "[Environment Observation]\n"
        f"{observation}\n\n"
        "[Latest Planner Message]\n"
        f"{planner_message}\n\n"
        "You are the Executor.\n"
        "Produce the next environment response exactly as the original single-agent agent would.\n"
        "The original task instruction, action surface, and response format remain unchanged.\n"
        "Use the planner message as guidance only; if it conflicts with the observation, follow the observation and the original task instruction.\n"
        f"Format requirement: {format_hint}\n"
        "Do not prepend role labels like Planner: or Executor:."
        f"{executor_rules}"
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
        normalized_invalid = normalize_executor_payload(invalid_executor_output, task_profile)
        bare_action_hint = _normalize_action_text(normalized_invalid)
        keep_action_note = ""
        if bare_action_hint and bare_action_hint in available_actions:
            keep_action_note = (
                f"\nThe previous response already contains the valid action '{bare_action_hint}'. "
                "Keep that action if it still fits, but rewrite the full response in the required format.\n"
            )
        return (
            "Your previous response was invalid and was not sent to the environment.\n"
            f"Failure reason: {validation_reason}\n\n"
            f"Observation:\n{observation}\n\n"
            "A planner agent has already reasoned about the task for you and provided this suggestion:\n\n"
            f"{planner_message}\n\n"
            "Previous invalid response:\n"
            f"{invalid_executor_output}\n"
            f"{keep_action_note}\n"
            "Respond again with exactly one valid BabyAI response in the original single-agent format.\n"
            "Your response must be exactly this structure:\n"
            "Thought:\n"
            "<one short sentence>\n\n"
            "Action:\n"
            "<exactly one action>\n"
            f"Action must be exactly one of: {action_list}\n"
            "Do not repeat the planner suggestion verbatim.\n"
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


def normalize_reviewer_payload(raw_text: str) -> str:
    text = raw_text.strip()
    for suffix in ("</s>", "<|im_end|>", "<|endoftext|>"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].rstrip()
    return _strip_control_headers(text)


def rewrite_planner_payload(
        raw_text: str,
        observation: str,
        task_profile: Optional[TaskProfile] = None,
) -> Optional[str]:
    normalized = normalize_planner_payload(raw_text)
    if not normalized:
        return None

    available_actions = extract_available_actions_from_observation(observation)
    stripped_action = _strip_use_action_wrapper(normalized, available_actions)
    if stripped_action:
        return _validated_rewritten_planner_phrase(stripped_action, observation, task_profile)

    target_action = _planner_action_from_bare_target_label(normalized, available_actions)
    if target_action:
        return _validated_rewritten_planner_phrase(target_action, observation, task_profile)

    exact_action_rewrite = _planner_guidance_from_exact_action(normalized, available_actions)
    if exact_action_rewrite:
        return _validated_rewritten_planner_phrase(exact_action_rewrite, observation, task_profile)

    lowered = normalized.lower()
    target = _select_best_planner_target(lowered, available_actions)
    if target:
        phrase = f"Face {target}" if "door" in target.split() else f"Approach {target}"
        return _validated_rewritten_planner_phrase(phrase, observation, task_profile)

    if "check" in lowered or "action" in lowered or "option" in lowered:
        return _validated_rewritten_planner_phrase("Check options now", observation, task_profile)
    if re.search(r"\bleft\b", lowered):
        return _validated_rewritten_planner_phrase("Search left side", observation, task_profile)
    if re.search(r"\bright\b", lowered):
        return _validated_rewritten_planner_phrase("Search right side", observation, task_profile)
    if any(token in lowered for token in ("forward", "ahead", "front")):
        return _validated_rewritten_planner_phrase("Advance toward target", observation, task_profile)
    if any(token in lowered for token in ("search", "explore", "look")):
        return _validated_rewritten_planner_phrase("Search nearby objects", observation, task_profile)

    fallback_target = _select_best_planner_target("", available_actions)
    if fallback_target:
        phrase = f"Face {fallback_target}" if "door" in fallback_target.split() else f"Approach {fallback_target}"
        return _validated_rewritten_planner_phrase(phrase, observation, task_profile)
    if available_actions:
        return _validated_rewritten_planner_phrase("Check options now", observation, task_profile)
    return None


def rewrite_reviewed_planner_plan(
        raw_text: str,
        observation: str,
        task_profile: Optional[TaskProfile] = None,
) -> Optional[str]:
    rewritten = rewrite_planner_payload(raw_text, observation=observation, task_profile=task_profile)
    if rewritten:
        return rewritten
    fallback = planner_fallback_message()
    validation = validate_reviewed_planner_plan(
        fallback,
        observation=observation,
        task_profile=task_profile,
    )
    return validation.reviewed_plan if validation.valid else None


def _strip_control_headers(text: str) -> str:
    header_line_pattern = re.compile(r"^\s*(?:\[[^\[\]\n]+\]\s*)+\s*$")
    inline_header_pattern = re.compile(r"^(?:\s*\[[^\[\]\n]+\]\s*)+")
    while True:
        updated = text.lstrip()
        lines = updated.splitlines()
        while lines and (not lines[0].strip() or header_line_pattern.match(lines[0])):
            lines.pop(0)
        updated = "\n".join(lines).lstrip()
        updated = inline_header_pattern.sub("", updated).lstrip()
        if updated == text:
            break
        text = updated
    return text


def _format_available_actions_block(observation: str) -> str:
    available_actions = extract_available_actions_from_observation(observation)
    if not available_actions:
        return "- Use the available actions described in the observation."
    return "\n".join(f"- {action}" for action in available_actions)


def _planner_min_words(task_profile: Optional[TaskProfile]) -> int:
    if task_profile is not None and task_profile.task_name == "babyai":
        return 3
    return PLANNER_MIN_WORDS


def _planner_max_words(task_profile: Optional[TaskProfile]) -> int:
    if task_profile is not None and task_profile.task_name == "babyai":
        return 10
    return PLANNER_MAX_WORDS


def _planner_max_chars(task_profile: Optional[TaskProfile]) -> int:
    if task_profile is not None and task_profile.task_name == "babyai":
        return 80
    return PLANNER_MAX_CHARS


def _normalize_action_text(action: str) -> str:
    action = re.sub(r"[^A-Za-z0-9, ]+", "", action)
    return " ".join(action.lower().split()).strip()


def _contains_disallowed_planner_tokens(text: str) -> bool:
    return bool(
        re.search(
            r"\[(planner|executor|reviewer|pl|ex)\]|(?:^|\s)(planner|executor|reviewer)\s*:|Thought:|Action:",
            text,
            re.IGNORECASE,
        )
    )


def _is_tag_only_planner_text(text: str) -> bool:
    stripped = re.sub(r"\[(planner|executor|pl|ex)\]", " ", text, flags=re.IGNORECASE)
    stripped = re.sub(r"[\s\W_]+", "", stripped)
    return stripped == ""


def _collapse_planner_text(text: str) -> str:
    collapsed = " ".join(text.split())
    collapsed = re.sub(r"[.!?;,:\s]+$", "", collapsed)
    return collapsed.strip()


def _starts_with_filler_opener(text: str) -> bool:
    lowered = text.lower()
    return any(lowered.startswith(opener) for opener in PLANNER_FILLER_OPENERS)


def _starts_with_non_imperative_token(text: str) -> bool:
    first_token = re.findall(r"[A-Za-z]+", text.lower())
    if not first_token:
        return True
    return first_token[0] in PLANNER_NON_IMPERATIVE_STARTERS


def _clean_planner_target(target: str) -> str:
    words = [word for word in target.lower().split() if not word.isdigit()]
    words = [word for word in words if word not in PLANNER_STATE_WORDS]
    words = [word for word in words if word not in {"the", "a", "an"}]
    if not words:
        return ""
    return " ".join(words[:3])


def _candidate_planner_targets(available_actions: List[str]) -> List[str]:
    targets: List[str] = []
    for action in available_actions:
        target = ""
        if action.startswith("go to "):
            target = action[len("go to "):]
        elif action.startswith("pickup "):
            target = action[len("pickup "):]
        elif action.startswith("toggle and go through "):
            target = action[len("toggle and go through "):]
        if target:
            cleaned = _clean_planner_target(target)
            if cleaned and cleaned not in targets:
                targets.append(cleaned)
    return targets


def _select_best_planner_target(text: str, available_actions: List[str]) -> Optional[str]:
    candidates = _candidate_planner_targets(available_actions)
    if not candidates:
        return None
    text_tokens = set(re.findall(r"[a-z]+", text.lower()))
    best_target = None
    best_score = -1
    for candidate in candidates:
        candidate_tokens = set(candidate.split())
        score = len(candidate_tokens & text_tokens)
        if score > best_score:
            best_score = score
            best_target = candidate
    if best_score <= 0:
        return candidates[0]
    return best_target


def _planner_guidance_from_exact_action(text: str, available_actions: List[str]) -> Optional[str]:
    action = _normalize_action_text(text)
    if action not in available_actions:
        return None
    if action == "turn left":
        return "Search left side"
    if action == "turn right":
        return "Search right side"
    if action == "move forward":
        return "Advance toward target"
    if action == "check available actions":
        return "Check options now"
    if action.startswith("go to "):
        target = _clean_planner_target(action[len("go to "):])
        return f"Approach {target}" if target else None
    if action.startswith("pickup "):
        target = _clean_planner_target(action[len("pickup "):])
        return f"Approach {target}" if target else None
    if action.startswith("toggle and go through "):
        target = _clean_planner_target(action[len("toggle and go through "):])
        return f"Face {target}" if target else None
    return None


def _strip_use_action_wrapper(text: str, available_actions: List[str]) -> Optional[str]:
    normalized = _normalize_action_text(text)
    if not normalized.startswith("use "):
        return None
    candidate = normalized[len("use "):].strip()
    return candidate if candidate in available_actions else None


def _action_targets_with_preferences(available_actions: List[str]) -> List[Tuple[str, str]]:
    targets: List[Tuple[str, str]] = []
    prefixes = (
        "go to ",
        "pickup ",
        "toggle and go through ",
        "go through ",
    )
    for prefix in prefixes:
        for action in available_actions:
            if action.startswith(prefix):
                targets.append((action[len(prefix):].strip(), action))
    return targets


def _planner_action_from_bare_target_label(text: str, available_actions: List[str]) -> Optional[str]:
    normalized = _normalize_action_text(text)
    if not normalized:
        return None
    for target, action in _action_targets_with_preferences(available_actions):
        if normalized == target:
            return action
    return None


def _planner_degenerate_fragment_reason(text: str, available_actions: List[str]) -> Optional[str]:
    if _strip_use_action_wrapper(text, available_actions):
        return "use_action_wrapper"
    if _planner_action_from_bare_target_label(text, available_actions):
        return "bare_target_label"
    return None


def _validated_rewritten_planner_phrase(
        phrase: str,
        observation: str,
        task_profile: Optional[TaskProfile],
) -> Optional[str]:
    phrase = _collapse_planner_text(phrase)
    if not phrase:
        return None
    words = phrase.split()
    planner_max_words = _planner_max_words(task_profile)
    if len(words) > planner_max_words:
        phrase = " ".join(words[:planner_max_words])
    if phrase:
        phrase = phrase[0].upper() + phrase[1:]
    validation = validate_planner_payload(phrase, observation=observation, task_profile=task_profile)
    return validation.message if validation.valid else None


def _extract_labeled_value(text: str, label: str) -> Optional[str]:
    match = re.search(
        rf"{label}:\s*(.*?)(?=\n(?:Verdict|Reason|ReviewedPlan):|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    value = " ".join(match.group(1).strip().split())
    return value or None


def _looks_like_garbage(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if REVIEWER_JUNK_RE.match(stripped):
        return True
    alnum_count = len(re.findall(r"[A-Za-z0-9]", stripped))
    punct_count = len(re.findall(r"[^\w\s]", stripped))
    return alnum_count == 0 or punct_count > max(alnum_count * 2, 12)


def validate_reviewed_planner_plan(
        reviewed_plan: str,
        observation: str,
        task_profile: Optional[TaskProfile] = None,
) -> PlannerReviewerDecision:
    normalized = normalize_reviewer_payload(reviewed_plan)
    collapsed = _collapse_planner_text(normalized)
    if not collapsed:
        return PlannerReviewerDecision(valid=False, verdict="RETRY", reason="empty_reviewed_plan", reviewed_plan=None)
    if _looks_like_garbage(collapsed):
        return PlannerReviewerDecision(valid=False, verdict="RETRY", reason="garbage_reviewed_plan", reviewed_plan=None)
    if _contains_disallowed_planner_tokens(collapsed):
        return PlannerReviewerDecision(valid=False, verdict="RETRY", reason="contains_role_or_schema_tokens",
                                       reviewed_plan=None)
    words = collapsed.split()
    if len(words) > PLANNER_REVIEW_MAX_WORDS or len(collapsed) > PLANNER_REVIEW_MAX_CHARS:
        return PlannerReviewerDecision(valid=False, verdict="RETRY", reason="reviewed_plan_too_long",
                                       reviewed_plan=None)
    if task_profile is not None and task_profile.task_name == "babyai":
        available_actions = extract_available_actions_from_observation(observation)
        if _normalize_action_text(collapsed) in available_actions:
            return PlannerReviewerDecision(valid=False, verdict="RETRY", reason="exact_env_action", reviewed_plan=None)
    return PlannerReviewerDecision(valid=True, verdict="PASS", reason="ok", reviewed_plan=collapsed)


def parse_planner_reviewer_output(
        raw_text: str,
        observation: str,
        task_profile: Optional[TaskProfile] = None,
) -> PlannerReviewerDecision:
    normalized = normalize_reviewer_payload(raw_text)
    if not normalized or _looks_like_garbage(normalized):
        return PlannerReviewerDecision(valid=False, verdict="RETRY", reason="invalid_reviewer_schema",
                                       reviewed_plan=None)
    verdict_match = re.search(r"Verdict:\s*(PASS|RETRY|REPAIR)\b", normalized, re.IGNORECASE)
    if not verdict_match:
        return PlannerReviewerDecision(valid=False, verdict="RETRY", reason="invalid_reviewer_schema",
                                       reviewed_plan=None)
    verdict = verdict_match.group(1).upper()
    reason = _extract_labeled_value(normalized, "Reason") or "unspecified"
    reviewed_plan = _extract_labeled_value(normalized, "ReviewedPlan")
    if verdict in {"PASS", "REPAIR"}:
        if reviewed_plan is None:
            return PlannerReviewerDecision(valid=False, verdict="RETRY", reason="missing_reviewed_plan",
                                           reviewed_plan=None)
        validation = validate_reviewed_planner_plan(reviewed_plan, observation=observation, task_profile=task_profile)
        if not validation.valid:
            return PlannerReviewerDecision(valid=False, verdict="RETRY", reason=validation.reason, reviewed_plan=None)
        return PlannerReviewerDecision(valid=True, verdict=verdict, reason=reason,
                                       reviewed_plan=validation.reviewed_plan)
    return PlannerReviewerDecision(valid=True, verdict=verdict, reason=reason, reviewed_plan=None)


def parse_executor_reviewer_output(raw_text: str) -> ExecutorReviewerDecision:
    normalized = normalize_reviewer_payload(raw_text)
    if not normalized or _looks_like_garbage(normalized):
        return ExecutorReviewerDecision(valid=False, verdict="RETRY", reason="invalid_reviewer_schema")
    verdict_match = re.search(r"Verdict:\s*(PASS|RETRY)\b", normalized, re.IGNORECASE)
    if not verdict_match:
        return ExecutorReviewerDecision(valid=False, verdict="RETRY", reason="invalid_reviewer_schema")
    verdict = verdict_match.group(1).upper()
    reason = _extract_labeled_value(normalized, "Reason") or "unspecified"
    return ExecutorReviewerDecision(valid=True, verdict=verdict, reason=reason)


def validate_planner_payload(
        raw_text: str,
        observation: str = "",
        task_profile: Optional[TaskProfile] = None,
) -> PlannerPayloadValidation:
    if _is_tag_only_planner_text(raw_text):
        return PlannerPayloadValidation(valid=False, reason="tag_only", message=None)
    normalized = normalize_planner_payload(raw_text)
    if not normalized:
        return PlannerPayloadValidation(valid=False, reason="empty", message=None)
    if _contains_disallowed_planner_tokens(normalized):
        return PlannerPayloadValidation(valid=False, reason="contains_role_or_schema_tokens", message=None)
    if task_profile is not None and task_profile.task_name == "babyai" and re.search(r"\[[^\]]+\]", normalized):
        return PlannerPayloadValidation(valid=False, reason="contains_role_or_schema_tokens", message=None)
    collapsed = _collapse_planner_text(normalized)
    if _looks_like_garbage(collapsed):
        return PlannerPayloadValidation(valid=False, reason="garbage", message=None)
    if task_profile is not None and task_profile.task_name == "babyai":
        available_actions = extract_available_actions_from_observation(observation)
        token_count = len(collapsed.split())
        exact_action = _normalize_action_text(collapsed) in available_actions
        if exact_action:
            return PlannerPayloadValidation(
                valid=True,
                reason="ok",
                message=collapsed,
                exact_action=True,
                token_count=token_count,
            )
        degenerate_reason = _planner_degenerate_fragment_reason(collapsed, available_actions)
        if degenerate_reason is not None:
            return PlannerPayloadValidation(
                valid=False,
                reason=degenerate_reason,
                message=None,
                degenerate_fragment=True,
                token_count=token_count,
            )
        return PlannerPayloadValidation(valid=True, reason="ok", message=collapsed, token_count=token_count)
    if _starts_with_filler_opener(collapsed):
        return PlannerPayloadValidation(valid=False, reason="filler_opener", message=None)
    words = collapsed.split()
    planner_min_words = _planner_min_words(task_profile)
    planner_max_words = _planner_max_words(task_profile)
    planner_max_chars = _planner_max_chars(task_profile)
    if len(words) < planner_min_words:
        return PlannerPayloadValidation(valid=False, reason="not_intent_phrase", message=None)
    if len(collapsed) > planner_max_chars or len(words) > planner_max_words:
        return PlannerPayloadValidation(valid=False, reason="too_long", message=None)
    if _starts_with_non_imperative_token(collapsed):
        return PlannerPayloadValidation(valid=False, reason="not_intent_phrase", message=None)
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
        planner_message: Optional[str] = None,
) -> ExecutorPayloadValidation:
    normalized = normalize_executor_payload(raw_text, task_profile)
    valid_format = is_executor_payload_valid(normalized, task_profile)

    if task_profile.task_name != "babyai":
        if valid_format:
            return ExecutorPayloadValidation(valid=True, reason="ok")
        return ExecutorPayloadValidation(valid=False, reason="invalid_format")

    if not valid_format:
        if planner_message and _looks_like_copied_planner_text(normalized, planner_message):
            return ExecutorPayloadValidation(valid=False, reason="copied_planner_text")
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


def _canonicalize_text_for_copy_check(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"thought:\s*", " ", lowered)
    lowered = re.sub(r"action:\s*", " ", lowered)
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split()).strip()


def _looks_like_copied_planner_text(executor_text: str, planner_message: str) -> bool:
    executor_canonical = _canonicalize_text_for_copy_check(executor_text)
    planner_canonical = _canonicalize_text_for_copy_check(normalize_planner_payload(planner_message))
    if not executor_canonical or not planner_canonical:
        return False
    if executor_canonical == planner_canonical:
        return True
    if len(planner_canonical) < 12:
        return False
    return executor_canonical.startswith(planner_canonical) or planner_canonical.startswith(executor_canonical)


def detect_invalid_action(observation: str, task_profile: TaskProfile) -> bool:
    text = observation if isinstance(observation, str) else str(observation)
    return any(pattern in text for pattern in task_profile.invalid_action_patterns)


def compute_reward_delta(previous_score: float, current_score: float) -> float:
    return float(current_score) - float(previous_score)
