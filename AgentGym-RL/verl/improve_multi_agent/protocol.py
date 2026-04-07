from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

PLANNER_SCHEMA_VERSION = "v1"
EXECUTOR_SCHEMA_VERSION = "v1"

SUBGOAL_IDS = {
    "explore",
    "approach",
    "pickup",
    "open",
    "unlock",
    "drop",
    "finish",
}
OBJECT_TYPES = {"ball", "box", "key", "door", "goal", "none"}
OBJECT_COLORS = {"red", "green", "blue", "yellow", "grey", "purple", "none"}
LOCATION_HINTS = {"front", "left", "right", "current_room", "adjacent_room", "unknown"}
ACTION_HINTS = {"turn_left", "turn_right", "move_forward", "pickup", "toggle", "drop", "done"}
DISALLOWED_PLANNER_TOKENS = {
    "thought:",
    "action:",
    "let's think",
    "in summary",
    "```",
    "[planner",
    "[executor",
}
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
AVAILABLE_ACTIONS_RE = re.compile(r"Available actions:\s*(\[[^\]]*\])", re.DOTALL)
VISIBLE_OBJECT_RE = re.compile(
    r"(red|green|blue|yellow|grey|purple)\s+(ball|box|key|door|goal)\s+\d+",
    re.IGNORECASE,
)
STEP_COUNT_RE = re.compile(r"(\d+)\s+steps?\s+in front of you", re.IGNORECASE)
JSON_FIELD_RE = re.compile(r'"(?P<key>[^"]+)"\s*:\s*"(?P<value>[^"]*)"')
JSON_NUMBER_RE = re.compile(r'"(?P<key>[^"]+)"\s*:\s*(?P<value>-?\d+(?:\.\d+)?)')


@dataclass(frozen=True)
class PlannerMessage:
    schema_version: str
    subgoal_id: str
    target: Dict[str, str]
    action_hint: str
    success_check: str
    confidence: float

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "subgoal_id": self.subgoal_id,
                "target": self.target,
                "action_hint": self.action_hint,
                "success_check": self.success_check,
                "confidence": self.confidence,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True)
class ExecutorDecision:
    reason: str
    action_id: int
    schema_version: str = EXECUTOR_SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "reason": self.reason,
                "action_id": self.action_id,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True)
class PlannerValidation:
    valid: bool
    reason: str
    message: PlannerMessage
    raw_dict: Optional[Dict[str, Any]]
    executable_hint: bool
    contradiction: bool


@dataclass(frozen=True)
class ExecutorValidation:
    valid: bool
    reason: str
    decision: Optional[ExecutorDecision]
    action: Optional[str]


def build_structured_bootstrap() -> tuple[list[dict[str, str]], str]:
    system = (
        "You are part of a structured exploration team. "
        "Wait for specialized role prompts and reply only in the requested JSON schema."
    )
    ack = "OK. I will follow the role prompt exactly and return only valid JSON."
    messages = [
        {"role": "user", "content": system},
        {"role": "assistant", "content": ack},
    ]
    prompt = (
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        "<|im_end|>\n<|im_start|>user\n"
        + system
        + "<|im_end|>\n<|im_start|>assistant\n"
        + ack
        + "<|im_end|>"
    )
    return messages, prompt


def extract_available_actions(observation: str) -> List[str]:
    match = AVAILABLE_ACTIONS_RE.search(observation)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    return [str(action).strip() for action in parsed if str(action).strip()]


def normalize_action_hint(action_hint: str) -> str:
    return str(action_hint).strip().lower().replace(" ", "_")


def _infer_action_hint_from_legal_actions(legal_actions: Sequence[str]) -> str:
    for action in legal_actions:
        normalized = action.lower().strip()
        if normalized == "turn left":
            return "turn_left"
        if normalized == "turn right":
            return "turn_right"
        if normalized == "move forward":
            return "move_forward"
        if "pick up" in normalized or normalized.startswith("pickup "):
            return "pickup"
        if "toggle" in normalized:
            return "toggle"
        if normalized == "drop":
            return "drop"
    return "done"


def _default_success_check(subgoal_id: str, action_hint: str) -> str:
    if subgoal_id == "pickup":
        return "inventory changes after pickup"
    if subgoal_id in {"open", "unlock"} or action_hint == "toggle":
        return "door state changes after toggle"
    if subgoal_id == "finish" or action_hint == "done":
        return "task reward changes after completion"
    return "observation changes after action"


def _normalize_success_check(value: str, subgoal_id: str, action_hint: str) -> str:
    cleaned = re.sub(r"[_-]+", " ", str(value or "")).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return _default_success_check(subgoal_id, action_hint)
    lowered = cleaned.lower()
    if lowered.startswith("success check"):
        cleaned = cleaned[len("success check") :].strip()
    elif lowered.startswith("success"):
        cleaned = cleaned[len("success") :].strip()
    cleaned = cleaned or _default_success_check(subgoal_id, action_hint)
    words = cleaned.split()
    if len(words) > 12:
        return _default_success_check(subgoal_id, action_hint)
    return cleaned


def _extract_string_field(raw_text: str, field_names: Sequence[str]) -> Optional[str]:
    for field in field_names:
        pattern = re.compile(rf'"{re.escape(field)}"\s*:\s*"([^"]*)"', re.IGNORECASE)
        match = pattern.search(raw_text or "")
        if match:
            return match.group(1)
    return None


def _extract_float_field(raw_text: str, field_name: str) -> Optional[float]:
    pattern = re.compile(rf'"{re.escape(field_name)}"\s*:\s*(-?\d+(?:\.\d+)?)', re.IGNORECASE)
    match = pattern.search(raw_text or "")
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _salvage_planner_payload(raw_text: str, legal_actions: Sequence[str]) -> Optional[Dict[str, Any]]:
    raw_text = raw_text or ""
    fields = {match.group("key"): match.group("value") for match in JSON_FIELD_RE.finditer(raw_text)}
    numbers = {match.group("key"): match.group("value") for match in JSON_NUMBER_RE.finditer(raw_text)}
    if not fields and not numbers:
        return None
    action_hint = normalize_action_hint(fields.get("action_hint") or "")
    subgoal_id = str(fields.get("subgoal_id", "")).strip().lower()
    target = {
        "object_type": str(fields.get("object_type", "none")).strip().lower() or "none",
        "color": str(fields.get("color", "none")).strip().lower() or "none",
        "location_hint": str(fields.get("location_hint", "unknown")).strip().lower() or "unknown",
    }
    if not (
        action_hint
        or subgoal_id
        or fields.get("object_type")
        or fields.get("color")
        or fields.get("location_hint")
    ):
        return None
    if subgoal_id not in SUBGOAL_IDS:
        subgoal_id = "explore"
    if action_hint not in ACTION_HINTS:
        action_hint = _infer_action_hint_from_legal_actions(legal_actions)
    confidence = _extract_float_field(raw_text, "confidence")
    if confidence is None:
        try:
            confidence = float(numbers.get("confidence", 0.0))
        except ValueError:
            confidence = 0.0
    success_check = _extract_string_field(raw_text, ["success_check", "success"]) or ""
    success_check = _normalize_success_check(success_check, subgoal_id, action_hint)
    return {
        "schema_version": fields.get("schema_version", PLANNER_SCHEMA_VERSION) or PLANNER_SCHEMA_VERSION,
        "subgoal_id": subgoal_id,
        "target": {
            "object_type": target["object_type"] if target["object_type"] in OBJECT_TYPES else "none",
            "color": target["color"] if target["color"] in OBJECT_COLORS else "none",
            "location_hint": target["location_hint"] if target["location_hint"] in LOCATION_HINTS else "unknown",
        },
        "action_hint": action_hint,
        "success_check": success_check,
        "confidence": max(0.0, min(1.0, float(confidence))),
    }


def _load_planner_payload(raw_text: str, legal_actions: Sequence[str]) -> tuple[Optional[Dict[str, Any]], bool]:
    blob = _extract_json_blob(raw_text)
    parsed_dict: Optional[Dict[str, Any]] = None
    if blob is not None:
        try:
            candidate = json.loads(blob)
        except json.JSONDecodeError:
            candidate = None
        if isinstance(candidate, dict):
            parsed_dict = candidate
    salvaged = _salvage_planner_payload(raw_text, legal_actions)
    if parsed_dict is None:
        return salvaged, bool(salvaged)
    if not salvaged:
        return parsed_dict, False
    merged = dict(parsed_dict)
    merged.setdefault("schema_version", salvaged["schema_version"])
    merged.setdefault("subgoal_id", salvaged["subgoal_id"])
    merged.setdefault("target", salvaged["target"])
    merged.setdefault("action_hint", salvaged["action_hint"])
    merged.setdefault("success_check", salvaged["success_check"])
    if merged.get("success_check") is None and "success" in merged:
        merged["success_check"] = merged.get("success")
    merged.setdefault("confidence", salvaged["confidence"])
    return merged, merged != parsed_dict


def _action_matches_target(action: str, target: Dict[str, str]) -> bool:
    normalized = action.lower()
    object_type = target.get("object_type", "none")
    color = target.get("color", "none")
    if object_type != "none" and object_type not in normalized:
        return False
    if color != "none" and color not in normalized:
        return False
    return True


def _match_action_hint_to_index(
    action_hint: str,
    legal_actions: Sequence[str],
    target: Optional[Dict[str, str]] = None,
) -> Optional[int]:
    target = target or {"object_type": "none", "color": "none", "location_hint": "unknown"}
    normalized_actions = [action.lower().strip() for action in legal_actions]

    def choose(predicate) -> Optional[int]:
        for idx, action in enumerate(normalized_actions):
            if predicate(action):
                return idx
        return None

    preferred_matches = {
        "turn_left": lambda action: action == "turn left",
        "turn_right": lambda action: action == "turn right",
        "move_forward": lambda action: action == "move forward",
        "pickup": lambda action: ("pick up" in action or action.startswith("pickup ")) and _action_matches_target(action, target),
        "toggle": lambda action: "toggle" in action and _action_matches_target(action, target),
        "drop": lambda action: action == "drop",
        "done": lambda action: action == "check available actions",
    }
    if action_hint in preferred_matches:
        match = choose(preferred_matches[action_hint])
        if match is not None:
            return match

    if action_hint in {"move_forward", "pickup", "toggle"}:
        targeted_go_to = choose(
            lambda action: action.startswith("go to ") and _action_matches_target(action, target)
        )
        if targeted_go_to is not None:
            return targeted_go_to

    if target.get("object_type", "none") != "none":
        targeted_any = choose(lambda action: _action_matches_target(action, target))
        if targeted_any is not None:
            return targeted_any

    safe_default = choose(lambda action: action != "check available actions")
    if safe_default is not None:
        return safe_default
    return 0 if legal_actions else None


def build_planner_prompt(observation: str, previous_action: str = "", legal_actions: Optional[Sequence[str]] = None) -> str:
    legal_lines = "\n".join(f"- {action}" for action in (legal_actions or []))
    return (
        "You are the planner for a structured BabyAI team.\n"
        "Return JSON only. Do not include markdown, chain-of-thought, role labels, or action wrappers.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "schema_version": "v1",\n'
        '  "subgoal_id": "explore|approach|pickup|open|unlock|drop|finish",\n'
        '  "target": {"object_type": "ball|box|key|door|goal|none", "color": "red|green|blue|yellow|grey|purple|none", "location_hint": "front|left|right|current_room|adjacent_room|unknown"},\n'
        '  "action_hint": "turn_left|turn_right|move_forward|pickup|toggle|drop|done",\n'
        '  "success_check": "short literal condition string",\n'
        '  "confidence": 0.0\n'
        "}\n\n"
        "Requirements:\n"
        "- One compact JSON object only\n"
        "- success_check must be short and literal\n"
        "- action_hint must be executable in BabyAI\n"
        "- target.object_type may be none only when no specific object is needed\n\n"
        f"Previous action: {previous_action or 'none'}\n"
        f"Observation:\n{observation}\n\n"
        f"Current legal actions:\n{legal_lines}\n"
    )


def build_executor_prompt(
    observation: str,
    planner_message: PlannerMessage,
    legal_actions: Sequence[str],
    previous_action: str = "",
) -> str:
    action_lines = "\n".join(f"{idx}: {action}" for idx, action in enumerate(legal_actions))
    planner_json = planner_message.to_json()
    return (
        "You are the executor for a structured BabyAI team.\n"
        "Use the planner JSON as advisory context only. Choose exactly one action index.\n"
        "Return JSON only with this schema:\n"
        '{\"schema_version\":\"v1\",\"reason\":\"one short sentence, max 20 words\",\"action_id\":0}\n\n'
        "Requirements:\n"
        "- JSON only\n"
        "- reason must be one short sentence with at most 20 words\n"
        "- action_id must be an integer index into the legal action list below\n"
        "- Do not output the final BabyAI action text yourself\n\n"
        f"Previous action: {previous_action or 'none'}\n"
        f"Observation:\n{observation}\n\n"
        f"Planner advisory JSON:\n{planner_json}\n\n"
        f"Legal actions:\n{action_lines}\n"
    )


def _extract_json_blob(raw_text: str) -> Optional[str]:
    if not raw_text:
        return None
    text = raw_text.strip()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        match = JSON_RE.search(text)
        if not match:
            return None
        candidate = match.group(0).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            return None


def _fallback_planner_message() -> PlannerMessage:
    return PlannerMessage(
        schema_version=PLANNER_SCHEMA_VERSION,
        subgoal_id="explore",
        target={"object_type": "none", "color": "none", "location_hint": "unknown"},
        action_hint="turn_right",
        success_check="observation changes after scouting",
        confidence=0.0,
    )


def _target_matches_action(target: Dict[str, str], legal_actions: Sequence[str]) -> bool:
    object_type = target.get("object_type", "none")
    if object_type == "none":
        return True
    color = target.get("color", "none")
    for action in legal_actions:
        normalized = action.lower()
        if object_type in normalized and (color == "none" or color in normalized):
            return True
    return False


def validate_planner_json(raw_text: str, legal_actions: Sequence[str]) -> PlannerValidation:
    fallback = _fallback_planner_message()
    lowered = raw_text.lower()
    if any(token in lowered for token in DISALLOWED_PLANNER_TOKENS):
        return PlannerValidation(False, "contains_disallowed_tokens", fallback, None, False, True)

    data, repaired = _load_planner_payload(raw_text, legal_actions)
    if data is None:
        return PlannerValidation(False, "invalid_json", fallback, None, False, True)

    if not isinstance(data, dict):
        return PlannerValidation(False, "not_object", fallback, None, False, True)

    target = data.get("target")
    if not isinstance(target, dict):
        target = {"object_type": "none", "color": "none", "location_hint": "unknown"}

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    subgoal_id = str(data.get("subgoal_id", "")).strip().lower()
    if subgoal_id not in SUBGOAL_IDS:
        subgoal_id = "explore"
    action_hint = normalize_action_hint(data.get("action_hint", ""))
    if action_hint not in ACTION_HINTS:
        action_hint = _infer_action_hint_from_legal_actions(legal_actions)
    target_object = str(target.get("object_type", "none")).strip().lower()
    target_color = str(target.get("color", "none")).strip().lower()
    target_location = str(target.get("location_hint", "unknown")).strip().lower()
    if target_object not in OBJECT_TYPES:
        target_object = "none"
    if target_color not in OBJECT_COLORS:
        target_color = "none"
    if target_location not in LOCATION_HINTS:
        target_location = "unknown"
    success_check = _normalize_success_check(data.get("success_check", data.get("success", "")), subgoal_id, action_hint)

    message = PlannerMessage(
        schema_version=str(data.get("schema_version", PLANNER_SCHEMA_VERSION) or PLANNER_SCHEMA_VERSION),
        subgoal_id=subgoal_id,
        target={
            "object_type": target_object,
            "color": target_color,
            "location_hint": target_location,
        },
        action_hint=action_hint,
        success_check=success_check,
        confidence=max(0.0, min(1.0, confidence)),
    )

    checks = [
        (message.schema_version == PLANNER_SCHEMA_VERSION, "bad_schema_version"),
        (0.0 <= message.confidence <= 1.0, "bad_confidence_range"),
        (bool(message.success_check) and len(message.success_check.split()) <= 12, "bad_success_check"),
    ]
    for ok, reason in checks:
        if not ok:
            return PlannerValidation(False, reason, fallback, data, False, True)

    executable_hint = any(
        legal_action == "turn left" and message.action_hint == "turn_left"
        or legal_action == "turn right" and message.action_hint == "turn_right"
        or legal_action == "move forward" and message.action_hint == "move_forward"
        or message.action_hint == "pickup" and "pick up" in legal_action.lower()
        or message.action_hint == "toggle" and "toggle" in legal_action.lower()
        or message.action_hint == "drop" and legal_action.lower() == "drop"
        or message.action_hint == "done"
        for legal_action in legal_actions
    )
    contradiction = not _target_matches_action(message.target, legal_actions)
    reason = "ok_repaired" if repaired else "ok"
    if contradiction:
        reason = "ok_repaired_target_not_grounded" if repaired else "ok_target_not_grounded"
    elif not executable_hint and message.action_hint != "done":
        reason = "ok_repaired_action_hint_not_executable" if repaired else "ok_action_hint_not_executable"

    return PlannerValidation(True, reason, message, data, executable_hint, contradiction)


def validate_executor_json(raw_text: str, legal_actions: Sequence[str]) -> ExecutorValidation:
    blob = _extract_json_blob(raw_text)
    if blob is None:
        planner_payload = _salvage_planner_payload(raw_text, legal_actions)
        if planner_payload is None:
            return ExecutorValidation(False, "invalid_json", None, None)
        action_id = _match_action_hint_to_index(
            planner_payload["action_hint"],
            legal_actions,
            planner_payload.get("target"),
        )
        if action_id is None:
            return ExecutorValidation(False, "bad_action_id", None, None)
        decision = ExecutorDecision(reason="Follow the structured subgoal.", action_id=action_id)
        return ExecutorValidation(True, "ok_repaired_from_planner", decision, legal_actions[action_id])
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        planner_payload = _salvage_planner_payload(raw_text, legal_actions)
        if planner_payload is None:
            return ExecutorValidation(False, "invalid_json", None, None)
        action_id = _match_action_hint_to_index(
            planner_payload["action_hint"],
            legal_actions,
            planner_payload.get("target"),
        )
        if action_id is None:
            return ExecutorValidation(False, "bad_action_id", None, None)
        decision = ExecutorDecision(reason="Follow the structured subgoal.", action_id=action_id)
        return ExecutorValidation(True, "ok_repaired_from_planner", decision, legal_actions[action_id])

    if not isinstance(data, dict):
        return ExecutorValidation(False, "not_object", None, None)

    try:
        action_id = int(data.get("action_id"))
    except (TypeError, ValueError):
        planner_payload = _salvage_planner_payload(raw_text, legal_actions)
        if planner_payload is None:
            return ExecutorValidation(False, "bad_action_id", None, None)
        action_id = _match_action_hint_to_index(
            planner_payload["action_hint"],
            legal_actions,
            planner_payload.get("target"),
        )
        if action_id is None:
            return ExecutorValidation(False, "bad_action_id", None, None)
        decision = ExecutorDecision(reason="Follow the structured subgoal.", action_id=action_id)
        return ExecutorValidation(True, "ok_repaired_from_planner", decision, legal_actions[action_id])

    reason = str(data.get("reason", "")).strip()
    schema_version = str(data.get("schema_version", EXECUTOR_SCHEMA_VERSION)).strip()
    if schema_version != EXECUTOR_SCHEMA_VERSION:
        return ExecutorValidation(False, "bad_schema_version", None, None)
    if not reason:
        return ExecutorValidation(False, "missing_reason", None, None)
    if len(reason.split()) > 20:
        return ExecutorValidation(False, "reason_too_long", None, None)
    if action_id < 0 or action_id >= len(legal_actions):
        return ExecutorValidation(False, "action_id_out_of_range", None, None)
    decision = ExecutorDecision(reason=reason, action_id=action_id)
    return ExecutorValidation(True, "ok", decision, legal_actions[action_id])


def render_babyai_action_payload(decision: ExecutorDecision, legal_actions: Sequence[str]) -> str:
    action = legal_actions[decision.action_id]
    return f"Thought:\n{decision.reason}\n\nAction:\n{action}"


def extract_visible_objects(observation: str) -> List[str]:
    return [f"{color.lower()} {obj.lower()}" for color, obj in VISIBLE_OBJECT_RE.findall(observation or "")]


def extract_carrying_item(observation: str) -> str:
    match = re.search(r"You are carrying (.+?)\.", observation or "", re.IGNORECASE)
    if not match:
        return "none"
    value = match.group(1).strip().lower()
    if value in {"nothing", "anything"}:
        return "none"
    return value


def extract_front_distance(observation: str) -> Optional[int]:
    match = STEP_COUNT_RE.search(observation or "")
    return int(match.group(1)) if match else None


def safe_json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def iter_action_prefixes(actions: Iterable[str]) -> Iterable[str]:
    for action in actions:
        normalized = action.lower()
        if normalized.startswith("turn left"):
            yield "turn_left"
        elif normalized.startswith("turn right"):
            yield "turn_right"
        elif normalized.startswith("move forward"):
            yield "move_forward"
        elif normalized.startswith("pick up"):
            yield "pickup"
        elif normalized.startswith("toggle"):
            yield "toggle"
        elif normalized.startswith("drop"):
            yield "drop"
        elif normalized.startswith("done"):
            yield "done"
