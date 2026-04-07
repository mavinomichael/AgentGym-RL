from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

PLANNER_JSON_KEYS = ("subgoal", "constraints", "done")
EXECUTOR_JSON_KEYS = ("action", "action_args", "done")
JSON_BLOB_RE = re.compile(r"\{.*\}", re.DOTALL)
AVAILABLE_ACTIONS_RE = re.compile(r"Available actions:\s*(\[[^\]]*\])", re.DOTALL)


@dataclass(frozen=True)
class PlannerJSONValidation:
    valid: bool
    reason: str
    payload: Optional[Dict[str, Any]]


@dataclass(frozen=True)
class ExecutorJSONValidation:
    valid: bool
    reason: str
    payload: Optional[Dict[str, Any]]
    action: Optional[str]


def normalize_json_candidate(raw_text: str) -> str:
    text = (raw_text or "").strip()
    for suffix in ("</s>", "<|im_end|>", "<|endoftext|>"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].rstrip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_blob(raw_text: str) -> Optional[str]:
    normalized = normalize_json_candidate(raw_text)
    if not normalized:
        return None
    if normalized.startswith("{") and normalized.endswith("}"):
        return normalized
    match = JSON_BLOB_RE.search(normalized)
    if not match:
        return None
    return match.group(0)


def _load_object(raw_text: str) -> Optional[Dict[str, Any]]:
    blob = _extract_json_blob(raw_text)
    if blob is None:
        return None
    try:
        loaded = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded


def canonical_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=False)


def extract_available_actions(observation: str) -> List[str]:
    match = AVAILABLE_ACTIONS_RE.search(observation or "")
    if not match:
        return []
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def validate_planner_json(raw_text: str) -> PlannerJSONValidation:
    payload = _load_object(raw_text)
    if payload is None:
        return PlannerJSONValidation(valid=False, reason="invalid_json", payload=None)
    if tuple(payload.keys()) != PLANNER_JSON_KEYS:
        if set(payload.keys()) != set(PLANNER_JSON_KEYS):
            return PlannerJSONValidation(valid=False, reason="schema_keys_mismatch", payload=None)
        ordered = {key: payload[key] for key in PLANNER_JSON_KEYS}
        payload = ordered
    subgoal = payload.get("subgoal")
    constraints = payload.get("constraints")
    done = payload.get("done")
    if not isinstance(subgoal, str) or not subgoal.strip():
        return PlannerJSONValidation(valid=False, reason="bad_subgoal", payload=None)
    if not isinstance(constraints, list) or any(not isinstance(item, str) for item in constraints):
        return PlannerJSONValidation(valid=False, reason="bad_constraints", payload=None)
    if not isinstance(done, bool):
        return PlannerJSONValidation(valid=False, reason="bad_done", payload=None)
    normalized = {
        "subgoal": subgoal.strip(),
        "constraints": [item.strip() for item in constraints],
        "done": done,
    }
    return PlannerJSONValidation(valid=True, reason="ok", payload=normalized)


def validate_executor_json(raw_text: str, legal_actions: Sequence[str]) -> ExecutorJSONValidation:
    payload = _load_object(raw_text)
    if payload is None:
        return ExecutorJSONValidation(valid=False, reason="invalid_json", payload=None, action=None)
    if tuple(payload.keys()) != EXECUTOR_JSON_KEYS:
        if set(payload.keys()) != set(EXECUTOR_JSON_KEYS):
            return ExecutorJSONValidation(valid=False, reason="schema_keys_mismatch", payload=None, action=None)
        ordered = {key: payload[key] for key in EXECUTOR_JSON_KEYS}
        payload = ordered
    action = payload.get("action")
    action_args = payload.get("action_args")
    done = payload.get("done")
    if not isinstance(action, str) or not action.strip():
        return ExecutorJSONValidation(valid=False, reason="bad_action", payload=None, action=None)
    if action_args not in (None, {}):
        return ExecutorJSONValidation(valid=False, reason="bad_action_args", payload=None, action=None)
    if not isinstance(done, bool):
        return ExecutorJSONValidation(valid=False, reason="bad_done", payload=None, action=None)
    normalized_action = action.strip()
    if legal_actions and normalized_action not in legal_actions:
        return ExecutorJSONValidation(
            valid=False,
            reason="action_not_in_available",
            payload=None,
            action=normalized_action,
        )
    normalized = {
        "action": normalized_action,
        "action_args": None if action_args is None else {},
        "done": done,
    }
    return ExecutorJSONValidation(valid=True, reason="ok", payload=normalized, action=normalized_action)


def render_executor_env_action(payload: Dict[str, Any], legal_actions: Sequence[str]) -> str:
    action = str(payload["action"]).strip()
    if legal_actions and action not in legal_actions:
        raise ValueError(f"Executor action not in available actions: {action}")
    return action


def build_planner_schema_example() -> str:
    return canonical_json({"subgoal": "approach the target", "constraints": ["use one available action"], "done": False})


def build_executor_schema_example(action: Optional[str] = None) -> str:
    return canonical_json({"action": action or "turn left", "action_args": None, "done": False})
