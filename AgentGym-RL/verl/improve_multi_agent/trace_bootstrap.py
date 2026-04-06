from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

from .protocol import (
    PlannerMessage,
    build_executor_prompt,
    build_planner_prompt,
    render_babyai_action_payload,
    safe_json_dumps,
)

REPORT_ROOT = Path(__file__).resolve().parents[3] / "reports"
DEFAULT_TRACE_SOURCES = [
    REPORT_ROOT / "2026-03-23_dense500_eval_and_traces/trace_train",
    REPORT_ROOT / "2026-03-23_plain_split_retry_v2_deep_research",
    REPORT_ROOT / "20260402T123754Z_babyai_2agent_fixed20_8gpu_plain_split_notag_persist_v1",
]
DEFAULT_EVAL_TRACE_SOURCES = [
    REPORT_ROOT / "20260403_130617_babyai_3agent_executor_reviewer_scaling_600_8gpu_no_tags_v1/trace_train",
]


def _iter_trace_files(trace_roots: Sequence[Path]) -> Iterator[Path]:
    for root in trace_roots:
        if root.is_file() and root.suffix == ".jsonl":
            yield root
            continue
        if not root.exists():
            continue
        yield from sorted(root.rglob("executor_payload_trace_rank*.jsonl"))


def collect_successful_trace_events(trace_roots: Sequence[Path] | None = None) -> List[Dict]:
    trace_roots = list(trace_roots or DEFAULT_TRACE_SOURCES)
    events: List[Dict] = []
    seen = set()
    for trace_file in _iter_trace_files(trace_roots):
        with trace_file.open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(event.get("task_name", "")).lower() != "babyai":
                    continue
                if not bool(event.get("validation_valid")):
                    continue
                if not bool(event.get("executor_action_valid", True)):
                    continue
                if not bool(event.get("executor_native_format_valid", True)):
                    continue
                if event.get("env_reward") is None and not event.get("env_step_called", False):
                    continue
                key = (
                    event.get("training_step"),
                    event.get("item_id"),
                    event.get("rank"),
                    event.get("round"),
                    event.get("extracted_action"),
                )
                if key in seen:
                    continue
                seen.add(key)
                events.append(event)
    return events


def _planner_target_from_event(event: Dict) -> Dict[str, str]:
    action = str(event.get("extracted_action", "")).lower()
    colors = ["red", "green", "blue", "yellow", "grey", "purple"]
    object_types = ["ball", "box", "key", "door", "goal"]
    color = next((candidate for candidate in colors if candidate in action), "none")
    object_type = next((candidate for candidate in object_types if candidate in action), "none")
    location_hint = "current_room"
    if "left" in str(event.get("observation_excerpt", "")).lower():
        location_hint = "left"
    elif "right" in str(event.get("observation_excerpt", "")).lower():
        location_hint = "right"
    elif "front" in str(event.get("observation_excerpt", "")).lower():
        location_hint = "front"
    return {
        "object_type": object_type,
        "color": color,
        "location_hint": location_hint,
    }


def _planner_message_from_event(event: Dict) -> PlannerMessage:
    action = str(event.get("extracted_action", "")).lower()
    if action.startswith("pick up"):
        subgoal = "pickup"
        action_hint = "pickup"
    elif action.startswith("toggle"):
        subgoal = "unlock" if "locked door" in action else "open"
        action_hint = "toggle"
    elif action.startswith("drop"):
        subgoal = "drop"
        action_hint = "drop"
    elif action.startswith("move forward"):
        subgoal = "approach"
        action_hint = "move_forward"
    elif action.startswith("turn left"):
        subgoal = "explore"
        action_hint = "turn_left"
    elif action.startswith("turn right"):
        subgoal = "explore"
        action_hint = "turn_right"
    else:
        subgoal = "approach"
        action_hint = "move_forward"
    return PlannerMessage(
        schema_version="v1",
        subgoal_id=subgoal,
        target=_planner_target_from_event(event),
        action_hint=action_hint,
        success_check="world state moves toward target",
        confidence=0.75,
    )


def _event_key(event: Dict) -> str:
    return "|".join(
        [
            str(event.get("training_step", "")),
            str(event.get("item_id", "")),
            str(event.get("rank", "")),
            str(event.get("round", "")),
            str(event.get("extracted_action", "")),
        ]
    )


def split_events_by_hash(events: Sequence[Dict], *, eval_ratio: float = 0.1) -> tuple[List[Dict], List[Dict]]:
    train_events: List[Dict] = []
    eval_events: List[Dict] = []
    cutoff = max(1, int(eval_ratio * 100))
    for event in events:
        digest = hashlib.md5(_event_key(event).encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 100
        if bucket < cutoff:
            eval_events.append(event)
        else:
            train_events.append(event)
    return train_events, eval_events


def dedupe_sft_examples(examples: Sequence[Dict]) -> List[Dict]:
    deduped: List[Dict] = []
    seen: Dict[str, int] = {}
    for example in examples:
        conversations = example.get("conversations", [])
        key = hashlib.sha256(json.dumps(conversations, ensure_ascii=True, sort_keys=True).encode("utf-8")).hexdigest()
        if key in seen:
            existing = deduped[seen[key]]
            existing_meta = existing.setdefault("metadata", {})
            existing_meta["duplicate_count"] = int(existing_meta.get("duplicate_count", 1)) + 1
            source_keys = existing_meta.setdefault("duplicate_event_keys", [])
            event_key = example.get("metadata", {}).get("event_key")
            if event_key and event_key not in source_keys:
                source_keys.append(event_key)
            continue
        cloned = json.loads(json.dumps(example))
        metadata = cloned.setdefault("metadata", {})
        metadata["duplicate_count"] = 1
        event_key = metadata.get("event_key")
        metadata["duplicate_event_keys"] = [event_key] if event_key else []
        seen[key] = len(deduped)
        deduped.append(cloned)
    return deduped


def build_executor_sft_examples(events: Sequence[Dict]) -> List[Dict]:
    examples: List[Dict] = []
    for event in events:
        legal_actions = list(event.get("available_actions") or [])
        extracted_action = event.get("extracted_action")
        if extracted_action not in legal_actions:
            continue
        planner_message = _planner_message_from_event(event)
        prompt = build_executor_prompt(
            observation=str(event.get("observation_excerpt", "")),
            planner_message=planner_message,
            legal_actions=legal_actions,
            previous_action="none",
        )
        target = {
            "schema_version": "v1",
            "reason": "Follow the grounded subgoal with a legal action.",
            "action_id": legal_actions.index(extracted_action),
        }
        examples.append(
            {
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": safe_json_dumps(target)},
                ],
                "metadata": {
                    "event_key": _event_key(event),
                    "training_step": event.get("training_step"),
                    "item_id": event.get("item_id"),
                    "rank": event.get("rank"),
                    "round": event.get("round"),
                    "available_actions": legal_actions,
                    "expected_action": extracted_action,
                    "planner_json": json.loads(planner_message.to_json()),
                },
            }
        )
    return dedupe_sft_examples(examples)


def build_planner_sft_examples(events: Sequence[Dict]) -> List[Dict]:
    examples: List[Dict] = []
    for event in events:
        legal_actions = list(event.get("available_actions") or [])
        prompt = build_planner_prompt(
            observation=str(event.get("observation_excerpt", "")),
            previous_action="none",
            legal_actions=legal_actions,
        )
        planner_message = _planner_message_from_event(event)
        examples.append(
            {
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": planner_message.to_json()},
                ],
                "metadata": {
                    "event_key": _event_key(event),
                    "training_step": event.get("training_step"),
                    "item_id": event.get("item_id"),
                    "rank": event.get("rank"),
                    "round": event.get("round"),
                    "available_actions": legal_actions,
                    "expected_action": event.get("extracted_action"),
                },
            }
        )
    return dedupe_sft_examples(examples)


def build_executor_sft_dataset(output_path: Path, trace_roots: Sequence[Path] | None = None) -> Path:
    examples = build_executor_sft_examples(collect_successful_trace_events(trace_roots))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(examples, ensure_ascii=True, indent=2), encoding="utf-8")
    return output_path


def build_planner_sft_dataset(output_path: Path, trace_roots: Sequence[Path] | None = None) -> Path:
    examples = build_planner_sft_examples(collect_successful_trace_events(trace_roots))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(examples, ensure_ascii=True, indent=2), encoding="utf-8")
    return output_path
