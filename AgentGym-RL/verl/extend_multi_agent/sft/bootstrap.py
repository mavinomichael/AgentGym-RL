from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from verl.extend_multi_agent.envs.prompt_policy import (
    build_executor_turn_prompt,
    build_planner_turn_prompt,
    extract_available_actions_from_observation,
)
from verl.extend_multi_agent.json_protocol import canonical_json
from verl.extend_multi_agent.envs.task_registry import get_task_profile

DEFAULT_TRACE_ROOTS = [
    "reports/babyai_multi_agent_diagnostics_2026-03-11/diagnostic_step100_v2_trace_train",
    "reports/babyai_multi_agent_diagnostics_2026-03-10/diagnostic_step60_trace_train/trace_train",
    "reports/babyai_multi_agent_diagnostics_2026-03-21/mavino_collapse_2agent_scaling_100/raw_logs/babyai_2agent_scaling_100_8gpu/trace_train",
]


@dataclass(frozen=True)
class SFTExample:
    conversations: List[Dict[str, str]]
    metadata: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {"conversations": self.conversations, "metadata": self.metadata}


def _iter_trace_files(trace_roots: Sequence[str]) -> Iterable[Path]:
    for root in trace_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        if root_path.is_file() and root_path.suffix == ".jsonl":
            yield root_path
            continue
        for path in sorted(root_path.rglob("*.jsonl")):
            yield path


def _iter_records(trace_roots: Sequence[str]) -> Iterable[Dict[str, object]]:
    for path in _iter_trace_files(trace_roots):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(record.get("task_name", "")).lower() != "babyai":
                    continue
                yield record


def _planner_target_from_record(record: Dict[str, object]) -> str:
    planner_message = str(record.get("planner_message") or record.get("planner_context_used_by_executor") or "").strip()
    executor_action = str(record.get("extracted_action") or "").strip()
    if planner_message:
        subgoal = planner_message
    elif executor_action:
        subgoal = f"prepare action {executor_action}"
    else:
        subgoal = "explore nearby state"
    constraints = ["action must stay in available actions"]
    available_actions = record.get("available_actions") or []
    if isinstance(available_actions, list) and available_actions:
        constraints.append(f"available_count={len(available_actions)}")
    return canonical_json({"subgoal": subgoal, "constraints": constraints, "done": False})


def _executor_target_from_record(record: Dict[str, object]) -> Optional[str]:
    action = str(record.get("extracted_action") or "").strip()
    if not action:
        return None
    return canonical_json({"action": action, "action_args": None, "done": False})


def _planner_example(record: Dict[str, object]) -> Optional[SFTExample]:
    observation = str(record.get("observation_excerpt") or "").strip()
    if not observation:
        return None
    profile = get_task_profile("babyai")
    prompt = build_planner_turn_prompt(observation, profile)
    target = _planner_target_from_record(record)
    metadata = {
        "item_id": record.get("item_id"),
        "training_step": record.get("training_step"),
        "source": "planner",
    }
    return SFTExample(
        conversations=[
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": target},
        ],
        metadata=metadata,
    )


def _executor_example(record: Dict[str, object]) -> Optional[SFTExample]:
    observation = str(record.get("observation_excerpt") or "").strip()
    if not observation:
        return None
    target = _executor_target_from_record(record)
    if not target:
        return None
    profile = get_task_profile("babyai")
    planner_json = _planner_target_from_record(record)
    prompt = build_executor_turn_prompt(observation, planner_json, profile)
    metadata = {
        "item_id": record.get("item_id"),
        "training_step": record.get("training_step"),
        "source": "executor",
    }
    return SFTExample(
        conversations=[
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": target},
        ],
        metadata=metadata,
    )


def _dedupe_examples(examples: Iterable[SFTExample]) -> List[SFTExample]:
    deduped: Dict[str, SFTExample] = {}
    for example in examples:
        key = json.dumps(example.conversations, ensure_ascii=True, sort_keys=True)
        deduped.setdefault(key, example)
    return list(deduped.values())


def build_sft_datasets(trace_roots: Sequence[str]) -> Dict[str, List[Dict[str, object]]]:
    planner_examples: List[SFTExample] = []
    executor_examples: List[SFTExample] = []
    for record in _iter_records(trace_roots):
        if not bool(record.get("validation_valid", False)):
            continue
        if not bool(record.get("env_step_called", False)):
            continue
        planner = _planner_example(record)
        executor = _executor_example(record)
        if planner is not None:
            planner_examples.append(planner)
        if executor is not None:
            executor_examples.append(executor)
    return {
        "planner": [example.to_dict() for example in _dedupe_examples(planner_examples)],
        "executor": [example.to_dict() for example in _dedupe_examples(executor_examples)],
    }


def write_sft_datasets(output_dir: str, trace_roots: Optional[Sequence[str]] = None) -> Dict[str, str]:
    roots = list(trace_roots or DEFAULT_TRACE_ROOTS)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    datasets = build_sft_datasets(roots)
    written: Dict[str, str] = {}
    for role, payload in datasets.items():
        path = output_path / f"babyai_{role}_sft.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
        written[role] = str(path)
    manifest_path = output_path / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump({"trace_roots": roots, "files": written}, handle, ensure_ascii=True, indent=2)
    written["manifest"] = str(manifest_path)
    return written
