#!/usr/bin/env python3
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT / "AgentGym-RL"
for candidate in (PKG_ROOT, REPO_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from verl.improve_multi_agent.trace_bootstrap import (
    DEFAULT_EVAL_TRACE_SOURCES,
    DEFAULT_TRACE_SOURCES,
    build_executor_sft_examples,
    build_executor_sft_dataset,
    build_planner_sft_examples,
    build_planner_sft_dataset,
    collect_successful_trace_events,
    split_events_by_hash,
)


def main():
    repo_root = REPO_ROOT
    dataset_root = repo_root / "improve_multi-agent/datasets"
    dataset_root.mkdir(parents=True, exist_ok=True)
    train_events, eval_events = split_events_by_hash(collect_successful_trace_events(DEFAULT_TRACE_SOURCES))
    heldout_events = collect_successful_trace_events(DEFAULT_EVAL_TRACE_SOURCES)

    outputs = {
        "babyai_executor_warmstart.json": build_executor_sft_examples(train_events + eval_events),
        "babyai_planner_warmstart.json": build_planner_sft_examples(train_events + eval_events),
        "babyai_executor_warmstart_train.json": build_executor_sft_examples(train_events),
        "babyai_executor_warmstart_eval.json": build_executor_sft_examples(eval_events),
        "babyai_planner_warmstart_train.json": build_planner_sft_examples(train_events),
        "babyai_planner_warmstart_eval.json": build_planner_sft_examples(eval_events),
        "babyai_executor_warmstart_heldout.json": build_executor_sft_examples(heldout_events),
        "babyai_planner_warmstart_heldout.json": build_planner_sft_examples(heldout_events),
    }

    for name, examples in outputs.items():
        path = dataset_root / name
        path.write_text(json.dumps(examples, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"{name}: {path} ({len(examples)} examples)")

    print(
        "event splits:",
        json.dumps(
            {
                "train_events": len(train_events),
                "eval_events": len(eval_events),
                "heldout_events": len(heldout_events),
            },
            ensure_ascii=True,
        ),
    )


if __name__ == "__main__":
    main()
