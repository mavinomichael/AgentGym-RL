#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT / "AgentGym-RL"

for candidate in (PKG_ROOT, REPO_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from verl.improve_multi_agent.rollout_harness import (
    _SharedTransformersRunner,
    aggregate_episode_records,
    build_babyai_env_client,
    build_executor_agent,
    build_planner_agent,
    run_babyai_rollout_episode,
    write_aggregate_summary,
    write_episode_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone fail-fast BabyAI rollout harness for improve_multi_agent.")
    parser.add_argument("--env-server-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--episode-count", type=int, default=1)
    parser.add_argument("--start-item-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--planner-interval", type=int, default=3)
    parser.add_argument("--env-timeout", type=float, default=20.0)
    parser.add_argument("--reset-timeout", type=float, default=20.0)
    parser.add_argument("--observe-timeout", type=float, default=15.0)
    parser.add_argument("--available-actions-timeout", type=float, default=2.0)
    parser.add_argument("--planner-timeout", type=float, default=45.0)
    parser.add_argument("--executor-timeout", type=float, default=45.0)
    parser.add_argument("--env-step-timeout", type=float, default=30.0)
    parser.add_argument("--planner-mode", choices=["scripted", "model"], default="scripted")
    parser.add_argument("--executor-mode", choices=["scripted", "model"], default="scripted")
    parser.add_argument("--planner-checkpoint", type=str)
    parser.add_argument("--executor-checkpoint", type=str)
    parser.add_argument("--planner-max-new-tokens", type=int, default=96)
    parser.add_argument("--executor-max-new-tokens", type=int, default=96)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    planner_runner = _SharedTransformersRunner() if args.planner_mode == "model" else None
    executor_runner = _SharedTransformersRunner() if args.executor_mode == "model" else None
    planner_agent = build_planner_agent(
        mode=args.planner_mode,
        checkpoint=args.planner_checkpoint,
        max_new_tokens=args.planner_max_new_tokens,
        runner=planner_runner,
    )
    executor_agent = build_executor_agent(
        mode=args.executor_mode,
        checkpoint=args.executor_checkpoint,
        max_new_tokens=args.executor_max_new_tokens,
        runner=executor_runner,
    )

    boundary_timeouts = {
        "reset": args.reset_timeout,
        "observe": args.observe_timeout,
        "available_actions": args.available_actions_timeout,
        "planner_generate": args.planner_timeout,
        "planner_validate": 2.0,
        "executor_generate": args.executor_timeout,
        "executor_validate": 2.0,
        "env_step": args.env_step_timeout,
    }

    records = []
    for episode_offset in range(args.episode_count):
        item_id = args.start_item_id + episode_offset
        env_client = build_babyai_env_client(args.env_server_url, timeout_seconds=args.env_timeout)
        try:
            record = run_babyai_rollout_episode(
                item_id=item_id,
                env_client=env_client,
                planner_agent=planner_agent,
                executor_agent=executor_agent,
                max_steps=args.max_steps,
                planner_interval=args.planner_interval,
                boundary_timeouts=boundary_timeouts,
            )
        finally:
            try:
                env_client.close()
            except Exception:
                pass

        record_dir = args.output_dir / f"episode_{episode_offset:03d}"
        write_episode_artifacts(
            record,
            summary_path=record_dir / "summary.json",
            trace_path=record_dir / "trace.jsonl",
        )
        records.append(record)

    aggregate = aggregate_episode_records(records)
    write_aggregate_summary(records, args.summary_path)
    print(json.dumps(aggregate, ensure_ascii=True, indent=2))
    return 0 if aggregate["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
