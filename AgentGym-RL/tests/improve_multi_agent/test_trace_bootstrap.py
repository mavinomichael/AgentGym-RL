import json
from pathlib import Path

from conftest import load_improve_multi_agent_module


trace_bootstrap = load_improve_multi_agent_module(
    "verl.improve_multi_agent.trace_bootstrap",
    "verl/improve_multi_agent/trace_bootstrap.py",
)


def test_trace_bootstrap_collects_successful_events():
    report_root = Path("/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/2026-03-23_dense500_eval_and_traces/trace_train")
    events = trace_bootstrap.collect_successful_trace_events([report_root])
    assert events
    first = events[0]
    assert first["task_name"] == "babyai"
    assert first["validation_valid"] is True


def test_trace_bootstrap_builds_executor_and_planner_datasets(tmp_path):
    report_root = Path("/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/2026-03-23_dense500_eval_and_traces/trace_train")
    executor_path = trace_bootstrap.build_executor_sft_dataset(tmp_path / "executor.json", [report_root])
    planner_path = trace_bootstrap.build_planner_sft_dataset(tmp_path / "planner.json", [report_root])
    executor_data = json.loads(executor_path.read_text())
    planner_data = json.loads(planner_path.read_text())
    assert executor_data
    assert planner_data
    assert executor_data[0]["conversations"][0]["from"] == "human"
    assert planner_data[0]["conversations"][1]["from"] == "gpt"
    assert "metadata" in executor_data[0]
    assert "metadata" in planner_data[0]


def test_trace_bootstrap_split_and_examples(tmp_path):
    report_root = Path("/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/2026-03-23_dense500_eval_and_traces/trace_train")
    events = trace_bootstrap.collect_successful_trace_events([report_root])
    train_events, eval_events = trace_bootstrap.split_events_by_hash(events, eval_ratio=0.1)
    assert train_events
    assert eval_events
    executor_examples = trace_bootstrap.build_executor_sft_examples(train_events[:5])
    planner_examples = trace_bootstrap.build_planner_sft_examples(eval_events[:5])
    assert executor_examples[0]["metadata"]["available_actions"]
    assert planner_examples[0]["metadata"]["event_key"]


def test_trace_bootstrap_dedupes_exact_examples():
    duplicated = [
        {
            "conversations": [
                {"from": "human", "value": "prompt"},
                {"from": "gpt", "value": "{\"action_id\":0}"},
            ],
            "metadata": {"event_key": "a"},
        },
        {
            "conversations": [
                {"from": "human", "value": "prompt"},
                {"from": "gpt", "value": "{\"action_id\":0}"},
            ],
            "metadata": {"event_key": "b"},
        },
    ]
    deduped = trace_bootstrap.dedupe_sft_examples(duplicated)
    assert len(deduped) == 1
    assert deduped[0]["metadata"]["duplicate_count"] == 2
    assert deduped[0]["metadata"]["duplicate_event_keys"] == ["a", "b"]
