import json
import time
from pathlib import Path
from types import SimpleNamespace

from conftest import load_improve_multi_agent_module


harness = load_improve_multi_agent_module(
    "verl.improve_multi_agent.rollout_harness",
    "verl/improve_multi_agent/rollout_harness.py",
)


class FakeBabyAIEnv:
    def __init__(self):
        self.item_id = None
        self.actions = []
        self.observation = (
            "Your goal: go to the red ball\n"
            "In front of you in this room, you can see several objects: "
            "There is a red ball 1 0 steps in front of you and 2 steps to your left. "
            "The room has walls around you. You are facing a wall 2 steps away. "
            'You are not carrying anything.\nAvailable actions: ["turn left", "turn right", "move forward", "go to red ball 1"]'
        )

    def reset(self, item_id: int):
        self.item_id = item_id
        self.actions.clear()
        return {"observation": self.observation, "reward": 0.0, "score": 0.0, "done": False}

    def observe(self):
        return self.observation

    def step(self, action: str):
        self.actions.append(action)
        self.observation = (
            "Your goal: go to the red ball\n"
            'You are standing next to the red ball.\nAvailable actions: ["turn left", "turn right", "move forward"]'
        )
        return SimpleNamespace(state=self.observation, reward=1.0, done=True)

    def close(self):
        return None


class SlowPlanner:
    mode = "scripted"

    def generate(self, observation, legal_actions, previous_action):
        del observation, legal_actions, previous_action
        time.sleep(0.05)
        return "{}"


class InvalidPlanner:
    mode = "scripted"

    def generate(self, observation, legal_actions, previous_action):
        del observation, legal_actions, previous_action
        return "not-json"


class InvalidExecutor:
    mode = "scripted"

    def generate(self, observation, planner_message, legal_actions, previous_action):
        del observation, planner_message, legal_actions, previous_action
        return "not-json"


class IllegalExecutor:
    mode = "scripted"

    def generate(self, observation, planner_message, legal_actions, previous_action):
        del observation, planner_message, legal_actions, previous_action
        return '{"schema_version":"v1","reason":"Pick a bad index.","action_id":99}'


def test_run_with_timeout_returns_timeout():
    status, payload = harness._run_with_timeout(lambda: time.sleep(0.05), timeout_seconds=0.01)
    assert status == "timeout"
    assert "Exceeded timeout" in str(payload)


def test_rollout_harness_fails_cleanly_on_invalid_planner_json():
    record = harness.run_babyai_rollout_episode(
        item_id=0,
        env_client=FakeBabyAIEnv(),
        planner_agent=InvalidPlanner(),
        executor_agent=harness.ScriptedExecutorAgent(),
        max_steps=2,
        planner_interval=1,
        boundary_timeouts={"planner_generate": 1.0},
    )
    assert record.status == "failed"
    assert record.failure_type == "planner_invalid_json"
    assert record.completed_env_steps == 0


def test_rollout_harness_converts_planner_timeout_into_typed_failure():
    record = harness.run_babyai_rollout_episode(
        item_id=0,
        env_client=FakeBabyAIEnv(),
        planner_agent=SlowPlanner(),
        executor_agent=harness.ScriptedExecutorAgent(),
        max_steps=2,
        planner_interval=1,
        boundary_timeouts={"planner_generate": 0.01},
    )
    assert record.status == "failed"
    assert record.failure_type == "planner_timeout"
    assert record.completed_env_steps == 0


def test_rollout_harness_fails_cleanly_on_invalid_executor_json():
    record = harness.run_babyai_rollout_episode(
        item_id=0,
        env_client=FakeBabyAIEnv(),
        planner_agent=harness.ScriptedPlannerAgent(),
        executor_agent=InvalidExecutor(),
        max_steps=2,
        planner_interval=1,
    )
    assert record.status == "failed"
    assert record.failure_type == "executor_invalid_json"
    assert record.completed_env_steps == 0


def test_rollout_harness_fails_cleanly_on_illegal_action_id():
    record = harness.run_babyai_rollout_episode(
        item_id=0,
        env_client=FakeBabyAIEnv(),
        planner_agent=harness.ScriptedPlannerAgent(),
        executor_agent=IllegalExecutor(),
        max_steps=2,
        planner_interval=1,
    )
    assert record.status == "failed"
    assert record.failure_type == "illegal_action_id"
    assert record.completed_env_steps == 0


def test_rollout_harness_completes_first_valid_transition_with_scripted_agents(tmp_path: Path):
    record = harness.run_babyai_rollout_episode(
        item_id=3,
        env_client=FakeBabyAIEnv(),
        planner_agent=harness.ScriptedPlannerAgent(),
        executor_agent=harness.ScriptedExecutorAgent(),
        max_steps=2,
        planner_interval=1,
    )
    assert record.status == "passed"
    assert record.first_transition_completed is True
    assert record.completed_env_steps >= 1
    summary_path = tmp_path / "summary.json"
    trace_path = tmp_path / "trace.jsonl"
    harness.write_episode_artifacts(record, summary_path=summary_path, trace_path=trace_path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert any('"type":"terminal"' in line for line in trace_path.read_text(encoding="utf-8").splitlines())


def test_scripted_planner_coerces_action_hint_to_a_legal_primitive():
    agent = harness.ScriptedPlannerAgent()
    observation = (
        "Your goal: go to the red ball\n"
        "There is a red ball 1 0 steps in front of you and 2 steps to your right.\n"
        'Available actions: ["turn left", "turn right", "check available actions"]'
    )
    raw = agent.generate(observation, ["turn left", "turn right", "check available actions"], previous_action="move forward")
    validation = harness.validate_planner_json(raw, ["turn left", "turn right", "check available actions"])
    assert validation.valid is True
    assert validation.message.action_hint in {"turn_left", "turn_right", "done"}


def test_aggregate_summary_and_gate_require_full_pass(tmp_path: Path):
    passed = harness.run_babyai_rollout_episode(
        item_id=1,
        env_client=FakeBabyAIEnv(),
        planner_agent=harness.ScriptedPlannerAgent(),
        executor_agent=harness.ScriptedExecutorAgent(),
        max_steps=1,
        planner_interval=1,
    )
    failed = harness.run_babyai_rollout_episode(
        item_id=2,
        env_client=FakeBabyAIEnv(),
        planner_agent=InvalidPlanner(),
        executor_agent=harness.ScriptedExecutorAgent(),
        max_steps=1,
        planner_interval=1,
    )
    summary_path = tmp_path / "aggregate.json"
    harness.write_aggregate_summary([passed, failed], summary_path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    try:
        harness.ensure_harness_passed(summary_path)
    except RuntimeError as exc:
        assert "Rollout harness gate failed" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ensure_harness_passed to reject a failed summary")
