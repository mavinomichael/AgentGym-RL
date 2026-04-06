import json

import pytest

from conftest import load_improve_multi_agent_module


protocol = load_improve_multi_agent_module(
    "verl.improve_multi_agent.protocol",
    "verl/improve_multi_agent/protocol.py",
)
rewarding = load_improve_multi_agent_module(
    "verl.improve_multi_agent.rewarding",
    "verl/improve_multi_agent/rewarding.py",
)
monitoring = load_improve_multi_agent_module(
    "verl.improve_multi_agent.monitoring",
    "verl/improve_multi_agent/monitoring.py",
)
rollout_module = load_improve_multi_agent_module(
    "verl.improve_multi_agent.workers.rollout.agent_vllm_rollout.vllm_rollout",
    "verl/improve_multi_agent/workers/rollout/agent_vllm_rollout/vllm_rollout.py",
)


def test_planner_validation_accepts_schema_valid_json():
    legal_actions = ["turn left", "turn right", "move forward", "pick up red key 1"]
    payload = {
        "schema_version": "v1",
        "subgoal_id": "pickup",
        "target": {"object_type": "key", "color": "red", "location_hint": "front"},
        "action_hint": "pickup",
        "success_check": "carrying red key",
        "confidence": 0.7,
    }
    validation = protocol.validate_planner_json(json.dumps(payload), legal_actions)
    assert validation.valid is True
    assert validation.message.subgoal_id == "pickup"


def test_planner_validation_rejects_chain_of_thought_tokens():
    legal_actions = ["turn left", "turn right", "move forward"]
    payload = (
        '{"schema_version":"v1","subgoal_id":"explore","target":{"object_type":"none","color":"none","location_hint":"unknown"},'
        '"action_hint":"turn_left","success_check":"see more tiles","confidence":0.5}\n'
        "Let's think step-by-step."
    )
    validation = protocol.validate_planner_json(payload, legal_actions)
    assert validation.valid is False
    assert validation.reason == "contains_disallowed_tokens"


def test_executor_validation_requires_legal_action_index():
    legal_actions = ["turn left", "turn right", "move forward"]
    validation = protocol.validate_executor_json('{"schema_version":"v1","reason":"Turn to scout.","action_id":1}', legal_actions)
    assert validation.valid is True
    assert validation.action == "turn right"
    invalid = protocol.validate_executor_json('{"schema_version":"v1","reason":"Too far.","action_id":9}', legal_actions)
    assert invalid.valid is False
    assert invalid.reason == "action_id_out_of_range"


def test_executor_renderer_produces_babyai_native_payload():
    decision_json = protocol.validate_executor_json(
        '{"schema_version":"v1","reason":"Turn to scout.","action_id":0}',
        ["turn left", "turn right"],
    ).decision
    rendered = protocol.render_babyai_action_payload(decision_json, ["turn left", "turn right"])
    assert rendered == "Thought:\nTurn to scout.\n\nAction:\nturn left"


def test_prompt_builders_include_structured_contracts():
    planner_prompt = protocol.build_planner_prompt("obs", legal_actions=["turn left"])
    assert '"schema_version": "v1"' in planner_prompt
    planner_message = protocol.PlannerMessage(
        schema_version="v1",
        subgoal_id="explore",
        target={"object_type": "none", "color": "none", "location_hint": "unknown"},
        action_hint="turn_left",
        success_check="observation changes",
        confidence=0.2,
    )
    executor_prompt = protocol.build_executor_prompt("obs", planner_message, ["turn left"])
    assert '"action_id":0' in executor_prompt
    assert planner_message.to_json() in executor_prompt


def test_reward_helpers_capture_milestones_and_subgoal_success():
    planner = protocol.PlannerMessage(
        schema_version="v1",
        subgoal_id="pickup",
        target={"object_type": "key", "color": "red", "location_hint": "front"},
        action_hint="pickup",
        success_check="carrying red key",
        confidence=0.8,
    )
    previous = "Your goal: pick up the red key\nYou are not carrying anything."
    current = "Your goal: pick up the red key\nYou are carrying red key 1."
    milestones = rewarding.detect_babyai_milestones(
        previous,
        current,
        planner,
        previous_score=0.0,
        current_score=0.2,
        valid_action_streak=3,
    )
    assert milestones["inventory_changed"] == 1.0
    assert milestones["positive_task_delta"] == 1.0
    assert rewarding.detect_subgoal_completion(planner, previous, current, "pick up red key 1", milestones) is True
    planner_reward = rewarding.compute_planner_reward(
        valid_json=True,
        executable_hint=True,
        subgoal_success=True,
        contradiction=False,
    )
    executor_reward = rewarding.compute_executor_reward(
        valid_json=True,
        legal_action=True,
        observation_changed=True,
        milestone_hits=milestones,
        subgoal_success=True,
    )
    assert planner_reward.total > 0
    assert executor_reward.total > 0


def test_rolewise_advantage_and_reward_normalizer():
    torch = pytest.importorskip("torch")
    advantage = load_improve_multi_agent_module(
        "verl.improve_multi_agent.advantage",
        "verl/improve_multi_agent/advantage.py",
    )
    rewards = torch.tensor([[1.0, 0.5, 0.0]], dtype=torch.float32)
    values = torch.zeros_like(rewards)
    response_mask = torch.tensor([[1, 1, 1]], dtype=torch.float32)
    planner_mask = torch.tensor([[1, 0, 0]], dtype=torch.float32)
    executor_mask = torch.tensor([[0, 1, 1]], dtype=torch.float32)
    normalizer = advantage.RoleRewardNormalizer()
    normalized = normalizer.normalize(rewards, planner_mask, executor_mask)
    advantages, returns = advantage.compute_rolewise_gae_advantage_return(
        normalized,
        values,
        response_mask,
        planner_mask,
        executor_mask,
        gamma=1.0,
        lam=0.95,
    )
    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape


def test_align_non_tensor_entries_keeps_nested_payloads_one_dimensional():
    aligned = rollout_module.vLLMRollout._align_non_tensor_entries(
        entries=[[{"role": "user"}], [{"role": "assistant"}]],
        target_size=2,
    )
    assert aligned.shape == (2,)
    assert aligned.dtype == object
    assert aligned[0] == [{"role": "user"}]


def test_collapse_monitor_detects_warning_then_collapse():
    monitor = monitoring.CollapseMonitor()
    warning = monitor.update(
        {
            "planner_json_validity": 0.94,
            "executor_legal_action_rate": 1.0,
            "milestone_hit_rate": 1.0,
            "planner_fallback_rate": 0.0,
            "executor_invalid_output_rate": 0.0,
            "planner_tag_only_rate": 0.0,
            "actor/kl_loss": 0.1,
            "actor/pg_loss": 0.1,
            "actor/entropy_loss": 0.1,
        }
    )
    assert warning.status == "ok"
    warning = monitor.update(
        {
            "planner_json_validity": 0.94,
            "executor_legal_action_rate": 1.0,
            "milestone_hit_rate": 0.4,
            "planner_fallback_rate": 0.0,
            "executor_invalid_output_rate": 0.0,
            "planner_tag_only_rate": 0.0,
            "actor/kl_loss": 0.1,
            "actor/pg_loss": 0.1,
            "actor/entropy_loss": 0.1,
        }
    )
    assert warning.status == "warning"
    collapse = monitor.update(
        {
            "planner_json_validity": 0.5,
            "executor_legal_action_rate": 0.5,
            "milestone_hit_rate": 0.1,
            "planner_fallback_rate": 0.2,
            "executor_invalid_output_rate": 0.2,
            "planner_tag_only_rate": 1.0,
            "actor/kl_loss": float("nan"),
            "actor/pg_loss": 0.1,
            "actor/entropy_loss": 0.1,
        }
    )
    assert collapse.status == "collapse"
    assert "nan_actor_loss" in collapse.reasons
