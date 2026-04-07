from conftest import load_extend_module


planner_executor = load_extend_module(
    "verl.extend_multi_agent.workers.rollout.agent_vllm_rollout.planner_executor",
    "verl/extend_multi_agent/workers/rollout/agent_vllm_rollout/planner_executor.py",
)


def test_planner_valid_json_parses_correctly():
    profile = planner_executor.get_task_profile("babyai")
    payload = '{"subgoal":"approach the red ball","constraints":["stay grounded"],"done":false}'
    result = planner_executor.validate_planner_payload(payload, observation="obs", task_profile=profile)
    assert result.valid
    assert result.reason == "ok"
    assert result.message == payload


def test_executor_valid_json_parses_correctly():
    profile = planner_executor.get_task_profile("babyai")
    observation = 'obs\nAvailable actions: ["turn left", "move forward"]'
    payload = '{"action":"move forward","action_args":null,"done":false}'
    result = planner_executor.validate_executor_payload(payload, observation, profile)
    assert result.valid
    assert result.action == "move forward"


def test_invalid_planner_json_is_rejected():
    profile = planner_executor.get_task_profile("babyai")
    payload = '{"subgoal":"approach","done":false}'
    result = planner_executor.validate_planner_payload(payload, observation="obs", task_profile=profile)
    assert not result.valid
    assert result.reason == "schema_keys_mismatch"


def test_invalid_executor_json_is_rejected():
    profile = planner_executor.get_task_profile("babyai")
    observation = 'obs\nAvailable actions: ["turn left", "move forward"]'
    payload = '{"action":"jump","action_args":null,"done":false}'
    result = planner_executor.validate_executor_payload(payload, observation, profile)
    assert not result.valid
    assert result.reason == "action_not_in_available"


def test_executor_adapter_rejects_actions_not_in_available_list():
    observation = 'obs\nAvailable actions: ["turn left", "move forward"]'
    try:
        planner_executor.adapt_executor_payload_to_env_action(
            '{"action":"turn right","action_args":null,"done":false}',
            observation,
        )
    except ValueError as exc:
        assert "Cannot adapt invalid executor JSON" in str(exc)
    else:
        raise AssertionError("adapter should reject unavailable actions")


def test_executor_adapter_accepts_exact_available_action_strings_only():
    observation = 'obs\nAvailable actions: ["turn left", "move forward"]'
    action = planner_executor.adapt_executor_payload_to_env_action(
        '{"action":"turn left","action_args":null,"done":false}',
        observation,
    )
    assert action == "turn left"


def test_extra_json_keys_are_rejected_for_both_roles():
    profile = planner_executor.get_task_profile("babyai")
    observation = 'obs\nAvailable actions: ["turn left", "move forward"]'
    planner_bad = '{"subgoal":"approach","constraints":[],"done":false,"extra":1}'
    executor_bad = '{"action":"turn left","action_args":null,"done":false,"extra":1}'
    planner_result = planner_executor.validate_planner_payload(planner_bad, observation=observation, task_profile=profile)
    executor_result = planner_executor.validate_executor_payload(executor_bad, observation, profile)
    assert not planner_result.valid
    assert planner_result.reason == "schema_keys_mismatch"
    assert not executor_result.valid
    assert executor_result.reason == "schema_keys_mismatch"
