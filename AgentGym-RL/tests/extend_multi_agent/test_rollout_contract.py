from conftest import load_extend_module


planner_executor = load_extend_module(
    "verl.extend_multi_agent.workers.rollout.agent_vllm_rollout.planner_executor",
    "verl/extend_multi_agent/workers/rollout/agent_vllm_rollout/planner_executor.py",
)


class DummyEnv:
    def __init__(self):
        self.calls = []

    def step(self, action: str):
        self.calls.append(action)
        return {"state": "ok", "reward": 0.0, "done": False}


def _run_single_turn(observation: str, planner_raw: str, executor_raw: str, phase):
    planner_calls = 0
    executor_calls = 0
    env = DummyEnv()

    planner_calls += 1
    planner_validation = planner_executor.validate_planner_payload(planner_raw, observation=observation)
    if not planner_validation.valid:
        return {
            "planner_calls": planner_calls,
            "executor_calls": executor_calls,
            "env_calls": len(env.calls),
            "failure": "planner_invalid_json",
        }

    executor_calls += 1
    executor_validation = planner_executor.validate_executor_payload(executor_raw, observation, planner_executor.get_task_profile("babyai"))
    if not executor_validation.valid:
        return {
            "planner_calls": planner_calls,
            "executor_calls": executor_calls,
            "env_calls": len(env.calls),
            "failure": "executor_invalid_json",
        }

    env_action = planner_executor.adapt_executor_payload_to_env_action(executor_raw, observation)
    env.step(env_action)
    return {
        "planner_calls": planner_calls,
        "executor_calls": executor_calls,
        "env_calls": len(env.calls),
        "failure": None,
        "env_action": env_action,
    }


def test_both_roles_are_invoked_during_rollout_even_when_one_role_is_frozen():
    observation = 'obs\nAvailable actions: ["turn left", "move forward"]'
    result = _run_single_turn(
        observation,
        '{"subgoal":"face target","constraints":["stay grounded"],"done":false}',
        '{"action":"turn left","action_args":null,"done":false}',
        None,
    )
    assert result["planner_calls"] == 1
    assert result["executor_calls"] == 1
    assert result["env_calls"] == 1
    assert result["failure"] is None


def test_invalid_planner_json_terminates_without_env_step():
    observation = 'obs\nAvailable actions: ["turn left", "move forward"]'
    result = _run_single_turn(
        observation,
        '{"subgoal":"broken","done":false}',
        '{"action":"turn left","action_args":null,"done":false}',
        None,
    )
    assert result["executor_calls"] == 0
    assert result["env_calls"] == 0
    assert result["failure"] == "planner_invalid_json"


def test_invalid_executor_json_terminates_without_env_step():
    observation = 'obs\nAvailable actions: ["turn left", "move forward"]'
    result = _run_single_turn(
        observation,
        '{"subgoal":"face target","constraints":["stay grounded"],"done":false}',
        '{"action":"turn right","action_args":null,"done":false}',
        None,
    )
    assert result["planner_calls"] == 1
    assert result["executor_calls"] == 1
    assert result["env_calls"] == 0
    assert result["failure"] == "executor_invalid_json"


def test_valid_executor_json_reaches_single_env_action_adapter_and_calls_env_step():
    observation = 'obs\nAvailable actions: ["turn left", "move forward"]'
    result = _run_single_turn(
        observation,
        '{"subgoal":"face target","constraints":["stay grounded"],"done":false}',
        '{"action":"move forward","action_args":null,"done":false}',
        None,
    )
    assert result["env_calls"] == 1
    assert result["env_action"] == "move forward"
