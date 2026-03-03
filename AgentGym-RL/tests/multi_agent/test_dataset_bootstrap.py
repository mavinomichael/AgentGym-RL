from types import SimpleNamespace

from conftest import load_multi_agent_module


planner_executor = load_multi_agent_module(
    "verl.multi_agent.workers.rollout.agent_vllm_rollout.planner_executor",
    "verl/multi_agent/workers/rollout/agent_vllm_rollout/planner_executor.py",
)


def test_bootstrap_preserves_original_instruction_and_adds_team_wrapper():
    env_client = SimpleNamespace(
        conversation_start=(
            {"from": "human", "loss": None, "value": "Original task instruction."},
            {"from": "gpt", "loss": False, "value": "Ok."},
        )
    )
    messages, prompt = planner_executor.build_multi_agent_bootstrap(env_client)
    assert messages[0]["role"] == "user"
    assert "Original task instruction." in messages[0]["content"]
    assert "Planner" in messages[0]["content"]
    assert "Executor" in messages[0]["content"]
    assert "Understood." in messages[1]["content"]
    assert "Original task instruction." in prompt
