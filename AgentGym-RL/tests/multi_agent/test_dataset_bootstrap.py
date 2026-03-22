from types import SimpleNamespace

from conftest import load_multi_agent_module


planner_executor = load_multi_agent_module(
    "verl.multi_agent.workers.rollout.agent_vllm_rollout.planner_executor",
    "verl/multi_agent/workers/rollout/agent_vllm_rollout/planner_executor.py",
)


def test_bootstrap_preserves_original_instruction_and_assistant_acknowledgement():
    env_client = SimpleNamespace(
        conversation_start=(
            {"from": "human", "loss": None, "value": "Original task instruction."},
            {"from": "gpt", "loss": False, "value": "Ok."},
        )
    )
    profile = planner_executor.get_task_profile("searchqa")
    messages, prompt = planner_executor.build_multi_agent_bootstrap(env_client, profile)
    assert messages[0]["role"] == "user"
    assert "Original task instruction." in messages[0]["content"]
    assert "Planner" in messages[0]["content"]
    assert "Executor" in messages[0]["content"]
    assert messages[1]["content"] == "Ok."
    assert "Planner" in prompt
    assert "Original task instruction." in prompt


def test_babyai_bootstrap_uses_plain_two_agent_instruction_without_reviewers():
    env_client = SimpleNamespace(
        conversation_start=(
            {"from": "human", "loss": None, "value": "Original BabyAI instruction."},
            {"from": "gpt", "loss": False, "value": "Ok."},
        )
    )
    profile = planner_executor.get_task_profile("babyai")
    messages, prompt = planner_executor.build_multi_agent_bootstrap(env_client, profile)
    assert "Original BabyAI instruction." not in messages[0]["content"]
    assert "part of an exploration team" in messages[0]["content"]
    assert "role-specific instructions" in messages[0]["content"]
    assert "[" not in prompt
