from conftest import load_multi_agent_module


planner_executor = load_multi_agent_module(
    "verl.multi_agent.workers.rollout.agent_vllm_rollout.planner_executor",
    "verl/multi_agent/workers/rollout/agent_vllm_rollout/planner_executor.py",
)


def test_multi_agent_instruction_wraps_original_prompt():
    base_prompt = "Original env instructions"
    wrapped = planner_executor.build_multi_agent_instruction(base_prompt)
    assert "Planner" in wrapped
    assert "Executor" in wrapped
    assert base_prompt in wrapped


def test_turn_prompts_are_role_specific():
    observation = "obs"
    planner_prompt = planner_executor.build_planner_turn_prompt(observation)
    executor_prompt = planner_executor.build_executor_turn_prompt(observation, "Planner: do x")
    assert "Planner Turn" in planner_prompt
    assert "Do not output an environment action" in planner_prompt
    assert "Executor Turn" in executor_prompt
    assert "Planner: do x" in executor_prompt


def test_strip_executor_prefix_and_reward_delta():
    assert planner_executor.strip_speaker_prefix("Executor: open door", "Executor") == "open door"
    assert planner_executor.strip_speaker_prefix("  Executor:  look around  ", "Executor") == "look around"
    assert planner_executor.compute_reward_delta(1.5, 3.0) == 1.5
    assert planner_executor.TEAM_ACK.startswith("Understood")
