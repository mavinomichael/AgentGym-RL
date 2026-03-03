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


def test_turn_prompts_are_role_specific_without_role_prefixes():
    observation = "obs"
    profile = planner_executor.get_task_profile("webarena")
    planner_prompt = planner_executor.build_planner_turn_prompt(observation, profile)
    executor_prompt = planner_executor.build_executor_turn_prompt(observation, "Use the search box.", profile)
    assert "Planner Turn" in planner_prompt
    assert "Do not emit an environment action" in planner_prompt
    assert "Executor Turn" in executor_prompt
    assert "Use the search box." in executor_prompt
    assert "Do not prepend role labels" in executor_prompt


def test_executor_payload_passthrough_and_validation_are_environment_aware():
    searchqa_profile = planner_executor.get_task_profile("searchqa")
    webarena_profile = planner_executor.get_task_profile("webarena")
    searchqa_payload = "  <search>water cycle</search>\n"
    webarena_payload = "Reasoning\n```click [12]```\n"

    assert planner_executor.normalize_executor_payload(searchqa_payload, searchqa_profile) == "<search>water cycle</search>"
    assert planner_executor.normalize_executor_payload(webarena_payload, webarena_profile) == "Reasoning\n```click [12]```"
    assert planner_executor.is_executor_payload_valid(searchqa_payload, searchqa_profile)
    assert planner_executor.is_executor_payload_valid(webarena_payload, webarena_profile)
    assert not planner_executor.is_executor_payload_valid("Executor: click [12]", searchqa_profile)


def test_invalid_action_detection_and_reward_delta():
    sciworld_profile = planner_executor.get_task_profile("sciworld")
    webarena_profile = planner_executor.get_task_profile("webarena")
    assert planner_executor.detect_invalid_action("Invalid Action. Try again.", sciworld_profile)
    assert planner_executor.detect_invalid_action("TimeoutError while stepping browser env", webarena_profile)
    assert planner_executor.compute_reward_delta(1.5, 3.0) == 1.5
