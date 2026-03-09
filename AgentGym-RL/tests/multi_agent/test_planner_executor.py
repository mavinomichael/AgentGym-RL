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


def test_babyai_executor_prompt_includes_strict_action_constraints():
    observation = 'obs\nAvailable actions: ["turn left", "turn right", "move forward"]'
    profile = planner_executor.get_task_profile("babyai")
    executor_prompt = planner_executor.build_executor_turn_prompt(observation, "Go near the key.", profile)
    assert "BabyAI strict rules" in executor_prompt
    assert "Action must be exactly one of" in executor_prompt
    assert "Go, Up, Left, or Right" in executor_prompt


def test_babyai_executor_retry_prompt_restates_schema_and_allowed_actions():
    observation = 'obs\nAvailable actions: ["turn left", "turn right", "move forward"]'
    profile = planner_executor.get_task_profile("babyai")
    retry_prompt = planner_executor.build_executor_retry_prompt(
        observation=observation,
        planner_message="[PLANNER]\nAction:\nturn right",
        invalid_executor_output="Go",
        validation_reason="invalid_format",
        task_profile=profile,
    )
    assert "[Executor Retry]" in retry_prompt
    assert "Failure reason: invalid_format" in retry_prompt
    assert "Thought:" in retry_prompt
    assert "Action:" in retry_prompt
    assert "Action must be exactly one of: turn left, turn right, move forward" in retry_prompt
    assert "Do not output bare words like Go, Up, Left, or Right." in retry_prompt


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


def test_babyai_valid_payload_and_available_action_passes_validation():
    profile = planner_executor.get_task_profile("babyai")
    observation = 'obs\nAvailable actions: ["turn left", "turn right", "move forward"]'
    payload = "Thought:\nI should rotate.\n\nAction:\nturn left"
    result = planner_executor.validate_executor_payload(payload, observation, profile)
    assert result.valid
    assert result.reason == "ok"
    assert result.action == "turn left"


def test_babyai_bare_token_is_invalid_format_without_coercion():
    profile = planner_executor.get_task_profile("babyai")
    observation = 'obs\nAvailable actions: ["turn left", "turn right", "move forward"]'
    payload = "Go"
    result = planner_executor.validate_executor_payload(payload, observation, profile)
    assert not result.valid
    assert result.reason == "invalid_format"
    assert result.action is None
    assert planner_executor.normalize_executor_payload(payload, profile) == "Go"


def test_babyai_action_not_in_available_list_is_invalid():
    profile = planner_executor.get_task_profile("babyai")
    observation = 'obs\nAvailable actions: ["turn left", "turn right", "move forward"]'
    payload = "Thought:\nI will pick up.\n\nAction:\npick up key 1"
    result = planner_executor.validate_executor_payload(payload, observation, profile)
    assert not result.valid
    assert result.reason == "action_not_in_available"
    assert result.action == "pick up key 1"


def test_non_babyai_validation_path_is_unchanged():
    profile = planner_executor.get_task_profile("searchqa")
    payload = "<search>water cycle</search>"
    result = planner_executor.validate_executor_payload(payload, "observation", profile)
    assert result.valid
    assert result.reason == "ok"


def test_invalid_action_detection_and_reward_delta():
    sciworld_profile = planner_executor.get_task_profile("sciworld")
    webarena_profile = planner_executor.get_task_profile("webarena")
    assert planner_executor.detect_invalid_action("Invalid Action. Try again.", sciworld_profile)
    assert planner_executor.detect_invalid_action("TimeoutError while stepping browser env", webarena_profile)
    assert planner_executor.compute_reward_delta(1.5, 3.0) == 1.5
