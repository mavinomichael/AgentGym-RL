from conftest import load_multi_agent_module


planner_executor = load_multi_agent_module(
    "verl.multi_agent.workers.rollout.agent_vllm_rollout.planner_executor",
    "verl/multi_agent/workers/rollout/agent_vllm_rollout/planner_executor.py",
)
rollout_schemas = load_multi_agent_module(
    "verl.multi_agent.workers.rollout.schemas",
    "verl/multi_agent/workers/rollout/schemas.py",
)


class DummyTokenizer:
    def __init__(self):
        self.last_conversations = None

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(self, conversations, add_generation_prompt=True, tokenize=True):
        self.last_conversations = conversations
        rendered = "".join(f"<{item['role']}>{item['content']}" for item in conversations)
        if add_generation_prompt:
            rendered += "<assistant>"
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered


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
    assert "Do not output role labels or sections" in planner_prompt
    assert "Output only the guidance sentence." in planner_prompt
    assert "Keep it under 12 words." in planner_prompt
    assert "Executor Turn" in executor_prompt
    assert "Use the search box." in executor_prompt
    assert "Do not prepend role labels" in executor_prompt


def test_planner_payload_validation_rejects_tags_and_accepts_sentence():
    valid = planner_executor.validate_planner_payload("Turn left and inspect the doorway.")
    assert valid.valid
    assert valid.reason == "ok"
    assert valid.message == "Turn left and inspect the doorway."

    tag_only = planner_executor.validate_planner_payload("[PLANNER]\n[PLANNER]\n[PL]")
    assert not tag_only.valid
    assert tag_only.reason == "tag_only"

    with_schema = planner_executor.validate_planner_payload("Thought:\nMove closer.\n\nAction:\nturn left")
    assert not with_schema.valid
    assert with_schema.reason == "contains_role_or_schema_tokens"

    too_long = planner_executor.validate_planner_payload(
        "Check if there is a red ball in the room or if you need to move to another part of the room."
    )
    assert not too_long.valid
    assert too_long.reason == "too_long"


def test_planner_retry_prompt_restates_short_sentence_constraint():
    retry_prompt = planner_executor.build_planner_retry_prompt(
        observation="goal obs",
        invalid_planner_output="Check if there is a red ball in the room or if you need to move to another part of the room.",
        validation_reason="too_long",
        task_profile=planner_executor.get_task_profile("babyai"),
    )
    assert "[Planner Retry]" in retry_prompt
    assert "Failure reason: too_long" in retry_prompt
    assert "will not be shown to the Executor" in retry_prompt
    assert "Keep it under 12 words." in retry_prompt
    assert "Output only the guidance sentence." in retry_prompt


def test_planner_normalization_strips_control_headers_only():
    payload = "[Planner Message]\n[PLANNER]\nTurn left and inspect the doorway."
    assert planner_executor.normalize_planner_payload(payload) == "Turn left and inspect the doorway."


def test_babyai_executor_prompt_includes_strict_action_constraints():
    observation = 'obs\nAvailable actions: ["turn left", "turn right", "move forward"]'
    profile = planner_executor.get_task_profile("babyai")
    executor_prompt = planner_executor.build_executor_turn_prompt(observation, "Go near the key.", profile)
    assert "BabyAI strict rules" in executor_prompt
    assert "Action must be exactly one of" in executor_prompt
    assert "Go, Up, Left, or Right" in executor_prompt
    assert "[Executor Response]" in executor_prompt


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
    assert "[Executor Response]" in retry_prompt


def test_executor_normalization_strips_control_headers_only():
    profile = planner_executor.get_task_profile("babyai")
    payload = "[Executor Response]\n[Executor]\nThought:\nTurn right.\n\nAction:\nturn right"
    assert planner_executor.normalize_executor_payload(payload, profile) == "Thought:\nTurn right.\n\nAction:\nturn right"


def test_planner_fallback_message_is_stable():
    assert (
        planner_executor.planner_fallback_message()
        == "Planner guidance unavailable. Infer the next step from the observation only."
    )


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


def test_generation_prompt_uses_prompt_content_without_mutating_raw_message():
    tokenizer = DummyTokenizer()
    message = rollout_schemas.Message(
        role="assistant",
        content="Turn left and inspect the doorway.",
        speaker="planner",
        prompt_content="Turn left and inspect the doorway.",
    )
    handler = rollout_schemas.RolloutHandler(
        messages=[message],
        task_name="babyai",
        item_id=1,
        score=0.0,
        done=False,
        input_ids=[],
        prompt_ids=[],
        response_ids=[],
        attention_mask=[],
        prompt_attention_mask=[],
        response_attention_mask=[],
        position_ids=[],
        prompt_position_ids=[],
        response_position_ids=[],
        loss_mask=[],
        prompt_loss_mask=[],
        response_loss_mask=[],
        speaker_ids=[],
        prompt_speaker_ids=[],
        response_speaker_ids=[],
        planner_response_mask=[],
        prompt_planner_response_mask=[],
        response_planner_response_mask=[],
        executor_response_mask=[],
        prompt_executor_response_mask=[],
        response_executor_response_mask=[],
        score_values=[],
        prompt_score_values=[],
        response_score_values=[],
        reward_event_mask=[],
        prompt_reward_event_mask=[],
        response_reward_event_mask=[],
    )

    handler.get_generation_prompt(tokenizer)

    assert handler.messages[0].content == "Turn left and inspect the doorway."
    assert tokenizer.last_conversations == [{"role": "assistant", "content": "Turn left and inspect the doorway."}]


def test_assistant_prompt_prefix_is_zero_loss_context():
    tokenizer = DummyTokenizer()
    suffix_token_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    handler = rollout_schemas.RolloutHandler(
        messages=[],
        task_name="babyai",
        item_id=1,
        score=0.0,
        done=False,
        input_ids=list(suffix_token_ids),
        prompt_ids=list(suffix_token_ids),
        response_ids=[],
        attention_mask=[1] * len(suffix_token_ids),
        prompt_attention_mask=[1] * len(suffix_token_ids),
        response_attention_mask=[],
        position_ids=list(range(len(suffix_token_ids))),
        prompt_position_ids=list(range(len(suffix_token_ids))),
        response_position_ids=[],
        loss_mask=[0] * len(suffix_token_ids),
        prompt_loss_mask=[0] * len(suffix_token_ids),
        response_loss_mask=[],
        speaker_ids=[0] * len(suffix_token_ids),
        prompt_speaker_ids=[0] * len(suffix_token_ids),
        response_speaker_ids=[],
        planner_response_mask=[0] * len(suffix_token_ids),
        prompt_planner_response_mask=[0] * len(suffix_token_ids),
        response_planner_response_mask=[],
        executor_response_mask=[0] * len(suffix_token_ids),
        prompt_executor_response_mask=[0] * len(suffix_token_ids),
        response_executor_response_mask=[],
        score_values=[0.0] * len(suffix_token_ids),
        prompt_score_values=[0.0] * len(suffix_token_ids),
        response_score_values=[],
        reward_event_mask=[0] * len(suffix_token_ids),
        prompt_reward_event_mask=[0] * len(suffix_token_ids),
        response_reward_event_mask=[],
    )

    raw_content = "Turn left and inspect the doorway."
    prompt_content = "Context:\nTurn left and inspect the doorway."
    handler.add_assistant_message(
        tokenizer,
        raw_content,
        rollout_schemas.PLANNER_SPEAKER_ID,
        prompt_content=prompt_content,
    )

    added_loss_mask = handler.loss_mask[len(suffix_token_ids) :]
    assistant_prefix_len = len(tokenizer.encode("\n<|im_start|>assistant\n", add_special_tokens=False))
    planner_prompt_prefix_len = len(tokenizer.encode("Context:\n", add_special_tokens=False))
    raw_content_len = len(tokenizer.encode(raw_content, add_special_tokens=False))

    assert all(mask == 0 for mask in added_loss_mask[: assistant_prefix_len + planner_prompt_prefix_len])
    assert all(mask == 1 for mask in added_loss_mask[
        assistant_prefix_len + planner_prompt_prefix_len : assistant_prefix_len + planner_prompt_prefix_len + raw_content_len
    ])


def test_non_trainable_planner_context_keeps_history_but_zeroes_loss():
    tokenizer = DummyTokenizer()
    suffix_token_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    handler = rollout_schemas.RolloutHandler(
        messages=[],
        task_name="babyai",
        item_id=1,
        score=0.0,
        done=False,
        input_ids=list(suffix_token_ids),
        prompt_ids=list(suffix_token_ids),
        response_ids=[],
        attention_mask=[1] * len(suffix_token_ids),
        prompt_attention_mask=[1] * len(suffix_token_ids),
        response_attention_mask=[],
        position_ids=list(range(len(suffix_token_ids))),
        prompt_position_ids=list(range(len(suffix_token_ids))),
        response_position_ids=[],
        loss_mask=[0] * len(suffix_token_ids),
        prompt_loss_mask=[0] * len(suffix_token_ids),
        response_loss_mask=[],
        speaker_ids=[0] * len(suffix_token_ids),
        prompt_speaker_ids=[0] * len(suffix_token_ids),
        response_speaker_ids=[],
        planner_response_mask=[0] * len(suffix_token_ids),
        prompt_planner_response_mask=[0] * len(suffix_token_ids),
        response_planner_response_mask=[],
        executor_response_mask=[0] * len(suffix_token_ids),
        prompt_executor_response_mask=[0] * len(suffix_token_ids),
        response_executor_response_mask=[],
        score_values=[0.0] * len(suffix_token_ids),
        prompt_score_values=[0.0] * len(suffix_token_ids),
        response_score_values=[],
        reward_event_mask=[0] * len(suffix_token_ids),
        prompt_reward_event_mask=[0] * len(suffix_token_ids),
        response_reward_event_mask=[],
    )

    handler.add_assistant_message(
        tokenizer,
        "Planner guidance unavailable. Infer the next step from the observation only.",
        rollout_schemas.PLANNER_SPEAKER_ID,
        trainable=False,
    )

    assert handler.messages[-1].content.startswith("Planner guidance unavailable")
    assert sum(handler.planner_response_mask) == 0
    assert sum(handler.loss_mask) == 0
