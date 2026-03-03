from conftest import load_multi_agent_module


planner_executor = load_multi_agent_module(
    "verl.multi_agent.workers.rollout.agent_vllm_rollout.planner_executor",
    "verl/multi_agent/workers/rollout/agent_vllm_rollout/planner_executor.py",
)
schemas = load_multi_agent_module(
    "verl.multi_agent.workers.rollout.schemas",
    "verl/multi_agent/workers/rollout/schemas.py",
)


class FakeTokenizer:
    def apply_chat_template(self, conversations, add_generation_prompt=True, tokenize=True):
        text = "".join(f"<{item['role']}>{item['content']}" for item in conversations)
        if add_generation_prompt:
            text += "<assistant>"
        return self.encode(text, add_special_tokens=False)

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(chr(token) for token in tokens)


def make_handler(tokenizer):
    prompt_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    return schemas.RolloutHandler(
        messages=[schemas.Message(role="user", content="bootstrap")],
        task_name="sciworld",
        item_id=1,
        score=0.0,
        done=False,
        input_ids=list(prompt_ids),
        prompt_ids=list(prompt_ids),
        response_ids=[],
        attention_mask=[1] * len(prompt_ids),
        prompt_attention_mask=[1] * len(prompt_ids),
        response_attention_mask=[],
        position_ids=list(range(len(prompt_ids))),
        prompt_position_ids=list(range(len(prompt_ids))),
        response_position_ids=[],
        loss_mask=[0] * len(prompt_ids),
        prompt_loss_mask=[0] * len(prompt_ids),
        response_loss_mask=[],
        speaker_ids=[planner_executor.CONTROL_SPEAKER_ID] * len(prompt_ids),
        prompt_speaker_ids=[planner_executor.CONTROL_SPEAKER_ID] * len(prompt_ids),
        response_speaker_ids=[],
        planner_response_mask=[0] * len(prompt_ids),
        prompt_planner_response_mask=[0] * len(prompt_ids),
        response_planner_response_mask=[],
        executor_response_mask=[0] * len(prompt_ids),
        prompt_executor_response_mask=[0] * len(prompt_ids),
        response_executor_response_mask=[],
        score_values=[0.0] * len(prompt_ids),
        prompt_score_values=[0.0] * len(prompt_ids),
        response_score_values=[],
        reward_event_mask=[0] * len(prompt_ids),
        prompt_reward_event_mask=[0] * len(prompt_ids),
        response_reward_event_mask=[],
        max_response_len=512,
        max_model_len=1024,
    )


def test_masks_are_disjoint_and_rewards_attach_to_executor_terminal_token():
    tokenizer = FakeTokenizer()
    handler = make_handler(tokenizer)
    handler.add_user_message(tokenizer, "obs")
    handler.add_user_message(tokenizer, "planner turn")
    handler.add_assistant_message(tokenizer, "Planner: search the room", planner_executor.PLANNER_SPEAKER_ID)
    handler.add_user_message(tokenizer, "executor turn")
    handler.add_assistant_message(tokenizer, "Executor: look around", planner_executor.EXECUTOR_SPEAKER_ID)
    handler.mark_last_executor_reward(2.5)
    handler.truncate_output_ids()

    planner_mask = handler.response_planner_response_mask
    executor_mask = handler.response_executor_response_mask
    reward_mask = handler.response_reward_event_mask
    reward_values = handler.response_score_values

    assert all((planner + executor) <= 1 for planner, executor in zip(planner_mask, executor_mask))
    assert sum(planner_mask) > 0
    assert sum(executor_mask) > 0
    assert sum(reward_mask) == 1
    assert abs(sum(reward_values) - 2.5) < 1e-6


def test_one_env_round_can_contain_two_assistant_turns_and_one_reward_event():
    tokenizer = FakeTokenizer()
    handler = make_handler(tokenizer)
    handler.add_user_message(tokenizer, "initial observation")
    handler.add_user_message(tokenizer, "planner prompt")
    handler.add_assistant_message(tokenizer, "Planner: collect water", planner_executor.PLANNER_SPEAKER_ID)
    handler.add_user_message(tokenizer, "executor prompt")
    handler.add_assistant_message(tokenizer, "Executor: use beaker", planner_executor.EXECUTOR_SPEAKER_ID)
    handler.mark_last_executor_reward(1.0)
    handler.team_env_rounds += 1
    handler.truncate_output_ids()

    assistant_messages = [message for message in handler.messages if message.role == "assistant"]
    assert len(assistant_messages) == 2
    assert handler.team_env_rounds == 1
    assert sum(handler.response_reward_event_mask) == 1
