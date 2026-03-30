from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from conftest import load_multi_agent_module


ray_trainer = load_multi_agent_module(
    "verl.multi_agent.ppo.ray_trainer",
    "verl/multi_agent/ppo/ray_trainer.py",
)


def test_two_agent_metrics_omit_reviewer_and_rewrite_fields_when_absent(monkeypatch):
    monkeypatch.setattr(ray_trainer, "compute_single_agent_metrics", lambda batch, use_critic=True: {"base": 1.0})

    batch = SimpleNamespace(
        batch={
            "planner_response_mask": torch.ones((2, 3), dtype=torch.int64),
            "executor_response_mask": torch.ones((2, 4), dtype=torch.int64),
            "reward_event_mask": torch.zeros((2, 4), dtype=torch.int64),
            "team_env_rounds": torch.tensor([1.0, 2.0]),
            "task_rounds": torch.tensor([1.0, 2.0]),
            "executor_action_valid": torch.ones(2),
            "executor_native_format_valid": torch.ones(2),
            "planner_output_valid": torch.ones(2),
            "planner_fallback_used": torch.zeros(2),
            "planner_rewrite_used": torch.zeros(2),
            "planner_tag_only": torch.zeros(2),
            "planner_exact_action": torch.tensor([1.0, 0.0]),
            "planner_degenerate_fragment": torch.tensor([0.0, 1.0]),
            "planner_message_token_count": torch.tensor([4.0, 6.0]),
            "executor_first_pass_valid": torch.tensor([1.0, 0.0]),
            "executor_retry_used": torch.tensor([0.0, 1.0]),
            "executor_retry_resolved": torch.tensor([0.0, 1.0]),
            "executor_retry_count": torch.tensor([0.0, 2.0]),
            "executor_action_changed_after_retry": torch.tensor([0.0, 1.0]),
            "invalid_format_terminated": torch.zeros(2),
            "invalid_action_terminated": torch.zeros(2),
            "env_step_failed": torch.zeros(2),
            "timeout_occurred": torch.zeros(2),
        },
        non_tensor_batch={"item_id": ["babyai_0", "babyai_1"]},
    )

    metrics = ray_trainer.compute_data_metrics(batch, use_critic=False)

    assert metrics["base"] == 1.0
    assert "planner_reviewer_tokens/mean" not in metrics
    assert "executor_reviewer_tokens/mean" not in metrics
    assert "planner_reviewer_retry_mean/babyai" not in metrics
    assert "executor_reviewer_pass_rate/babyai" not in metrics
    assert metrics["planner_rewrite_rate/babyai"] == 0.0
    assert metrics["planner_exact_action_rate/babyai"] == 0.5
    assert metrics["planner_degenerate_fragment_rate/babyai"] == 0.5
    assert metrics["planner_avg_token_length/babyai"] == 5.0
    assert metrics["executor_first_pass_valid_rate/babyai"] == 0.5
    assert metrics["executor_retry_used_rate/babyai"] == 0.5
    assert metrics["executor_retry_resolved_rate/babyai"] == 0.5
    assert metrics["executor_retry_count_mean/babyai"] == 1.0
    assert metrics["executor_action_changed_after_retry_rate/babyai"] == 0.5
