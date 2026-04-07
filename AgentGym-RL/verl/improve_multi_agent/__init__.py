"""Research-only structured multi-agent stack.

This package intentionally lives beside the legacy multi-agent implementation so
we can iterate on protocol, optimization, and staged-training ideas without
destabilizing the existing baselines.
"""

from .advantage import RoleRewardNormalizer, compute_rolewise_gae_advantage_return
from .dataset import ImproveRLHFDataset
from .monitoring import CollapseMonitor, CollapseState
from .protocol import (
    EXECUTOR_SCHEMA_VERSION,
    PLANNER_SCHEMA_VERSION,
    ExecutorDecision,
    ExecutorValidation,
    PlannerMessage,
    PlannerValidation,
    build_executor_prompt,
    build_planner_prompt,
    build_structured_bootstrap,
    render_babyai_action_payload,
    validate_executor_json,
    validate_planner_json,
)
from .rewarding import (
    ExecutorRewardBreakdown,
    PlannerRewardBreakdown,
    detect_babyai_milestones,
    detect_subgoal_completion,
)
from .rollout_harness import (
    ScriptedExecutorAgent,
    ScriptedPlannerAgent,
    aggregate_episode_records,
    ensure_harness_passed,
    run_babyai_rollout_episode,
)
from .trace_bootstrap import (
    build_executor_sft_dataset,
    build_planner_sft_dataset,
    collect_successful_trace_events,
)

__all__ = [
    "CollapseMonitor",
    "CollapseState",
    "EXECUTOR_SCHEMA_VERSION",
    "ExecutorDecision",
    "ExecutorRewardBreakdown",
    "ExecutorValidation",
    "ImproveRLHFDataset",
    "PLANNER_SCHEMA_VERSION",
    "PlannerMessage",
    "PlannerRewardBreakdown",
    "PlannerValidation",
    "RoleRewardNormalizer",
    "build_executor_prompt",
    "build_executor_sft_dataset",
    "build_planner_prompt",
    "build_planner_sft_dataset",
    "build_structured_bootstrap",
    "collect_successful_trace_events",
    "compute_rolewise_gae_advantage_return",
    "detect_babyai_milestones",
    "detect_subgoal_completion",
    "aggregate_episode_records",
    "ensure_harness_passed",
    "render_babyai_action_payload",
    "run_babyai_rollout_episode",
    "ScriptedExecutorAgent",
    "ScriptedPlannerAgent",
    "validate_executor_json",
    "validate_planner_json",
]
