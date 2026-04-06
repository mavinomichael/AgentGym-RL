from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List

from .rewarding import trailing_median


@dataclass(frozen=True)
class CollapseState:
    status: str
    reasons: List[str]


@dataclass
class CollapseMonitor:
    warning_window: int = 2
    milestone_history_window: int = 20
    warning_counts: Dict[str, int] = field(default_factory=dict)
    milestone_history: Deque[float] = field(default_factory=lambda: deque(maxlen=20))

    def update(self, metrics: Dict[str, float]) -> CollapseState:
        reasons: List[str] = []
        warning_reasons: List[str] = []

        milestone_rate = float(metrics.get("milestone_hit_rate", 0.0))
        self.milestone_history.append(milestone_rate)
        trailing = trailing_median(self.milestone_history)

        collapse_checks = {
            "nan_actor_loss": any(math.isnan(float(metrics.get(key, 0.0))) for key in ("actor/kl_loss", "actor/pg_loss", "actor/entropy_loss")),
            "planner_json_validity<0.80": float(metrics.get("planner_json_validity", 1.0)) < 0.80,
            "executor_legal_action_rate<0.90": float(metrics.get("executor_legal_action_rate", 1.0)) < 0.90,
            "tag_only_planner_output": float(metrics.get("planner_tag_only_rate", 0.0)) > 0.0,
            "invalid_executor_outputs>0.10": float(metrics.get("executor_invalid_output_rate", 0.0)) > 0.10,
        }
        for reason, triggered in collapse_checks.items():
            if triggered:
                reasons.append(reason)

        warning_checks = {
            "planner_json_validity<0.95": float(metrics.get("planner_json_validity", 1.0)) < 0.95,
            "executor_legal_action_rate<0.98": float(metrics.get("executor_legal_action_rate", 1.0)) < 0.98,
            "fallback_usage>0.05": float(metrics.get("planner_fallback_rate", 0.0)) > 0.05,
            "milestone_hit_rate_collapsed": trailing > 0 and milestone_rate < 0.5 * trailing,
        }
        for reason, triggered in warning_checks.items():
            count = self.warning_counts.get(reason, 0)
            count = count + 1 if triggered else 0
            self.warning_counts[reason] = count
            if count >= self.warning_window:
                warning_reasons.append(reason)

        if reasons:
            return CollapseState(status="collapse", reasons=reasons)
        if warning_reasons:
            return CollapseState(status="warning", reasons=warning_reasons)
        return CollapseState(status="ok", reasons=[])
