from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

import verl.utils.torch_functional as verl_F


def _safe_masked_whiten(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if torch.count_nonzero(mask).item() == 0:
        return torch.zeros_like(values)
    return verl_F.masked_whiten(values, mask)


def compute_rolewise_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    planner_mask: torch.Tensor,
    executor_mask: torch.Tensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]
        for t in reversed(range(gen_len)):
            nextvalues = (
                (values[:, t + 1] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues)
                if t < gen_len - 1
                else 0.0
            )
            delta = (
                token_level_rewards[:, t] * response_mask[:, t]
                + gamma * nextvalues
                + (1 - response_mask[:, t]) * (1 - gamma) * nextvalues
                - response_mask[:, t] * values[:, t]
            )
            lastgaelam = (
                delta * response_mask[:, t]
                + gamma * lam * lastgaelam
                + (1 - response_mask[:, t]) * (1 - gamma * lam) * lastgaelam
            )
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        planner_adv = _safe_masked_whiten(advantages, planner_mask)
        executor_adv = _safe_masked_whiten(advantages, executor_mask)
        advantages = planner_adv * planner_mask + executor_adv * executor_mask
        uncovered = response_mask - torch.clamp(planner_mask + executor_mask, max=1)
        if torch.count_nonzero(uncovered).item() > 0:
            advantages = advantages + _safe_masked_whiten(advantages + values * 0, uncovered) * uncovered
    return advantages, returns


@dataclass
class _RunningStats:
    mean: float = 0.0
    var: float = 1.0
    initialized: bool = False

    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        batch_mean = float(values.mean().item())
        batch_var = float(values.var(unbiased=False).item()) if values.numel() > 1 else 1.0
        if not self.initialized:
            self.mean = batch_mean
            self.var = batch_var if batch_var > 1e-6 else 1.0
            self.initialized = True
            return
        momentum = 0.1
        self.mean = (1.0 - momentum) * self.mean + momentum * batch_mean
        self.var = (1.0 - momentum) * self.var + momentum * max(batch_var, 1e-6)


class RoleRewardNormalizer:
    def __init__(self) -> None:
        self.stats: Dict[str, _RunningStats] = {
            "planner": _RunningStats(),
            "executor": _RunningStats(),
        }

    def normalize(
        self,
        token_level_rewards: torch.Tensor,
        planner_mask: torch.Tensor,
        executor_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = token_level_rewards.clone()
        for role, mask in (("planner", planner_mask), ("executor", executor_mask)):
            active = output[mask.bool()]
            if active.numel() == 0:
                continue
            self.stats[role].update(active)
            mean = self.stats[role].mean
            std = max(self.stats[role].var ** 0.5, 1e-6)
            output[mask.bool()] = (active - mean) / std
        return output
