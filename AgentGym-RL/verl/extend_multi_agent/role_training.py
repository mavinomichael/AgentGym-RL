from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import torch


@dataclass(frozen=True)
class RolePhase:
    train_role: str = "joint"
    freeze_planner: bool = False
    freeze_executor: bool = False

    @property
    def planner_trainable(self) -> bool:
        return self.train_role in {"planner", "joint"} and not self.freeze_planner

    @property
    def executor_trainable(self) -> bool:
        return self.train_role in {"executor", "joint"} and not self.freeze_executor


ROLE_MASK_KEY = {
    "planner": "planner_response_mask",
    "executor": "executor_response_mask",
}


def resolve_role_phase(train_role: str, freeze_planner: bool, freeze_executor: bool) -> RolePhase:
    normalized = str(train_role).strip().lower()
    if normalized not in {"planner", "executor", "joint"}:
        raise ValueError(f"Unsupported train_role: {train_role}")
    phase = RolePhase(train_role=normalized, freeze_planner=bool(freeze_planner), freeze_executor=bool(freeze_executor))
    if not phase.planner_trainable and not phase.executor_trainable:
        raise ValueError("Both planner and executor are frozen; nothing would train.")
    return phase


def clone_batch_with_role_weights(batch: Mapping[str, torch.Tensor], role: str) -> Dict[str, torch.Tensor]:
    role_key = ROLE_MASK_KEY[role]
    cloned = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    response_mask = cloned["response_mask"].float()
    role_mask = cloned[role_key].float()
    weighted = response_mask * role_mask
    if "ppo_loss_weights" in cloned:
        cloned["ppo_loss_weights"] = cloned["ppo_loss_weights"].float() * role_mask
    else:
        cloned["ppo_loss_weights"] = role_mask
    if "kl_loss_weights" in cloned:
        cloned["kl_loss_weights"] = cloned["kl_loss_weights"].float() * role_mask
    cloned["response_mask"] = response_mask.to(dtype=cloned["response_mask"].dtype)
    return cloned


def merge_role_log_probs(
    planner_log_probs: torch.Tensor,
    executor_log_probs: torch.Tensor,
    planner_mask: torch.Tensor,
    executor_mask: torch.Tensor,
) -> torch.Tensor:
    planner_weight = planner_mask.float()
    executor_weight = executor_mask.float()
    return planner_log_probs * planner_weight + executor_log_probs * executor_weight


def optimizer_param_names(named_parameters: Iterable[tuple[str, torch.nn.Parameter]], optimizer: Optional[torch.optim.Optimizer]) -> set[str]:
    if optimizer is None:
        return set()
    id_to_name = {id(param): name for name, param in named_parameters}
    names = set()
    for group in optimizer.param_groups:
        for param in group["params"]:
            if id(param) in id_to_name:
                names.add(id_to_name[id(param)])
    return names


def assert_optimizer_excludes_module(module: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer]) -> None:
    if optimizer is None:
        return
    module_param_ids = {id(param) for param in module.parameters()}
    for group in optimizer.param_groups:
        for param in group["params"]:
            if id(param) in module_param_ids:
                raise AssertionError("Frozen module parameter leaked into active optimizer.")


def assert_module_has_no_gradients(module: torch.nn.Module) -> None:
    for param in module.parameters():
        if param.grad is None:
            continue
        if torch.count_nonzero(param.grad).item() != 0:
            raise AssertionError("Frozen module received a non-zero gradient.")


def parameter_snapshot(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: param.detach().cpu().clone() for name, param in module.named_parameters()}


def parameters_changed(before: Mapping[str, torch.Tensor], module: torch.nn.Module) -> bool:
    for name, param in module.named_parameters():
        if name not in before:
            return True
        if not torch.equal(before[name], param.detach().cpu()):
            return True
    return False


def grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        total += float(torch.sum(param.grad.detach().float() ** 2).item())
    return total ** 0.5


def apply_role_phase_step(
    *,
    planner_module: torch.nn.Module,
    executor_module: torch.nn.Module,
    planner_loss: torch.Tensor,
    executor_loss: torch.Tensor,
    planner_optimizer: Optional[torch.optim.Optimizer],
    executor_optimizer: Optional[torch.optim.Optimizer],
    phase: RolePhase,
) -> Dict[str, float]:
    if planner_optimizer is not None:
        planner_optimizer.zero_grad()
    if executor_optimizer is not None:
        executor_optimizer.zero_grad()

    metrics = {
        "planner_loss": 0.0,
        "executor_loss": 0.0,
        "planner_grad_norm": 0.0,
        "executor_grad_norm": 0.0,
    }

    if phase.planner_trainable:
        metrics["planner_loss"] = float(planner_loss.detach().item())
        planner_loss.backward(retain_graph=phase.executor_trainable)
    else:
        assert_optimizer_excludes_module(planner_module, planner_optimizer)

    if phase.executor_trainable:
        metrics["executor_loss"] = float(executor_loss.detach().item())
        executor_loss.backward()
    else:
        assert_optimizer_excludes_module(executor_module, executor_optimizer)

    if planner_optimizer is not None and phase.planner_trainable:
        metrics["planner_grad_norm"] = grad_norm(planner_module)
        planner_optimizer.step()
    else:
        assert_module_has_no_gradients(planner_module)

    if executor_optimizer is not None and phase.executor_trainable:
        metrics["executor_grad_norm"] = grad_norm(executor_module)
        executor_optimizer.step()
    else:
        assert_module_has_no_gradients(executor_module)

    return metrics
