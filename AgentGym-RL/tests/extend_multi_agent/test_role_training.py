import pytest

torch = pytest.importorskip("torch")

from conftest import load_extend_module


role_training = load_extend_module(
    "verl.extend_multi_agent.role_training",
    "verl/extend_multi_agent/role_training.py",
)


class TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


def _make_loss(module: TinyModule) -> torch.Tensor:
    inputs = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
    target = torch.tensor([[0.2], [0.1]])
    output = module(inputs)
    return torch.nn.functional.mse_loss(output, target)


def test_when_planner_is_frozen_planner_params_do_not_change_after_one_update():
    planner = TinyModule()
    executor = TinyModule()
    planner_before = role_training.parameter_snapshot(planner)
    planner_opt = None
    executor_opt = torch.optim.SGD(executor.parameters(), lr=0.1)
    phase = role_training.resolve_role_phase("executor", freeze_planner=True, freeze_executor=False)
    role_training.apply_role_phase_step(
        planner_module=planner,
        executor_module=executor,
        planner_loss=_make_loss(planner),
        executor_loss=_make_loss(executor),
        planner_optimizer=planner_opt,
        executor_optimizer=executor_opt,
        phase=phase,
    )
    assert not role_training.parameters_changed(planner_before, planner)


def test_when_executor_is_frozen_executor_params_do_not_change_after_one_update():
    planner = TinyModule()
    executor = TinyModule()
    executor_before = role_training.parameter_snapshot(executor)
    planner_opt = torch.optim.SGD(planner.parameters(), lr=0.1)
    executor_opt = None
    phase = role_training.resolve_role_phase("planner", freeze_planner=False, freeze_executor=True)
    role_training.apply_role_phase_step(
        planner_module=planner,
        executor_module=executor,
        planner_loss=_make_loss(planner),
        executor_loss=_make_loss(executor),
        planner_optimizer=planner_opt,
        executor_optimizer=executor_opt,
        phase=phase,
    )
    assert not role_training.parameters_changed(executor_before, executor)


def test_train_role_planner_skips_executor_optimizer_and_loss():
    planner = TinyModule()
    executor = TinyModule()
    planner_opt = torch.optim.SGD(planner.parameters(), lr=0.1)
    phase = role_training.resolve_role_phase("planner", freeze_planner=False, freeze_executor=True)
    metrics = role_training.apply_role_phase_step(
        planner_module=planner,
        executor_module=executor,
        planner_loss=_make_loss(planner),
        executor_loss=_make_loss(executor),
        planner_optimizer=planner_opt,
        executor_optimizer=None,
        phase=phase,
    )
    assert metrics["planner_loss"] > 0.0
    assert metrics["executor_loss"] == 0.0


def test_train_role_executor_skips_planner_optimizer_and_loss():
    planner = TinyModule()
    executor = TinyModule()
    executor_opt = torch.optim.SGD(executor.parameters(), lr=0.1)
    phase = role_training.resolve_role_phase("executor", freeze_planner=True, freeze_executor=False)
    metrics = role_training.apply_role_phase_step(
        planner_module=planner,
        executor_module=executor,
        planner_loss=_make_loss(planner),
        executor_loss=_make_loss(executor),
        planner_optimizer=None,
        executor_optimizer=executor_opt,
        phase=phase,
    )
    assert metrics["executor_loss"] > 0.0
    assert metrics["planner_loss"] == 0.0
