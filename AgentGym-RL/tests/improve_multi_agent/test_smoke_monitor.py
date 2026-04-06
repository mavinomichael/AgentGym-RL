import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MONITOR_PATH = REPO_ROOT / "scripts/improve_multi-agent/monitor_stage3_smoke.py"


def _load_monitor_module():
    spec = importlib.util.spec_from_file_location("monitor_stage3_smoke", MONITOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_monitor_parses_deadlines_and_steps():
    module = _load_monitor_module()
    assert module.parse_deadlines(["1:900", "20:3600"]) == [(1, 900), (20, 3600)]
    assert module.parse_latest_step("foo\nstep:3\nbar\nstep:12") == 12
    assert module.parse_latest_step("no steps here") is None


def test_monitor_detects_expected_failure_reasons():
    module = _load_monitor_module()
    assert module.detect_failure_reason("Traceback (most recent call last):\nboom") == "traceback"
    assert module.detect_failure_reason("ray.exceptions.ActorDiedError: worker died") == "actor_died"
    assert module.detect_failure_reason("torch.OutOfMemoryError: CUDA out of memory") == "cuda_oom"
