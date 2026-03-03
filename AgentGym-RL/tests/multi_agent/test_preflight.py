import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PRELIGHT_PATH = REPO_ROOT / "scripts/multi_agent/preflight.py"

spec = importlib.util.spec_from_file_location("multi_agent_preflight", PRELIGHT_PATH)
preflight = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = preflight
spec.loader.exec_module(preflight)


class FakeResponse:
    status = 200

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def test_preflight_server_probe_accepts_http_200(monkeypatch):
    monkeypatch.setattr(preflight.request, "urlopen", lambda *args, **kwargs: FakeResponse())
    result = preflight.check_server_url("http://127.0.0.1:36005")
    assert result.ok


def test_preflight_dataset_checks_follow_task_profile(tmp_path):
    train_file = tmp_path / "AgentItemId" / "webarena_train.json"
    train_file.parent.mkdir(parents=True)
    train_file.write_text("[]", encoding="utf-8")

    train_results = list(preflight.check_dataset_paths("webarena", "train", tmp_path))
    assert all(result.ok for result in train_results)

    eval_dir = tmp_path / "AgentEval" / "webarena"
    eval_dir.mkdir(parents=True)
    (eval_dir / "webarena_test.json").write_text("[]", encoding="utf-8")
    eval_results = list(preflight.check_dataset_paths("webarena", "eval", tmp_path))
    assert all(result.ok for result in eval_results)
