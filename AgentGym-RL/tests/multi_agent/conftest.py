import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _ensure_package(name: str) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module


def load_multi_agent_module(module_name: str, relative_path: str):
    parts = module_name.split(".")
    for idx in range(1, len(parts)):
        _ensure_package(".".join(parts[:idx]))

    file_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
