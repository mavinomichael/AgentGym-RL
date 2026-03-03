import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent

for candidate in (ROOT, REPO_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


def _ensure_package(name: str) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    package_path = ROOT.joinpath(*name.split("."))
    if package_path.is_dir():
        module.__path__ = [str(package_path)]
    else:
        module.__path__ = [str(ROOT)]
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
