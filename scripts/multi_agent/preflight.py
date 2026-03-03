#!/usr/bin/env python3
"""Multi-agent runtime preflight checks for training and evaluation."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib import error, request


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = REPO_ROOT / "AgentGym-RL"
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

from verl.multi_agent.deploy import get_server_spec
from verl.multi_agent.envs import get_task_profile


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = True


def load_env_file(env_file: Path) -> None:
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def check_python_version(expected: str) -> CheckResult:
    current = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    expected_prefix = ".".join(expected.split(".")[:2])
    current_prefix = f"{sys.version_info.major}.{sys.version_info.minor}"
    ok = current_prefix == expected_prefix
    detail = f"current={current} recommended_server_python={expected}"
    return CheckResult("python_version", ok, detail, required=False)


def check_torch_cuda(requested_gpus: int) -> Iterable[CheckResult]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - import failure path is still useful in preflight
        yield CheckResult("torch_import", False, f"torch import failed: {exc}")
        return

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<not-set>")
    device_count = torch.cuda.device_count()
    yield CheckResult("cuda_visible_devices", True, f"CUDA_VISIBLE_DEVICES={visible}", required=False)
    yield CheckResult(
        "gpu_count",
        device_count >= requested_gpus,
        f"requested={requested_gpus} available={device_count}",
    )


def check_path_exists(name: str, path_value: str) -> CheckResult:
    path = Path(path_value).expanduser()
    return CheckResult(name, path.exists(), str(path))


def check_dataset_paths(task_name: str, mode: str, data_root: Path) -> Iterable[CheckResult]:
    task_profile = get_task_profile(task_name)
    if mode == "train":
        yield CheckResult(
            "train_dataset",
            (data_root / task_profile.train_file).exists(),
            str((data_root / task_profile.train_file).resolve()),
        )
        return

    eval_dir = data_root / task_profile.eval_dir
    test_file = eval_dir / f"{task_name}_test.json"
    yield CheckResult("eval_dataset_dir", eval_dir.exists(), str(eval_dir.resolve()))
    yield CheckResult("eval_test_file", test_file.exists(), str(test_file.resolve()))


def check_import(package_name: str) -> CheckResult:
    try:
        importlib.import_module(package_name)
    except Exception as exc:
        return CheckResult("server_package_import", False, f"{package_name}: {exc}")
    return CheckResult("server_package_import", True, package_name)


def _candidate_urls(base_url: str) -> list[str]:
    normalized = base_url.rstrip("/")
    return [normalized, f"{normalized}/docs", f"{normalized}/openapi.json"]


def check_server_url(base_url: str) -> CheckResult:
    last_error = "no successful response"
    for url in _candidate_urls(base_url):
        try:
            with request.urlopen(url, timeout=5) as response:
                status = getattr(response, "status", None) or response.getcode()
                if status == 200:
                    return CheckResult("server_url", True, f"{url} -> 200")
                last_error = f"{url} -> {status}"
        except error.HTTPError as exc:
            last_error = f"{url} -> HTTP {exc.code}"
        except Exception as exc:  # pragma: no cover - depends on runtime network state
            last_error = f"{url} -> {exc}"
    return CheckResult("server_url", False, last_error)


def check_webarena_env() -> Iterable[CheckResult]:
    required_urls = ["SHOPPING", "SHOPPING_ADMIN", "REDDIT", "GITLAB", "MAP", "WIKIPEDIA", "HOMEPAGE"]
    for key in required_urls:
        value = os.environ.get(key, "")
        yield CheckResult(f"env:{key}", bool(value), value or "<unset>")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    yield CheckResult("env:OPENAI_API_KEY", bool(api_key), "set" if api_key else "<unset>")

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            executable_path = playwright.chromium.executable_path
        ok = bool(executable_path) and Path(executable_path).exists()
        yield CheckResult("playwright", ok, executable_path or "<missing executable>")
    except Exception as exc:  # pragma: no cover - depends on local playwright install
        yield CheckResult("playwright", False, str(exc))


def render_results(results: list[CheckResult]) -> int:
    exit_code = 0
    for result in results:
        level = "PASS" if result.ok else ("WARN" if not result.required else "FAIL")
        print(f"[{level}] {result.name}: {result.detail}")
        if not result.ok and result.required:
            exit_code = 1
    return exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate multi-agent train/eval prerequisites.")
    parser.add_argument("--task", required=True, choices=["babyai", "textcraft", "searchqa", "sciworld", "webarena"])
    parser.add_argument("--mode", required=True, choices=["train", "eval"])
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", ""))
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", str(TRAIN_ROOT)))
    parser.add_argument("--server-url", default=os.environ.get("ENV_SERVER_URL", "http://127.0.0.1:36005"))
    parser.add_argument("--webarena-env-file", default=os.environ.get("WEB_ARENA_ENV_FILE", str(REPO_ROOT / "AgentGym" / "agentenv-webarena" / ".env")))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = get_server_spec(args.task)
    requested_gpus = args.gpus if args.gpus is not None else int(os.environ.get("N_GPUS", "1"))

    if args.task == "webarena":
        load_env_file(Path(args.webarena_env_file).expanduser())

    results: list[CheckResult] = []
    results.append(check_python_version(spec.python_version))
    results.extend(check_torch_cuda(requested_gpus))

    if args.model_path:
        results.append(check_path_exists("model_path", args.model_path))
    else:
        results.append(CheckResult("model_path", False, "MODEL_PATH is unset"))

    data_root = Path(args.data_root).expanduser()
    results.extend(check_dataset_paths(args.task, args.mode, data_root))
    results.append(check_import(spec.package_name))
    results.append(check_server_url(args.server_url))

    if args.task == "webarena":
        results.extend(check_webarena_env())

    exit_code = render_results(results)
    if exit_code != 0:
        print("Preflight failed. Fix the FAIL items before running training or evaluation.")
    else:
        print("Preflight passed.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
