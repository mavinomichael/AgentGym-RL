#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


STEP_RE = re.compile(r"step:(\d+)")
FAILURE_PATTERNS = {
    "traceback": re.compile(r"Traceback \(most recent call last\):"),
    "actor_died": re.compile(r"ActorDiedError"),
    "cuda_oom": re.compile(r"(CUDA out of memory|OutOfMemoryError|torch\.OutOfMemoryError)"),
    "nccl_timeout": re.compile(r"(Watchdog caught collective operation timeout|ProcessGroupNCCL)"),
    "ray_worker_died": re.compile(r"worker died or was killed while executing a task"),
}


def log_status(message: str) -> None:
    print(f"[stage3-smoke-monitor] {message}", file=sys.stderr, flush=True)


def parse_deadlines(values: Iterable[str]) -> list[tuple[int, int]]:
    deadlines: list[tuple[int, int]] = []
    for raw in values:
        try:
            step_raw, seconds_raw = raw.split(":", 1)
            deadlines.append((int(step_raw), int(seconds_raw)))
        except ValueError as exc:  # pragma: no cover - CLI validation
            raise argparse.ArgumentTypeError(f"Invalid deadline {raw!r}; expected STEP:SECONDS") from exc
    return sorted(deadlines, key=lambda item: item[0])


def parse_latest_step(text: str) -> int | None:
    steps = [int(match.group(1)) for match in STEP_RE.finditer(text)]
    return max(steps) if steps else None


def detect_failure_reason(text: str) -> str | None:
    for reason, pattern in FAILURE_PATTERNS.items():
        if pattern.search(text):
            return reason
    return None


def tail_lines(path: Path, limit: int = 100) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()[-limit:]


def gpu_snapshot() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:  # pragma: no cover - best effort diagnostics
        return []

    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "gpu_index": parts[0],
                "utilization_gpu": parts[1],
                "utilization_memory": parts[2],
                "memory_used_mb": parts[3],
                "memory_total_mb": parts[4],
            }
        )
    return rows


def checkpoint_status(checkpoint_root: Path, expected_steps: list[int]) -> dict[str, bool]:
    return {
        f"global_step_{step}": (checkpoint_root / f"global_step_{step}").is_dir()
        for step in expected_steps
    }


def write_summary(
    *,
    summary_path: Path,
    status: str,
    failure_reason: str | None,
    latest_step: int | None,
    last_seen_stage: str,
    log_path: Path,
    checkpoint_root: Path,
    expected_checkpoints: list[int],
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "failure_reason": failure_reason,
        "latest_step": latest_step,
        "last_seen_stage": last_seen_stage,
        "log_path": str(log_path),
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_status": checkpoint_status(checkpoint_root, expected_checkpoints),
        "gpu_snapshot": gpu_snapshot(),
        "log_tail": tail_lines(log_path, limit=100),
        "written_at_unix": int(time.time()),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def kill_process_group(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=15)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait(timeout=15)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-fast monitor for the improve_multi_agent Stage 3 smoke run.")
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--success-step", type=int, default=20)
    parser.add_argument("--completion-grace-seconds", type=int, default=300)
    parser.add_argument("--step-deadline", action="append", default=[])
    parser.add_argument("--checkpoint-step", action="append", type=int, default=[10, 20])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("command is required after --")

    deadlines = parse_deadlines(args.step_deadline)
    log_path = Path(args.log_path)
    summary_path = Path(args.summary_path)
    checkpoint_root = Path(args.checkpoint_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    start_time = time.monotonic()
    success_seen_at: float | None = None
    latest_step: int | None = None
    termination_reason: str | None = None

    def handle_signal(signum: int, _frame: object) -> None:
        nonlocal termination_reason
        termination_reason = f"monitor_signal_{signal.Signals(signum).name.lower()}"
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    log_status(
        "starting command="
        + " ".join(args.command)
        + f" log_path={log_path} summary_path={summary_path} checkpoint_root={checkpoint_root}"
    )

    with log_path.open("wb") as handle:
        proc = subprocess.Popen(
            args.command,
            cwd=args.cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        try:
            while True:
                time.sleep(args.poll_seconds)
                text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
                latest_step = parse_latest_step(text)
                failure_reason = detect_failure_reason(text)
                if failure_reason:
                    log_status(f"failure detected reason={failure_reason} latest_step={latest_step}")
                    kill_process_group(proc)
                    write_summary(
                        summary_path=summary_path,
                        status="failed",
                        failure_reason=failure_reason,
                        latest_step=latest_step,
                        last_seen_stage="stage3_smoke",
                        log_path=log_path,
                        checkpoint_root=checkpoint_root,
                        expected_checkpoints=args.checkpoint_step,
                    )
                    return 1

                elapsed = int(time.monotonic() - start_time)
                for step, deadline_seconds in deadlines:
                    if latest_step is not None and latest_step >= step:
                        continue
                    if elapsed > deadline_seconds:
                        log_status(
                            f"deadline missed target_step={step} elapsed={elapsed}s latest_step={latest_step}"
                        )
                        kill_process_group(proc)
                        write_summary(
                            summary_path=summary_path,
                            status="failed",
                            failure_reason=f"missed_step_deadline_{step}",
                            latest_step=latest_step,
                            last_seen_stage="stage3_smoke",
                            log_path=log_path,
                            checkpoint_root=checkpoint_root,
                            expected_checkpoints=args.checkpoint_step,
                        )
                        return 1

                if latest_step is not None and latest_step >= args.success_step:
                    success_seen_at = success_seen_at or time.monotonic()
                    log_status(f"success step reached latest_step={latest_step}; waiting for checkpoint grace")
                    if proc.poll() is None and time.monotonic() - success_seen_at < args.completion_grace_seconds:
                        continue
                    checkpoint_ok = any((checkpoint_root / f"global_step_{step}").is_dir() for step in args.checkpoint_step)
                    status = "passed" if checkpoint_ok else "failed"
                    reason = None if checkpoint_ok else "missing_expected_checkpoint"
                    log_status(
                        f"completion status={status} latest_step={latest_step} checkpoint_ok={checkpoint_ok}"
                    )
                    if proc.poll() is None:
                        kill_process_group(proc)
                    write_summary(
                        summary_path=summary_path,
                        status=status,
                        failure_reason=reason,
                        latest_step=latest_step,
                        last_seen_stage="stage3_smoke",
                        log_path=log_path,
                        checkpoint_root=checkpoint_root,
                        expected_checkpoints=args.checkpoint_step,
                    )
                    return 0 if status == "passed" else 1

                if proc.poll() is not None:
                    checkpoint_ok = latest_step is not None and latest_step >= args.success_step and any(
                        (checkpoint_root / f"global_step_{step}").is_dir() for step in args.checkpoint_step
                    )
                    log_status(
                        f"child exited returncode={proc.returncode} latest_step={latest_step} checkpoint_ok={checkpoint_ok}"
                    )
                    write_summary(
                        summary_path=summary_path,
                        status="passed" if checkpoint_ok else "failed",
                        failure_reason=None if checkpoint_ok else f"process_exit_{proc.returncode}",
                        latest_step=latest_step,
                        last_seen_stage="stage3_smoke",
                        log_path=log_path,
                        checkpoint_root=checkpoint_root,
                        expected_checkpoints=args.checkpoint_step,
                    )
                    return 0 if checkpoint_ok else 1
        except BaseException:
            log_status(
                f"monitor exiting unexpectedly reason={termination_reason or 'monitor_exception'} latest_step={latest_step}"
            )
            write_summary(
                summary_path=summary_path,
                status="failed",
                failure_reason=termination_reason or "monitor_exception",
                latest_step=latest_step,
                last_seen_stage="stage3_smoke",
                log_path=log_path,
                checkpoint_root=checkpoint_root,
                expected_checkpoints=args.checkpoint_step,
            )
            raise
        finally:
            if proc.poll() is None:
                log_status("stopping child process group during monitor shutdown")
                kill_process_group(proc)


if __name__ == "__main__":
    sys.exit(main())
