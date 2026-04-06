#!/usr/bin/env python3
"""Run the FSDP SFT trainer while forcing accelerate to ignore DeepSpeed.

The remote training environment has `deepspeed` installed but no CUDA toolkit
path configured for building optional ops. During checkpoint save,
`transformers` -> `accelerate` tries to import DeepSpeed only to unwrap the
model class, and that import crashes on missing `CUDA_HOME`.

For the improve_multi_agent warm-start stages we are not using DeepSpeed at
all, so we patch the availability checks before importing the trainer.
"""

from __future__ import annotations

import sys
import runpy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT / "AgentGym-RL"
for candidate in (PKG_ROOT, REPO_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


def _disable_deepspeed() -> None:
    import accelerate.utils.imports as acc_imports
    import accelerate.utils.other as acc_other

    acc_imports.is_deepspeed_available = lambda: False
    acc_other.is_deepspeed_available = lambda: False


def main() -> None:
    _disable_deepspeed()
    runpy.run_module("verl.agent_trainer.fsdp_sft_trainer", run_name="__main__")


if __name__ == "__main__":
    main()
