#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT / "AgentGym-RL"
for candidate in (PKG_ROOT, REPO_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from verl.extend_multi_agent.sft.bootstrap import DEFAULT_TRACE_ROOTS, write_sft_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BabyAI SFT datasets for extend_multi_agent.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trace-root", action="append", dest="trace_roots")
    args = parser.parse_args()
    written = write_sft_datasets(args.output_dir, trace_roots=args.trace_roots or DEFAULT_TRACE_ROOTS)
    print(json.dumps(written, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
