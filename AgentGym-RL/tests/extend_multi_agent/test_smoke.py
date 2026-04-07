import subprocess
from pathlib import Path

import pytest

from conftest import load_extend_module


ROOT = Path(__file__).resolve().parents[3]


def test_extend_and_original_entrypoints_import_cleanly():
    pytest.importorskip("hydra")
    pytest.importorskip("ray")
    load_extend_module(
        "verl.extend_multi_agent.main_ppo",
        "verl/extend_multi_agent/main_ppo.py",
    )
    module = load_extend_module(
        "verl.multi_agent.main_ppo",
        "verl/multi_agent/main_ppo.py",
    )
    assert hasattr(module, "run_ppo")


def test_new_extend_scripts_parse_with_bash():
    scripts = [
        ROOT / "scripts/extend-multi-agent/run_babyai_sft_both.sh",
        ROOT / "scripts/extend-multi-agent/run_babyai_train_planner_phase.sh",
        ROOT / "scripts/extend-multi-agent/run_babyai_train_executor_phase.sh",
        ROOT / "scripts/extend-multi-agent/run_babyai_joint_phase.sh",
    ]
    for script in scripts:
        result = subprocess.run(["bash", "-n", str(script)], check=False, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
