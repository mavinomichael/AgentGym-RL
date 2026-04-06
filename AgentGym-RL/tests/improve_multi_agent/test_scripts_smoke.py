import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_improve_multi_agent_shell_scripts_parse_with_bash_n():
    script_paths = sorted((REPO_ROOT / "scripts/improve_multi-agent").glob("*.sh"))
    assert script_paths
    for script_path in script_paths:
        subprocess.run(["bash", "-n", str(script_path)], check=True)


def test_improve_multi_agent_workspace_docs_exist():
    expected_paths = [
        REPO_ROOT / "improve_multi-agent/README.md",
        REPO_ROOT / "improve_multi-agent/implementation_status.md",
        REPO_ROOT / "improve_multi-agent/paper_to_code_mapping.md",
        REPO_ROOT / "improve_multi-agent/manifests/babyai_ablation_manifest.json",
    ]
    for path in expected_paths:
        assert path.exists()
