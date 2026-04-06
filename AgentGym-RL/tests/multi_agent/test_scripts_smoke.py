import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_all_new_shell_scripts_parse_with_bash_n():
    script_paths = [
        REPO_ROOT / "scripts/multi_agent/bootstrap_training_env.sh",
        REPO_ROOT / "scripts/multi_agent/common.sh",
        REPO_ROOT / "scripts/multi_agent/bootstrap_webarena_websites.sh",
        REPO_ROOT / "scripts/multi_agent/setup_babyai_server.sh",
        REPO_ROOT / "scripts/multi_agent/setup_textcraft_server.sh",
        REPO_ROOT / "scripts/multi_agent/setup_searchqa_server.sh",
        REPO_ROOT / "scripts/multi_agent/setup_sciworld_server.sh",
        REPO_ROOT / "scripts/multi_agent/setup_webarena_server.sh",
        REPO_ROOT / "scripts/multi_agent/launch_babyai_server.sh",
        REPO_ROOT / "scripts/multi_agent/launch_textcraft_server.sh",
        REPO_ROOT / "scripts/multi_agent/launch_searchqa_server.sh",
        REPO_ROOT / "scripts/multi_agent/launch_sciworld_server.sh",
        REPO_ROOT / "scripts/multi_agent/launch_webarena_server.sh",
        REPO_ROOT / "scripts/multi_agent/run_babyai_3agent_executor_reviewer_scaling_600_8gpu.sh",
        REPO_ROOT / "scripts/multi_agent/run_webarena_2agent_8gpu.sh",
        REPO_ROOT / "scripts/multi_agent/run_webarena_2agent_fixed15_8gpu_tagged.sh",
        REPO_ROOT / "scripts/multi_agent/run_webarena_2agent_fixed15_8gpu_plain_split_notag.sh",
        REPO_ROOT / "scripts/multi_agent/run_webarena_2agent_scaling_600_8gpu_plain_split_notag.sh",
        REPO_ROOT / "scripts/multi_agent/run_webarena_3agent_executor_reviewer_scaling_600_8gpu.sh",
    ]
    script_paths.extend(sorted((REPO_ROOT / "examples/train/MultiAgent").glob("*.sh")))
    script_paths.extend(sorted((REPO_ROOT / "examples/eval/MultiAgent").glob("*.sh")))

    for script_path in script_paths:
        subprocess.run(["bash", "-n", str(script_path)], check=True)


def test_webarena_env_example_exists():
    env_example = REPO_ROOT / "AgentGym/agentenv-webarena/.env.example"
    assert env_example.exists()
    content = env_example.read_text(encoding="utf-8")
    assert "SHOPPING=" in content
    assert "OPENAI_API_KEY=" in content
