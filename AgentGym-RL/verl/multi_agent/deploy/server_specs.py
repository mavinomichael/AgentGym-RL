# Multi-agent extension.
# Derived from: /Users/mavinomichael/PycharmProjects/AgentGym-RL/AgentGym
# Original files left untouched for comparison.

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class ServerSpec:
    task_name: str
    package_name: str
    package_dir: str
    default_env_name: str
    python_version: str
    launch_command: str
    requires_playwright: bool = False
    requires_judge_credentials: bool = False
    extra_env_vars: List[str] | None = None


_SERVER_SPECS: Dict[str, ServerSpec] = {
    "babyai": ServerSpec(
        task_name="babyai",
        package_name="agentenv_babyai",
        package_dir="AgentGym/agentenv-babyai",
        default_env_name="agentenv-babyai",
        python_version="3.10",
        launch_command="babyai --host 0.0.0.0 --port 36005",
    ),
    "textcraft": ServerSpec(
        task_name="textcraft",
        package_name="agentenv_textcraft",
        package_dir="AgentGym/agentenv-textcraft",
        default_env_name="agentenv-textcraft",
        python_version="3.9",
        launch_command="textcraft --host 0.0.0.0 --port 36005",
    ),
    "searchqa": ServerSpec(
        task_name="searchqa",
        package_name="agentenv_searchqa",
        package_dir="AgentGym/agentenv-searchqa",
        default_env_name="agentenv-searchqa",
        python_version="3.10",
        launch_command="searchqa --host 0.0.0.0 --port 36005",
        extra_env_vars=[
            "SEARCHQA_FAISS_GPU",
            "SEARCHQA_INDEX_PATH",
            "SEARCHQA_CORPUS_PATH",
            "SEARCHQA_RETRIEVAL_MODEL_PATH",
        ],
    ),
    "sciworld": ServerSpec(
        task_name="sciworld",
        package_name="agentenv_sciworld",
        package_dir="AgentGym/agentenv-sciworld",
        default_env_name="agentenv-sciworld",
        python_version="3.8",
        launch_command="sciworld --host 0.0.0.0 --port 36005",
    ),
    "webarena": ServerSpec(
        task_name="webarena",
        package_name="agentenv_webarena",
        package_dir="AgentGym/agentenv-webarena",
        default_env_name="agentenv-webarena",
        python_version="3.10.13",
        launch_command="webarena --host 0.0.0.0 --port 36005",
        requires_playwright=True,
        requires_judge_credentials=True,
        extra_env_vars=[
            "SHOPPING",
            "SHOPPING_ADMIN",
            "REDDIT",
            "GITLAB",
            "MAP",
            "WIKIPEDIA",
            "HOMEPAGE",
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
        ],
    ),
}


def get_server_spec(task_name: str) -> ServerSpec:
    normalized = task_name.lower()
    if normalized not in _SERVER_SPECS:
        raise KeyError(f"Unsupported server spec: {task_name}")
    return _SERVER_SPECS[normalized]


def resolve_repo_path(repo_root: str, relative_path: str) -> Path:
    return Path(repo_root).joinpath(relative_path)
