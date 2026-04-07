from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass(frozen=True)
class TaskProfile:
    task_name: str
    train_file: str
    eval_dir: str
    max_prompt_length: int
    max_response_length: int
    default_max_rounds: int
    rollout_max_tokens: int
    planner_max_tokens: int
    executor_max_tokens: int
    train_batch_size: int
    rollout_n: int
    ppo_mini_batch_size: int
    ppo_micro_batch_size_per_gpu: int
    critic_ppo_micro_batch_size_per_gpu: int
    default_timeout: int
    invalid_action_patterns: List[str]
    executor_native_format_hint: str
    eval_max_prompt_length: int
    eval_max_response_length: int
    eval_batch_size: int
    eval_max_rounds: int
    total_epochs: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


_BABYAI_PROFILE = TaskProfile(
    task_name="babyai",
    train_file="AgentItemId/babyai_train.json",
    eval_dir="AgentEval/babyai",
    max_prompt_length=1024,
    max_response_length=4096,
    default_max_rounds=20,
    rollout_max_tokens=128,
    planner_max_tokens=96,
    executor_max_tokens=128,
    train_batch_size=16,
    rollout_n=4,
    ppo_mini_batch_size=8,
    ppo_micro_batch_size_per_gpu=1,
    critic_ppo_micro_batch_size_per_gpu=1,
    default_timeout=300,
    invalid_action_patterns=[
        "Error: Only one 'Action' is allowed per response.",
        "Your previous action is invalid.",
    ],
    executor_native_format_hint="Return strict executor JSON and let the adapter map it to one BabyAI action.",
    eval_max_prompt_length=512,
    eval_max_response_length=4096,
    eval_batch_size=16,
    eval_max_rounds=20,
    total_epochs=10,
)


def get_task_profile(task_name: str) -> TaskProfile:
    normalized = str(task_name).lower().strip()
    if normalized != "babyai":
        raise ValueError(
            f"extend_multi_agent currently supports only BabyAI, received task '{task_name}'. "
            "Use verl.multi_agent for other tasks."
        )
    return _BABYAI_PROFILE


def list_task_profiles() -> List[str]:
    return ["babyai"]
