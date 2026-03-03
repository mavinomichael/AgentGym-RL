# Multi-agent extension.
# Derived from: /Users/mavinomichael/PycharmProjects/AgentGym-RL/examples/train/AgentGym-RL
# Original files left untouched for comparison.

from dataclasses import dataclass, asdict
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


_TASK_PROFILES: Dict[str, TaskProfile] = {
    "babyai": TaskProfile(
        task_name="babyai",
        train_file="AgentItemId/babyai_train.json",
        eval_dir="AgentEval/babyai",
        max_prompt_length=1024,
        max_response_length=4096,
        default_max_rounds=20,
        rollout_max_tokens=200,
        planner_max_tokens=128,
        executor_max_tokens=200,
        train_batch_size=16,
        rollout_n=8,
        ppo_mini_batch_size=8,
        ppo_micro_batch_size_per_gpu=1,
        critic_ppo_micro_batch_size_per_gpu=1,
        default_timeout=600,
        invalid_action_patterns=[
            "Error: Only one 'Action' is allowed per response.",
        ],
        executor_native_format_hint="Use the BabyAI native format with a Thought section followed by a single Action line.",
        eval_max_prompt_length=512,
        eval_max_response_length=8192,
        eval_batch_size=32,
        eval_max_rounds=20,
        total_epochs=10,
    ),
    "textcraft": TaskProfile(
        task_name="textcraft",
        train_file="AgentItemId/textcraft_train.json",
        eval_dir="AgentEval/textcraft",
        max_prompt_length=512,
        max_response_length=10240,
        default_max_rounds=30,
        rollout_max_tokens=512,
        planner_max_tokens=192,
        executor_max_tokens=512,
        train_batch_size=32,
        rollout_n=8,
        ppo_mini_batch_size=8,
        ppo_micro_batch_size_per_gpu=1,
        critic_ppo_micro_batch_size_per_gpu=1,
        default_timeout=600,
        invalid_action_patterns=[
            "Error: Only one 'Action' is allowed per response.",
        ],
        executor_native_format_hint="Use the TextCraft native format with a Thought section followed by a single Action line.",
        eval_max_prompt_length=750,
        eval_max_response_length=14098,
        eval_batch_size=32,
        eval_max_rounds=30,
        total_epochs=30,
    ),
    "searchqa": TaskProfile(
        task_name="searchqa",
        train_file="AgentItemId/searchqa_train.json",
        eval_dir="AgentEval/searchqa",
        max_prompt_length=1024,
        max_response_length=8192,
        default_max_rounds=5,
        rollout_max_tokens=512,
        planner_max_tokens=192,
        executor_max_tokens=512,
        train_batch_size=32,
        rollout_n=4,
        ppo_mini_batch_size=8,
        ppo_micro_batch_size_per_gpu=1,
        critic_ppo_micro_batch_size_per_gpu=1,
        default_timeout=600,
        invalid_action_patterns=[
            "Your previous action is invalid.",
        ],
        executor_native_format_hint="Begin directly with SearchQA tags like <think>, <search>, and <answer>. Do not prepend any role labels.",
        eval_max_prompt_length=750,
        eval_max_response_length=14098,
        eval_batch_size=32,
        eval_max_rounds=30,
        total_epochs=20,
    ),
    "sciworld": TaskProfile(
        task_name="sciworld",
        train_file="AgentItemId/sciworld_train.json",
        eval_dir="AgentEval/sciworld",
        max_prompt_length=1024,
        max_response_length=4096,
        default_max_rounds=20,
        rollout_max_tokens=200,
        planner_max_tokens=128,
        executor_max_tokens=200,
        train_batch_size=16,
        rollout_n=8,
        ppo_mini_batch_size=8,
        ppo_micro_batch_size_per_gpu=1,
        critic_ppo_micro_batch_size_per_gpu=1,
        default_timeout=600,
        invalid_action_patterns=[
            "Invalid Action.",
        ],
        executor_native_format_hint="Use the SciWorld format expected by the environment. Preserve the original native action surface.",
        eval_max_prompt_length=1024,
        eval_max_response_length=8192,
        eval_batch_size=32,
        eval_max_rounds=30,
        total_epochs=10,
    ),
    "webarena": TaskProfile(
        task_name="webarena",
        train_file="AgentItemId/webarena_train.json",
        eval_dir="AgentEval/webarena",
        max_prompt_length=750,
        max_response_length=14098,
        default_max_rounds=15,
        rollout_max_tokens=512,
        planner_max_tokens=192,
        executor_max_tokens=512,
        train_batch_size=32,
        rollout_n=4,
        ppo_mini_batch_size=4,
        ppo_micro_batch_size_per_gpu=1,
        critic_ppo_micro_batch_size_per_gpu=1,
        default_timeout=600,
        invalid_action_patterns=[
            "Cannot parse action from response.",
            "TimeoutError",
            "Error in step:",
        ],
        executor_native_format_hint="Use the WebArena native format with reasoning and exactly one triple-backticked action block.",
        eval_max_prompt_length=750,
        eval_max_response_length=14098,
        eval_batch_size=32,
        eval_max_rounds=15,
        total_epochs=25,
    ),
}


def get_task_profile(task_name: str) -> TaskProfile:
    normalized = task_name.lower()
    if normalized not in _TASK_PROFILES:
        raise KeyError(f"Unsupported multi-agent task profile: {task_name}")
    return _TASK_PROFILES[normalized]


def list_task_profiles() -> List[str]:
    return sorted(_TASK_PROFILES.keys())
