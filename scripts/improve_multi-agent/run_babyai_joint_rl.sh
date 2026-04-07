#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PKG_ROOT="$REPO_ROOT/AgentGym-RL"

export MODEL_PATH="${MODEL_PATH:-/home/mavinomichael/AgentGym-RL/models/Qwen2.5-7B-Instruct}"
export DATA_ROOT="${DATA_ROOT:-$REPO_ROOT}"
export ENV_SERVER_URL="${ENV_SERVER_URL:-http://127.0.0.1:36006}"
export SAVE_ROOT="${SAVE_ROOT:-/mnt/data/saves}"
export EXP_NAME="${EXP_NAME:-improve_babyai_protocol_v1}"
export N_GPUS="${N_GPUS:-8}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-601}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
export ROLLOUT_MAX_TOKENS="${ROLLOUT_MAX_TOKENS:-192}"
export PLANNER_MAX_TOKENS="${PLANNER_MAX_TOKENS:-96}"
export EXECUTOR_MAX_TOKENS="${EXECUTOR_MAX_TOKENS:-96}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
export ACTOR_MAX_TOKEN_LEN_PER_GPU="${ACTOR_MAX_TOKEN_LEN_PER_GPU:-8192}"
export CRITIC_MAX_TOKEN_LEN_PER_GPU="${CRITIC_MAX_TOKEN_LEN_PER_GPU:-8192}"
export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-false}"
export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-false}"
export ACTOR_GRAD_OFFLOAD="${ACTOR_GRAD_OFFLOAD:-false}"
export ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-false}"
export REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-false}"
export CRITIC_PARAM_OFFLOAD="${CRITIC_PARAM_OFFLOAD:-false}"
export CRITIC_GRAD_OFFLOAD="${CRITIC_GRAD_OFFLOAD:-false}"
export CRITIC_OPTIMIZER_OFFLOAD="${CRITIC_OPTIMIZER_OFFLOAD:-false}"
export ROLE_LOCAL_OPTIMIZATION="${ROLE_LOCAL_OPTIMIZATION:-true}"
export ROLE_LOCAL_ADVANTAGE="${ROLE_LOCAL_ADVANTAGE:-true}"
export ROLE_AUX_REWARDS="${ROLE_AUX_REWARDS:-true}"
export MILESTONE_REWARDS="${MILESTONE_REWARDS:-true}"
export PLANNER_PPO_WEIGHT="${PLANNER_PPO_WEIGHT:-1.0}"
export PLANNER_KL_WEIGHT="${PLANNER_KL_WEIGHT:-1.5}"
export EXECUTOR_PPO_WEIGHT="${EXECUTOR_PPO_WEIGHT:-1.0}"
export EXECUTOR_KL_WEIGHT="${EXECUTOR_KL_WEIGHT:-1.0}"
export ENV_TIMEOUT="${ENV_TIMEOUT:-45}"
export DEBUG_PROGRESS="${DEBUG_PROGRESS:-0}"
export DEBUG_ALL_RANKS="${DEBUG_ALL_RANKS:-0}"
export SKIP_VLLM_OFFLOAD="${SKIP_VLLM_OFFLOAD:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ENFORCE_ROLLOUT_HARNESS="${ENFORCE_ROLLOUT_HARNESS:-true}"
export HARNESS_SUMMARY_PATH="${HARNESS_SUMMARY_PATH:-$SAVE_ROOT/improve_multi_agent/rollout_harness/learned_planner_frozen_executor/summary.json}"
export VERL_IMPROVE_DEBUG_PROGRESS="${VERL_IMPROVE_DEBUG_PROGRESS:-$DEBUG_PROGRESS}"
export VERL_IMPROVE_DEBUG_ALL_RANKS="${VERL_IMPROVE_DEBUG_ALL_RANKS:-$DEBUG_ALL_RANKS}"
export VERL_AGENTGYM_TIMEOUT="${VERL_AGENTGYM_TIMEOUT:-$ENV_TIMEOUT}"
export VERL_IMPROVE_SKIP_VLLM_OFFLOAD="${VERL_IMPROVE_SKIP_VLLM_OFFLOAD:-$SKIP_VLLM_OFFLOAD}"
if [[ "${DEBUG_ALL_RANKS}" == "1" ]]; then
  export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
fi

if [[ "$ENFORCE_ROLLOUT_HARNESS" == "true" ]]; then
  python3 - "$REPO_ROOT" "$HARNESS_SUMMARY_PATH" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
for candidate in (repo_root / "AgentGym-RL", repo_root):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from verl.improve_multi_agent.rollout_harness import ensure_harness_passed

ensure_harness_passed(sys.argv[2])
PY
fi

if [[ "${MODEL_PATH}" == "/home/mavinomichael/AgentGym-RL/models/Qwen2.5-7B-Instruct" ]]; then
  STAGE3_ROOT="$SAVE_ROOT/improve_multi_agent/improve_babyai_planner_frozen_rl_v1"
  if [[ -d "$STAGE3_ROOT" ]]; then
    LATEST_STAGE3_CKPT=$(find "$STAGE3_ROOT" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1 || true)
    if [[ -n "${LATEST_STAGE3_CKPT:-}" ]]; then
      export MODEL_PATH="$LATEST_STAGE3_CKPT"
    fi
  fi
fi

cd "$PKG_ROOT"
python3 -m verl.improve_multi_agent.main_ppo \
  task=babyai \
  runtime=qwen2_5_7b_8gpu \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.experiment_name="$EXP_NAME" \
  trainer.default_local_dir="$SAVE_ROOT/improve_multi_agent/$EXP_NAME" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  task.train_batch_size="$TRAIN_BATCH_SIZE" \
  task.rollout_n="$ROLLOUT_N" \
  task.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  task.rollout_max_tokens="$ROLLOUT_MAX_TOKENS" \
  task.planner_max_tokens="$PLANNER_MAX_TOKENS" \
  task.executor_max_tokens="$EXECUTOR_MAX_TOKENS" \
  runtime.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  runtime.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
  runtime.max_num_seqs="$MAX_NUM_SEQS" \
  runtime.actor_ppo_max_token_len_per_gpu="$ACTOR_MAX_TOKEN_LEN_PER_GPU" \
  runtime.critic_ppo_max_token_len_per_gpu="$CRITIC_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.model.use_remove_padding="$USE_REMOVE_PADDING" \
  critic.model.use_remove_padding="$USE_REMOVE_PADDING" \
  actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_PARAM_OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.grad_offload="$ACTOR_GRAD_OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="$ACTOR_OPTIMIZER_OFFLOAD" \
  actor_rollout_ref.ref.fsdp_config.param_offload="$REF_PARAM_OFFLOAD" \
  critic.model.fsdp_config.param_offload="$CRITIC_PARAM_OFFLOAD" \
  critic.model.fsdp_config.grad_offload="$CRITIC_GRAD_OFFLOAD" \
  critic.model.fsdp_config.optimizer_offload="$CRITIC_OPTIMIZER_OFFLOAD" \
  improve_multi_agent.features.role_local_optimization="$ROLE_LOCAL_OPTIMIZATION" \
  improve_multi_agent.features.role_local_advantage="$ROLE_LOCAL_ADVANTAGE" \
  improve_multi_agent.features.role_aux_rewards="$ROLE_AUX_REWARDS" \
  improve_multi_agent.features.milestone_rewards="$MILESTONE_REWARDS" \
  actor_rollout_ref.agentgym.timeout="$ENV_TIMEOUT" \
  improve_multi_agent.roles.planner.ppo_weight="$PLANNER_PPO_WEIGHT" \
  improve_multi_agent.roles.planner.kl_weight="$PLANNER_KL_WEIGHT" \
  improve_multi_agent.roles.executor.ppo_weight="$EXECUTOR_PPO_WEIGHT" \
  improve_multi_agent.roles.executor.kl_weight="$EXECUTOR_KL_WEIGHT"
