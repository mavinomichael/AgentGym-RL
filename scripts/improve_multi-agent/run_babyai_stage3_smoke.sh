#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

export SAVE_ROOT="${SAVE_ROOT:-/mnt/data/saves}"
export DATA_ROOT="${DATA_ROOT:-$REPO_ROOT}"
export ENV_SERVER_URL="${ENV_SERVER_URL:-http://127.0.0.1:36006}"
export EXP_NAME="${EXP_NAME:-improve_babyai_stage3_smoke_v1}"
export MODEL_PATH="${MODEL_PATH:-$SAVE_ROOT/improve_multi_agent/improve_babyai_planner_warmstart_v1/global_step_200}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-21}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export ROLLOUT_N="${ROLLOUT_N:-1}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
export ROLLOUT_MAX_TOKENS="${ROLLOUT_MAX_TOKENS:-128}"
export PLANNER_MAX_TOKENS="${PLANNER_MAX_TOKENS:-64}"
export EXECUTOR_MAX_TOKENS="${EXECUTOR_MAX_TOKENS:-64}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.40}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export ACTOR_MAX_TOKEN_LEN_PER_GPU="${ACTOR_MAX_TOKEN_LEN_PER_GPU:-4096}"
export CRITIC_MAX_TOKEN_LEN_PER_GPU="${CRITIC_MAX_TOKEN_LEN_PER_GPU:-4096}"
export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-true}"
export ACTOR_GRAD_OFFLOAD="${ACTOR_GRAD_OFFLOAD:-true}"
export ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-true}"
export REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-true}"
export CRITIC_PARAM_OFFLOAD="${CRITIC_PARAM_OFFLOAD:-true}"
export CRITIC_GRAD_OFFLOAD="${CRITIC_GRAD_OFFLOAD:-true}"
export CRITIC_OPTIMIZER_OFFLOAD="${CRITIC_OPTIMIZER_OFFLOAD:-true}"
export ROLE_LOCAL_OPTIMIZATION="${ROLE_LOCAL_OPTIMIZATION:-true}"
export ROLE_LOCAL_ADVANTAGE="${ROLE_LOCAL_ADVANTAGE:-true}"
export ROLE_AUX_REWARDS="${ROLE_AUX_REWARDS:-true}"
export MILESTONE_REWARDS="${MILESTONE_REWARDS:-false}"
export PLANNER_PPO_WEIGHT="${PLANNER_PPO_WEIGHT:-1.0}"
export PLANNER_KL_WEIGHT="${PLANNER_KL_WEIGHT:-1.5}"
export EXECUTOR_PPO_WEIGHT="${EXECUTOR_PPO_WEIGHT:-0.0}"
export EXECUTOR_KL_WEIGHT="${EXECUTOR_KL_WEIGHT:-0.0}"
export N_GPUS="${N_GPUS:-8}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Planner warm-start checkpoint not found at $MODEL_PATH" >&2
  exit 1
fi

LOG_ROOT="${LOG_ROOT:-$SAVE_ROOT/improve_multi_agent/smoke_logs}"
SUMMARY_ROOT="${SUMMARY_ROOT:-$SAVE_ROOT/improve_multi_agent/smoke_reports}"
LOG_PATH="$LOG_ROOT/${EXP_NAME}.log"
SUMMARY_PATH="$SUMMARY_ROOT/${EXP_NAME}.json"
CHECKPOINT_ROOT="$SAVE_ROOT/improve_multi_agent/$EXP_NAME"

mkdir -p "$LOG_ROOT" "$SUMMARY_ROOT" "$CHECKPOINT_ROOT"

python3 "$SCRIPT_DIR/monitor_stage3_smoke.py" \
  --log-path "$LOG_PATH" \
  --summary-path "$SUMMARY_PATH" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --cwd "$REPO_ROOT/AgentGym-RL" \
  --step-deadline "1:900" \
  --step-deadline "5:1800" \
  --step-deadline "20:3600" \
  --checkpoint-step 10 \
  --checkpoint-step 20 \
  -- bash "$SCRIPT_DIR/run_babyai_planner_frozen_rl.sh"
