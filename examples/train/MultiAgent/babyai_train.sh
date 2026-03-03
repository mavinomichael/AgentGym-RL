#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../../../scripts/multi_agent/common.sh"

TASK_NAME="babyai"
TRAIN_ROOT=$(multi_agent::train_root)

export MODEL_PATH="${MODEL_PATH:-}"
export DATA_ROOT="${DATA_ROOT:-$TRAIN_ROOT}"
export ENV_SERVER_URL="${ENV_SERVER_URL:-http://127.0.0.1:36005}"
export SAVE_ROOT="${SAVE_ROOT:-$(multi_agent::repo_root)/runs}"
export EXP_NAME="${EXP_NAME:-babyai_planner_executor}"
export N_GPUS="${N_GPUS:-8}"
export TP_SIZE="${TP_SIZE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

multi_agent::require_var MODEL_PATH
multi_agent::print_run_header "train" "$TASK_NAME"

cd "$TRAIN_ROOT"
HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" python3 -m verl.multi_agent.main_ppo   task=$TASK_NAME   runtime=qwen2_5_7b_8gpu   algo=multi_agent_gae   "$@"
