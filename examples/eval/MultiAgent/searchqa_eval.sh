#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../../../scripts/multi_agent/common.sh"

TASK_NAME="searchqa"
TRAIN_ROOT=$(multi_agent::train_root)

export DATA_ROOT="${DATA_ROOT:-$TRAIN_ROOT}"
export ENV_SERVER_URL="${ENV_SERVER_URL:-http://127.0.0.1:36005}"
export SAVE_ROOT="${SAVE_ROOT:-$(multi_agent::repo_root)/runs}"
export EXP_NAME="${EXP_NAME:-searchqa_planner_executor_eval}"
export N_GPUS="${N_GPUS:-1}"
export TP_SIZE="${TP_SIZE:-1}"
export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

multi_agent::maybe_merge_checkpoint "$TRAIN_ROOT"
multi_agent::require_var MODEL_PATH
multi_agent::print_run_header "eval" "$TASK_NAME"

cd "$TRAIN_ROOT"
HYDRA_FULL_ERROR=1 python3 -m verl.multi_agent.main_generation   task=$TASK_NAME   runtime=qwen2_5_7b_1gpu_smoke   "$@"
