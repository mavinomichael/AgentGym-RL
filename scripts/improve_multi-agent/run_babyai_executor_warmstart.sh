#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PKG_ROOT="$REPO_ROOT/AgentGym-RL"
DEFAULT_DATASET="$REPO_ROOT/improve_multi-agent/datasets/babyai_executor_warmstart_train.json"
if [[ ! -f "$DEFAULT_DATASET" ]]; then
  DEFAULT_DATASET="$REPO_ROOT/improve_multi-agent/datasets/babyai_executor_warmstart.json"
fi
DATASET="${DATASET:-$DEFAULT_DATASET}"
MODEL_PATH="${MODEL_PATH:-/home/mavinomichael/AgentGym-RL/models/Qwen2.5-7B-Instruct}"
SAVE_ROOT="${SAVE_ROOT:-/mnt/data/saves}"
EXP_NAME="${EXP_NAME:-improve_babyai_executor_warmstart_v1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

if [[ ! -f "$DATASET" ]]; then
  cd "$REPO_ROOT"
  python3 "$REPO_ROOT/scripts/improve_multi-agent/build_babyai_trace_datasets.py"
fi
cd "$PKG_ROOT"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" "$REPO_ROOT/scripts/improve_multi-agent/run_fsdp_sft_no_deepspeed.py" \
  data.train_files="$DATASET" \
  data.train_batch_size=8 \
  data.micro_batch_size_per_gpu=1 \
  data.max_length=2048 \
  model.partial_pretrain="$MODEL_PATH" \
  trainer.default_local_dir="$SAVE_ROOT/improve_multi_agent/$EXP_NAME" \
  trainer.experiment_name="$EXP_NAME" \
  trainer.project_name=improve_multi_agent \
  trainer.total_epochs=1 \
  trainer.total_training_steps=200
