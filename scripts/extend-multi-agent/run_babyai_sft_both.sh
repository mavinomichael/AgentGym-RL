#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
TRAIN_ROOT="${REPO_ROOT}/AgentGym-RL"
DATASET_DIR="${REPO_ROOT}/extend-multi-agent/datasets"
MODEL_PATH="${MODEL_PATH:-}"
SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/output/extend-multi-agent-sft}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

if [ -z "${MODEL_PATH}" ]; then
  echo "Missing MODEL_PATH" >&2
  exit 1
fi

python3 "${SCRIPT_DIR}/build_babyai_sft_datasets.py" --output-dir "${DATASET_DIR}"

for ROLE in planner executor; do
  DATA_FILE="${DATASET_DIR}/babyai_${ROLE}_sft.json"
  EXP_NAME="extend_babyai_${ROLE}_sft"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/run_fsdp_sft_no_deepspeed.py" \
    data.train_files="${DATA_FILE}" \
    data.train_batch_size=8 \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=2048 \
    model.partial_pretrain="${MODEL_PATH}" \
    trainer.default_local_dir="${SAVE_ROOT}/${ROLE}" \
    trainer.project_name=extend_multi_agent_sft \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.total_epochs=1
 done
