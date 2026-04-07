#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
TRAIN_ROOT="${REPO_ROOT}/AgentGym-RL"
RUN_DIR="${RUN_DIR:-}"
ENV_SERVER_URL="${ENV_SERVER_URL:-http://127.0.0.1:36006}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-agentgym-rl}"
EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-${REPO_ROOT}/output/extend-multi-agent-eval}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "Missing RUN_DIR" >&2
  exit 1
fi

mkdir -p "${EVAL_LOG_ROOT}"

if [[ -f /home/mavinomichael/miniconda/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /home/mavinomichael/miniconda/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME}"
fi

merge_role() {
  local role_dir="$1"
  [[ -d "${role_dir}" ]] || return 1
  (cd "${TRAIN_ROOT}" && python3 scripts/model_merger.py --local_dir "${role_dir}")
}

eval_step() {
  local step_dir="$1"
  local step_name
  step_name=$(basename "${step_dir}")
  local planner_dir="${step_dir}/actor/planner"
  local executor_dir="${step_dir}/actor/executor"

  merge_role "${planner_dir}"
  merge_role "${executor_dir}"

  local planner_hf="${planner_dir}/huggingface"
  local executor_hf="${executor_dir}/huggingface"
  [[ -d "${planner_hf}" && -d "${executor_hf}" ]] || return 1

  (
    cd "${TRAIN_ROOT}"
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    python3 -m verl.extend_multi_agent.main_generation \
      task=babyai \
      runtime=qwen2_5_7b_1gpu_smoke \
      runtime.model_path="${planner_hf}" \
      runtime.nnodes=1 \
      runtime.n_gpus_per_node=1 \
      runtime.tensor_model_parallel_size=1 \
      runtime.eval_gpu_memory_utilization=0.55 \
      actor_rollout_ref.agentgym.env_addr="${ENV_SERVER_URL}" \
      extend_multi_agent.task_scope=babyai \
      extend_multi_agent.strict_json=true \
      extend_multi_agent.rollout_runtime=hf \
      extend_multi_agent.planner_model_path="${planner_hf}" \
      extend_multi_agent.executor_model_path="${executor_hf}" \
      extend_multi_agent.planner_ref_model_path="${planner_hf}" \
      extend_multi_agent.executor_ref_model_path="${executor_hf}" \
      data.batch_size=8 \
      rollout.gpu_memory_utilization=0.55 \
      rollout.max_num_batched_tokens=1024 \
      rollout.max_num_seqs=16 \
      agentgym.max_rounds=20
  ) 2>&1 | tee "${EVAL_LOG_ROOT}/${step_name}.log"
}

for step_dir in "${RUN_DIR}"/global_step_*; do
  [[ -d "${step_dir}" ]] || continue
  eval_step "${step_dir}"
done
