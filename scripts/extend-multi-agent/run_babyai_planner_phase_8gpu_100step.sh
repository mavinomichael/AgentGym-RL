#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
MODEL_PATH="${MODEL_PATH:-}"
PLANNER_MODEL_PATH="${PLANNER_MODEL_PATH:-${MODEL_PATH}}"
EXECUTOR_MODEL_PATH="${EXECUTOR_MODEL_PATH:-${MODEL_PATH}}"
PLANNER_REF_MODEL_PATH="${PLANNER_REF_MODEL_PATH:-${PLANNER_MODEL_PATH}}"
EXECUTOR_REF_MODEL_PATH="${EXECUTOR_REF_MODEL_PATH:-${EXECUTOR_MODEL_PATH}}"
SAVE_ROOT="${SAVE_ROOT:-/home/mavinomichael/agentgym_runs/saves/extend_multi_agent}"
ENV_SERVER_URL="${ENV_SERVER_URL:-http://127.0.0.1:36006}"
RAY_TMPDIR="${RAY_TMPDIR:-/home/mavinomichael/agentgym_runs/raytmp_extend}"
TMPDIR="${TMPDIR:-/home/mavinomichael/agentgym_runs/tmp_extend}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-agentgym-rl}"

if [ -z "${MODEL_PATH}" ]; then
  echo "Missing MODEL_PATH" >&2
  exit 1
fi

mkdir -p "${SAVE_ROOT}" "${RAY_TMPDIR}/spill" "${TMPDIR}"

if [[ -f /home/mavinomichael/miniconda/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /home/mavinomichael/miniconda/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME}"
fi

cd "${REPO_ROOT}/AgentGym-RL"

HYDRA_FULL_ERROR=1 \
PYTHONUNBUFFERED=1 \
TMPDIR="${TMPDIR}" \
RAY_TMPDIR="${RAY_TMPDIR}" \
RAY_object_spilling_config="{\"type\":\"filesystem\",\"params\":{\"directory_path\":\"${RAY_TMPDIR}/spill\"}}" \
python3 -m verl.extend_multi_agent.main_ppo \
  task=babyai \
  runtime=qwen2_5_7b_8gpu \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.agentgym.env_addr="${ENV_SERVER_URL}" \
  trainer.default_local_dir="${SAVE_ROOT}/planner_phase_8gpu_100step" \
  trainer.experiment_name=extend_babyai_planner_phase_8gpu_100step \
  trainer.project_name=extend_multi_agent \
  trainer.total_training_steps=100 \
  trainer.save_freq=20 \
  task.default_max_rounds=4 \
  actor_rollout_ref.agentgym.max_rounds=4 \
  actor_rollout_ref.agentgym.timeout=120 \
  algorithm.rounds_ctrl.rounds=4 \
  task.train_batch_size=8 \
  task.rollout_n=1 \
  task.ppo_mini_batch_size=8 \
  task.ppo_micro_batch_size_per_gpu=1 \
  task.critic_ppo_micro_batch_size_per_gpu=1 \
  task.rollout_max_tokens=96 \
  task.planner_max_tokens=40 \
  task.executor_max_tokens=40 \
  runtime.gpu_memory_utilization=0.35 \
  runtime.max_num_batched_tokens=2048 \
  runtime.max_num_seqs=16 \
  runtime.actor_ppo_max_token_len_per_gpu=4096 \
  runtime.critic_ppo_max_token_len_per_gpu=4096 \
  extend_multi_agent.train_role=planner \
  extend_multi_agent.freeze_planner=false \
  extend_multi_agent.freeze_executor=true \
  extend_multi_agent.task_scope=babyai \
  extend_multi_agent.strict_json=true \
  extend_multi_agent.rollout_runtime=hf \
  extend_multi_agent.planner_model_path="${PLANNER_MODEL_PATH}" \
  extend_multi_agent.executor_model_path="${EXECUTOR_MODEL_PATH}" \
  extend_multi_agent.planner_ref_model_path="${PLANNER_REF_MODEL_PATH}" \
  extend_multi_agent.executor_ref_model_path="${EXECUTOR_REF_MODEL_PATH}" \
  multi_agent.invalid_output.max_retries=0 \
  multi_agent.invalid_output.planner_max_retries=0 \
  multi_agent.debug.generation_progress=true \
  multi_agent.debug.trace_executor_payload=true \
  multi_agent.debug.trace_dir="${SAVE_ROOT}/planner_phase_8gpu_100step/trace_train" \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.grad_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  critic.model.fsdp_config.param_offload=true \
  critic.model.fsdp_config.grad_offload=true \
  critic.model.fsdp_config.optimizer_offload=true
