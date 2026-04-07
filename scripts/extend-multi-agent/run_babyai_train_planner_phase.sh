#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
TRAIN_ROOT="${REPO_ROOT}/AgentGym-RL"
MODEL_PATH="${MODEL_PATH:-}"
SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/output/extend-multi-agent-ppo}"
ENV_SERVER_URL="${ENV_SERVER_URL:-http://127.0.0.1:36005}"
N_GPUS="${N_GPUS:-1}"

if [ -z "${MODEL_PATH}" ]; then
  echo "Missing MODEL_PATH" >&2
  exit 1
fi

python3 -m verl.extend_multi_agent.main_ppo \
  task=babyai \
  runtime=qwen2_5_7b_1gpu_smoke \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.agentgym.env_addr="${ENV_SERVER_URL}" \
  trainer.default_local_dir="${SAVE_ROOT}/planner_phase" \
  trainer.experiment_name=extend_babyai_planner_phase \
  trainer.project_name=extend_multi_agent \
  extend_multi_agent.train_role=planner \
  extend_multi_agent.freeze_executor=true \
  extend_multi_agent.freeze_planner=false \
  extend_multi_agent.task_scope=babyai \
  extend_multi_agent.strict_json=true
