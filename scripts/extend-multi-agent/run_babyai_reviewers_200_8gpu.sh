#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

TASK_NAME="babyai"
TRAIN_ROOT=$(multi_agent::train_root)
REPO_ROOT=$(multi_agent::repo_root)

export MODEL_PATH="${MODEL_PATH:-/home/mavinomichael/AgentGym-RL/models/Qwen2.5-7B-Instruct}"
export DATA_ROOT="${DATA_ROOT:-$(dirname "$TRAIN_ROOT")}"
export ENV_SERVER_URL="${ENV_SERVER_URL:-http://127.0.0.1:36005}"
export SAVE_ROOT="${SAVE_ROOT:-/dev/shm/agentgym_saves}"
export LOG_ROOT="${LOG_ROOT:-/dev/shm/agentgym_logs}"
export EXP_NAME="${EXP_NAME:-babyai_reviewers_200_8gpu}"
export N_GPUS="${N_GPUS:-8}"
export NNODES="${NNODES:-1}"
export TP_SIZE="${TP_SIZE:-1}"
export TMPDIR="${TMPDIR:-/dev/shm/agentgym_tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-/dev/shm/agentgym_raytmp}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-agentgym-rl}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-201}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-50 100 150 200}"
export ROUNDS_CTRL_TYPE="${ROUNDS_CTRL_TYPE:-fixed}"
export ROUNDS_CTRL_ROUNDS="${ROUNDS_CTRL_ROUNDS:-20}"
export ROUNDS_CTRL_STEPS="${ROUNDS_CTRL_STEPS:-100}"

RUN_DIR="$SAVE_ROOT/agentgym_multi_agent/$EXP_NAME"
LOG_DIR="$LOG_ROOT/$EXP_NAME"
TRACE_DIR="$LOG_DIR/trace_train"
mkdir -p "$RUN_DIR" "$LOG_DIR" "$TRACE_DIR" "$TMPDIR" "$RAY_TMPDIR" "$RAY_TMPDIR/spill"

if [[ -f /home/mavinomichael/miniconda/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  set +u
  source /home/mavinomichael/miniconda/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV_NAME"
  set -u
  PYTHON_BIN=$(command -v python)
fi

cleanup_old_ray_sessions() {
  local -a roots=("/tmp/ray" "$RAY_TMPDIR" "$RAY_TMPDIR/ray")
  local active_sessions
  active_sessions=$(ps aux | sed -n 's#.*\(/[^ ]*/ray/session_[^ /]*\).*#\1#p' | sort -u)
  for ray_root in "${roots[@]}"; do
    [[ -d "$ray_root" ]] || continue
    for session_dir in "$ray_root"/session_*; do
      [[ -d "$session_dir" ]] || continue
      if [[ -n "$active_sessions" ]] && grep -Fxq "$session_dir" <<<"$active_sessions"; then
        continue
      fi
      rm -rf "$session_dir"
    done
  done
  rm -rf "$RAY_TMPDIR/spill"/*
}

merge_checkpoint() {
  local step="$1"
  local actor_dir="$RUN_DIR/global_step_${step}/actor"
  [[ -d "$actor_dir" ]] || return 0
  (
    cd "$TRAIN_ROOT"
    "$PYTHON_BIN" scripts/model_merger.py --local_dir "$actor_dir"
  ) 2>&1 | tee "$LOG_DIR/merge_step${step}.log"
}

run_eval() {
  local step="$1"
  local model_path="$RUN_DIR/global_step_${step}/actor/huggingface"
  [[ -d "$model_path" ]] || return 0
  (
    cd "$TRAIN_ROOT"
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "$PYTHON_BIN" -m verl.extend_multi_agent.main_generation \
      task=$TASK_NAME \
      runtime=qwen2_5_7b_1gpu_smoke \
      runtime.nnodes=1 \
      runtime.n_gpus_per_node=1 \
      runtime.tensor_model_parallel_size=1 \
      runtime.eval_gpu_memory_utilization=0.55 \
      runtime.model_path="$model_path" \
      runtime.env_server_url="$ENV_SERVER_URL" \
      data.batch_size=8 \
      rollout.gpu_memory_utilization=0.55 \
      rollout.max_num_batched_tokens=1024 \
      rollout.max_num_seqs=16 \
      multi_agent.topology=planner_executor_reviewers \
      multi_agent.invalid_output.max_retries=5 \
      multi_agent.invalid_output.retry_temperature=0.2 \
      multi_agent.invalid_output.retry_max_tokens=80 \
      multi_agent.invalid_output.planner_max_retries=5 \
      multi_agent.invalid_output.planner_retry_temperature=0.1 \
      multi_agent.invalid_output.planner_retry_max_tokens=64 \
      multi_agent.roles.planner.max_tokens=64 \
      multi_agent.roles.planner.temperature=0.7 \
      multi_agent.roles.planner_reviewer.max_tokens=96 \
      multi_agent.roles.planner_reviewer.temperature=0.2 \
      multi_agent.roles.executor_reviewer.max_tokens=64 \
      multi_agent.roles.executor_reviewer.temperature=0.2 \
      multi_agent.debug.trace_executor_payload=false
  ) 2>&1 | tee "$LOG_DIR/eval_step${step}.log"
}

cleanup_old_ray_sessions
multi_agent::print_run_header "babyai-reviewers-200-8gpu" "$TASK_NAME"
printf '  %-18s %s\n' "RUN_DIR" "$RUN_DIR"
printf '  %-18s %s\n' "TRACE_DIR" "$TRACE_DIR"
printf '  %-18s %s\n' "TOTAL_STEPS" "$TOTAL_TRAINING_STEPS"
printf '  %-18s %s\n' "SAVE_FREQ" "$SAVE_FREQ"
printf '  %-18s %s\n' "CHECKPOINTS" "$CHECKPOINT_STEPS"
printf '  %-18s %s\n' "ROUNDS_CTRL" "$ROUNDS_CTRL_TYPE"
printf '  %-18s %s\n' "ROUNDS" "$ROUNDS_CTRL_ROUNDS"
printf '  %-18s %s\n' "ROUND_STEP" "$ROUNDS_CTRL_STEPS"
(
  cd "$TRAIN_ROOT"
  HYDRA_FULL_ERROR=1 \
  PYTHONUNBUFFERED=1 \
  TMPDIR="$TMPDIR" \
  RAY_TMPDIR="$RAY_TMPDIR" \
  RAY_object_spilling_config="{\"type\":\"filesystem\",\"params\":{\"directory_path\":\"$RAY_TMPDIR/spill\"}}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  MODEL_PATH="$MODEL_PATH" \
  DATA_ROOT="$DATA_ROOT" \
  ENV_SERVER_URL="$ENV_SERVER_URL" \
  SAVE_ROOT="$SAVE_ROOT" \
  EXP_NAME="$EXP_NAME" \
  N_GPUS="$N_GPUS" \
  TP_SIZE="$TP_SIZE" \
  "$PYTHON_BIN" -m verl.extend_multi_agent.main_ppo \
    task=$TASK_NAME \
    task.train_file=AgentItemId/train/babyai_train.json \
    task.train_batch_size=8 \
    task.rollout_n=2 \
    task.ppo_mini_batch_size=8 \
    runtime=qwen2_5_7b_8gpu \
    runtime.nnodes=$NNODES \
    runtime.n_gpus_per_node=$N_GPUS \
    runtime.tensor_model_parallel_size=$TP_SIZE \
    runtime.gpu_memory_utilization=0.55 \
    runtime.max_num_batched_tokens=1024 \
    runtime.max_num_seqs=64 \
    algo=multi_agent_gae \
    algorithm.rounds_ctrl.type=$ROUNDS_CTRL_TYPE \
    algorithm.rounds_ctrl.rounds=$ROUNDS_CTRL_ROUNDS \
    algorithm.rounds_ctrl.steps_scaling_inter=$ROUNDS_CTRL_STEPS \
    trainer.default_local_dir="$RUN_DIR" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.nnodes=$NNODES \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.resume_mode=disable \
    trainer.save_freq=$SAVE_FREQ \
    algo.use_kl_loss=true \
    algo.kl_coef=0.001 \
    multi_agent.topology=planner_executor_reviewers \
    multi_agent.roles.planner.max_tokens=64 \
    multi_agent.roles.planner.temperature=0.7 \
    multi_agent.roles.planner_reviewer.max_tokens=96 \
    multi_agent.roles.planner_reviewer.temperature=0.2 \
    multi_agent.roles.executor_reviewer.max_tokens=64 \
    multi_agent.roles.executor_reviewer.temperature=0.2 \
    multi_agent.invalid_output.policy=terminate_with_penalty \
    multi_agent.invalid_output.penalty=-0.2 \
    multi_agent.invalid_output.max_retries=5 \
    multi_agent.invalid_output.retry_temperature=0.2 \
    multi_agent.invalid_output.retry_max_tokens=80 \
    multi_agent.invalid_output.planner_max_retries=5 \
    multi_agent.invalid_output.planner_retry_temperature=0.1 \
    multi_agent.invalid_output.planner_retry_max_tokens=64 \
    multi_agent.debug.trace_executor_payload=true \
    multi_agent.debug.trace_dir="$TRACE_DIR" \
    multi_agent.debug.trace_max_chars=800 \
    multi_agent.debug.trace_first_training_steps=15 \
    multi_agent.debug.trace_every_training_steps=5 \
    multi_agent.debug.trace_on_planner_invalid=true \
    multi_agent.debug.trace_on_planner_fallback=true \
    multi_agent.debug.trace_on_executor_invalid_format=true \
    multi_agent.debug.trace_on_invalid_action=true
) 2>&1 | tee "$LOG_DIR/train_step200.log"

for step in $CHECKPOINT_STEPS; do
  merge_checkpoint "$step"
  run_eval "$step"
done

printf '4-agent reviewer run complete.\n'
