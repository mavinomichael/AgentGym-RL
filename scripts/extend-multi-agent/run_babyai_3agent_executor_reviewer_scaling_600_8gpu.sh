#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

TASK_NAME="babyai"
TRAIN_ROOT=$(multi_agent::train_root)

export MODEL_PATH="${MODEL_PATH:-/home/mavinomichael/AgentGym-RL/models/Qwen2.5-7B-Instruct}"
export DATA_ROOT="${DATA_ROOT:-$(dirname "$TRAIN_ROOT")}"
export ENV_SERVER_URL="${ENV_SERVER_URL:-http://127.0.0.1:36005}"
export SAVE_ROOT="${SAVE_ROOT:-/home/mavinomichael/agentgym_runs/saves}"
export LOG_ROOT="${LOG_ROOT:-/home/mavinomichael/agentgym_runs/logs}"
export EXP_NAME="${EXP_NAME:-babyai_3agent_executor_reviewer_scaling_600_8gpu_no_tags_v1}"
export N_GPUS="${N_GPUS:-8}"
export NNODES="${NNODES:-1}"
export TP_SIZE="${TP_SIZE:-1}"
export TMPDIR="${TMPDIR:-/home/mavinomichael/agentgym_runs/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-/home/mavinomichael/agentgym_runs/raytmp}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-agentgym-rl}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-601}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export SAVE_STEPS="${SAVE_STEPS:-[]}"
export PLANNER_MAX_TOKENS="${PLANNER_MAX_TOKENS:-96}"
export EXECUTOR_REVIEWER_MAX_TOKENS="${EXECUTOR_REVIEWER_MAX_TOKENS:-64}"
export ROUNDS_CTRL_TYPE="${ROUNDS_CTRL_TYPE:-scaling_inter_stepwise}"
export ROUNDS_CTRL_ROUNDS="${ROUNDS_CTRL_ROUNDS:-[6,8,10,13,16,20]}"
export ROUNDS_CTRL_STEPS="${ROUNDS_CTRL_STEPS:-100}"
export RESUME_MODE="${RESUME_MODE:-disable}"

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

cleanup_old_ray_sessions
multi_agent::print_run_header "babyai-3agent-executor-reviewer-scaling-600-8gpu" "$TASK_NAME"
printf '  %-22s %s\n' "RUN_DIR" "$RUN_DIR"
printf '  %-22s %s\n' "TRACE_DIR" "$TRACE_DIR"
printf '  %-22s %s\n' "TOTAL_STEPS" "$TOTAL_TRAINING_STEPS"
printf '  %-22s %s\n' "SAVE_FREQ" "$SAVE_FREQ"
printf '  %-22s %s\n' "SAVE_STEPS" "$SAVE_STEPS"
printf '  %-22s %s\n' "RESUME_MODE" "$RESUME_MODE"
printf '  %-22s %s\n' "PLANNER_TOKENS" "$PLANNER_MAX_TOKENS"
printf '  %-22s %s\n' "EXEC_REVIEW_TOKENS" "$EXECUTOR_REVIEWER_MAX_TOKENS"
printf '  %-22s %s\n' "ROUNDS_CTRL" "$ROUNDS_CTRL_TYPE"
printf '  %-22s %s\n' "ROUNDS" "$ROUNDS_CTRL_ROUNDS"
printf '  %-22s %s\n' "ROUND_STEP" "$ROUNDS_CTRL_STEPS"

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
    trainer.resume_mode=$RESUME_MODE \
    trainer.save_freq=$SAVE_FREQ \
    trainer.save_steps=$SAVE_STEPS \
    algo.use_kl_loss=true \
    algo.kl_coef=0.001 \
    multi_agent.topology=planner_executor_reviewer \
    multi_agent.roles.planner.max_tokens=$PLANNER_MAX_TOKENS \
    multi_agent.roles.planner.temperature=0.2 \
    multi_agent.roles.executor_reviewer.max_tokens=$EXECUTOR_REVIEWER_MAX_TOKENS \
    multi_agent.roles.executor_reviewer.temperature=0.2 \
    multi_agent.roles.executor_reviewer.ppo_weight=0.5 \
    multi_agent.roles.executor_reviewer.kl_weight=6.0 \
    actor_rollout_ref.actor.planner_kl_weight=4.0 \
    multi_agent.invalid_output.policy=terminate_with_penalty \
    multi_agent.invalid_output.penalty=-0.2 \
    multi_agent.invalid_output.max_retries=3 \
    multi_agent.invalid_output.retry_temperature=0.2 \
    multi_agent.invalid_output.retry_max_tokens=80 \
    multi_agent.invalid_output.planner_max_retries=0 \
    multi_agent.invalid_output.planner_retry_temperature=0.1 \
    multi_agent.invalid_output.planner_retry_max_tokens=$PLANNER_MAX_TOKENS \
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

printf '3-agent BabyAI executor-reviewer scaling run complete.\n'
