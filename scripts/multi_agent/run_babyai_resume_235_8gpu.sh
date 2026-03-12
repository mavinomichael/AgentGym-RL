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
export SAVE_ROOT="${SAVE_ROOT:-/mnt/data/saves}"
export LOG_ROOT="${LOG_ROOT:-/mnt/data/logs}"
export EXP_NAME="${EXP_NAME:-babyai_resume_235_8gpu}"
export N_GPUS="${N_GPUS:-8}"
export NNODES="${NNODES:-1}"
export TP_SIZE="${TP_SIZE:-1}"
export TMPDIR="${TMPDIR:-/mnt/data/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-/mnt/data/raytmp}"
export EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.72}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-agentgym-rl}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-236}"
export CHECKPOINT_STEP="${CHECKPOINT_STEP:-235}"
export SAVE_FREQ="${SAVE_FREQ:-100}"
export END_EVAL_TRACE="${END_EVAL_TRACE:-false}"
export RUN_TAG="${RUN_TAG:-step${CHECKPOINT_STEP}}"

SOURCE_CHECKPOINT_DIR="${SOURCE_CHECKPOINT_DIR:-/mnt/data/saves/agentgym_multi_agent/babyai_planner_sanity_15_8gpu/global_step_15}"
RUN_DIR="$SAVE_ROOT/agentgym_multi_agent/$EXP_NAME"
LOG_DIR="$LOG_ROOT/$EXP_NAME"
TRAIN_LOG="$LOG_DIR/train_resume_${RUN_TAG}.log"
MERGE_LOG="$LOG_DIR/merge_checkpoints.log"
EVAL_LOG="$LOG_DIR/eval_${RUN_TAG}.log"
EVAL_TRACE_DIR="$LOG_DIR/trace_eval_${RUN_TAG}"
REPORT_DIR="$REPO_ROOT/reports/babyai_multi_agent_diagnostics_2026-03-09"
REPORT_EVAL_LOG="$REPORT_DIR/eval_stage${CHECKPOINT_STEP}_resume.log"
REPORT_TRAIN_LOG="$REPORT_DIR/train_stage${CHECKPOINT_STEP}_resume.log"
REPORT_STEP_LOG="$REPORT_DIR/train_resume_${RUN_TAG}_vm.log"
PROGRESS_REPORT_SCRIPT="$REPO_ROOT/scripts/generate_babyai_progress_report.py"
MERGER_PID=""

mkdir -p "$RUN_DIR" "$LOG_DIR" "$TMPDIR" "$RAY_TMPDIR" "$REPORT_DIR"

if [[ -f /home/mavinomichael/miniconda/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  set +u
  source /home/mavinomichael/miniconda/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV_NAME"
  set -u
  PYTHON_BIN=$(command -v python)
fi

cleanup_old_ray_sessions() {
  local ray_root="/tmp/ray"
  if [[ ! -d "$ray_root" ]]; then
    return
  fi

  local active_sessions
  active_sessions=$(
    ps aux \
      | sed -n 's#.*\(/tmp/ray/session_[^ /]*\).*#\1#p' \
      | sort -u
  )

  for session_dir in "$ray_root"/session_*; do
    [[ -d "$session_dir" ]] || continue
    if grep -Fxq "$session_dir" <<<"$active_sessions"; then
      continue
    fi
    rm -rf "$session_dir"
  done
}

has_existing_checkpoints() {
  find "$RUN_DIR" -maxdepth 1 -type d -name 'global_step_*' | grep -q .
}

verify_source_checkpoint() {
  if [[ ! -d "$SOURCE_CHECKPOINT_DIR" ]]; then
    printf 'Missing source checkpoint: %s\n' "$SOURCE_CHECKPOINT_DIR" >&2
    exit 1
  fi
  if [[ ! -d "$SOURCE_CHECKPOINT_DIR/actor" || ! -d "$SOURCE_CHECKPOINT_DIR/critic" ]]; then
    printf 'Source checkpoint is incomplete: %s\n' "$SOURCE_CHECKPOINT_DIR" >&2
    exit 1
  fi
}

determine_resume_mode() {
  if has_existing_checkpoints; then
    printf '%s\n' "auto"
  else
    verify_source_checkpoint
    printf '%s\n' "$SOURCE_CHECKPOINT_DIR"
  fi
}

list_saved_steps() {
  find "$RUN_DIR" -maxdepth 1 -type d -name 'global_step_*' -print \
    | sed -E 's#.*global_step_([0-9]+)$#\1#' \
    | sort -n
}

resolve_eval_step() {
  if [[ -d "$RUN_DIR/global_step_${CHECKPOINT_STEP}" ]]; then
    printf '%s\n' "$CHECKPOINT_STEP"
    return 0
  fi

  local latest_step
  latest_step=$(list_saved_steps | tail -n 1)
  if [[ -z "$latest_step" ]]; then
    printf 'No saved checkpoints found in %s\n' "$RUN_DIR" >&2
    exit 1
  fi
  printf '%s\n' "$latest_step"
}

merge_checkpoint_if_ready() {
  local step="$1"
  local checkpoint_actor_dir="$RUN_DIR/global_step_${step}/actor"
  local merged_dir="$checkpoint_actor_dir/huggingface"
  local lock_file="$checkpoint_actor_dir/.merge_in_progress"

  if [[ ! -d "$checkpoint_actor_dir" || -d "$merged_dir" ]]; then
    return 0
  fi
  if [[ -f "$lock_file" ]]; then
    return 0
  fi

  touch "$lock_file"
  {
    printf '[merge] step=%s actor_dir=%s\n' "$step" "$checkpoint_actor_dir"
    (
      cd "$TRAIN_ROOT"
      "$PYTHON_BIN" scripts/model_merger.py --local_dir "$checkpoint_actor_dir"
    )
    printf '[merge] step=%s complete\n' "$step"
  } >> "$MERGE_LOG" 2>&1 || {
    printf '[merge] step=%s failed\n' "$step" >> "$MERGE_LOG"
    rm -f "$lock_file"
    return 1
  }

  rm -f "$lock_file"
}

start_merger_sidecar() {
  (
    while true; do
      local step
      for step in $(seq "$SAVE_FREQ" "$SAVE_FREQ" "$CHECKPOINT_STEP"); do
        merge_checkpoint_if_ready "$step" || true
      done

      if [[ -f "$RUN_DIR/.training_complete" ]]; then
        merge_checkpoint_if_ready "$(resolve_eval_step)" || true
      fi

      if [[ -f "$RUN_DIR/.training_complete" ]] && [[ -d "$RUN_DIR/global_step_$(resolve_eval_step)/actor/huggingface" ]]; then
        break
      fi
      sleep 45
    done
  ) &
  MERGER_PID=$!
}

stop_merger_sidecar() {
  if [[ -n "$MERGER_PID" ]]; then
    kill "$MERGER_PID" 2>/dev/null || true
    wait "$MERGER_PID" 2>/dev/null || true
    MERGER_PID=""
  fi
}

run_training() {
  local resume_mode="$1"

  multi_agent::print_run_header "resume-235-8gpu" "$TASK_NAME"
  printf '  %-18s %s\n' "RUN_DIR" "$RUN_DIR"
  printf '  %-18s %s\n' "LOG_DIR" "$LOG_DIR"
  printf '  %-18s %s\n' "RESUME_MODE" "$resume_mode"
  printf '  %-18s %s\n' "SOURCE_CKPT" "$SOURCE_CHECKPOINT_DIR"
  printf '  %-18s %s\n' "TOTAL_STEPS" "$TOTAL_TRAINING_STEPS"
  printf '  %-18s %s\n' "CHECKPOINT_STEP" "$CHECKPOINT_STEP"
  printf '  %-18s %s\n' "SAVE_FREQ" "$SAVE_FREQ"
  printf '  %-18s %s\n' "GPU_MEM_UTIL" "$GPU_MEMORY_UTILIZATION"
  printf '  %-18s %s\n' "MAX_BATCHED_TOK" "$MAX_NUM_BATCHED_TOKENS"
  printf '  %-18s %s\n' "MAX_NUM_SEQS" "$MAX_NUM_SEQS"

  cleanup_old_ray_sessions
  rm -f "$RUN_DIR/.training_complete"
  start_merger_sidecar

  (
    cd "$TRAIN_ROOT"
    HYDRA_FULL_ERROR=1 \
    PYTHONUNBUFFERED=1 \
    TMPDIR="$TMPDIR" \
    RAY_TMPDIR="$RAY_TMPDIR" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
    MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
    MAX_NUM_SEQS="$MAX_NUM_SEQS" \
    MODEL_PATH="$MODEL_PATH" \
    DATA_ROOT="$DATA_ROOT" \
    ENV_SERVER_URL="$ENV_SERVER_URL" \
    SAVE_ROOT="$SAVE_ROOT" \
    EXP_NAME="$EXP_NAME" \
    N_GPUS="$N_GPUS" \
    TP_SIZE="$TP_SIZE" \
    "$PYTHON_BIN" -m verl.multi_agent.main_ppo \
      task=$TASK_NAME \
      task.train_file=AgentItemId/train/babyai_train.json \
      task.train_batch_size=8 \
      task.rollout_n=4 \
      task.ppo_mini_batch_size=8 \
      runtime=qwen2_5_7b_8gpu \
      runtime.nnodes=$NNODES \
      runtime.n_gpus_per_node=$N_GPUS \
      runtime.tensor_model_parallel_size=$TP_SIZE \
      algo=multi_agent_gae \
      trainer.default_local_dir="$RUN_DIR" \
      trainer.experiment_name="$EXP_NAME" \
      trainer.nnodes=$NNODES \
      trainer.n_gpus_per_node=$N_GPUS \
      trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
      trainer.resume_mode="$resume_mode" \
      trainer.save_freq=$SAVE_FREQ \
      multi_agent.roles.planner.max_tokens=32 \
      multi_agent.roles.planner.temperature=0.3 \
      multi_agent.invalid_output.policy=terminate_with_penalty \
      multi_agent.invalid_output.penalty=-0.2 \
      multi_agent.invalid_output.max_retries=2 \
      multi_agent.invalid_output.retry_temperature=0.2 \
      multi_agent.invalid_output.retry_max_tokens=80
  ) 2>&1 | tee "$TRAIN_LOG"

  touch "$RUN_DIR/.training_complete"
  merge_checkpoint_if_ready "$(resolve_eval_step)"
  stop_merger_sidecar
}

run_end_eval() {
  local eval_step
  eval_step=$(resolve_eval_step)
  local merged_model_path="$RUN_DIR/global_step_${eval_step}/actor/huggingface"
  if [[ ! -d "$merged_model_path" ]]; then
    printf 'Missing merged model for end eval: %s\n' "$merged_model_path" >&2
    exit 1
  fi

  cleanup_old_ray_sessions
  rm -rf "$EVAL_TRACE_DIR"
  mkdir -p "$EVAL_TRACE_DIR"

  (
    cd "$TRAIN_ROOT"
    HYDRA_FULL_ERROR=1 \
    PYTHONUNBUFFERED=1 \
    TMPDIR="$TMPDIR" \
    RAY_TMPDIR="$RAY_TMPDIR" \
    MODEL_PATH="$merged_model_path" \
    DATA_ROOT="$DATA_ROOT" \
    ENV_SERVER_URL="$ENV_SERVER_URL" \
    SAVE_ROOT="$SAVE_ROOT" \
    EXP_NAME="${EXP_NAME}_eval_step${eval_step}" \
    N_GPUS=1 \
    TP_SIZE=1 \
    "$PYTHON_BIN" -m verl.multi_agent.main_generation \
      task=$TASK_NAME \
      runtime=qwen2_5_7b_1gpu_smoke \
      runtime.nnodes=1 \
      runtime.n_gpus_per_node=1 \
      runtime.tensor_model_parallel_size=1 \
      runtime.eval_gpu_memory_utilization="$EVAL_GPU_MEMORY_UTILIZATION" \
      runtime.model_path="$merged_model_path" \
      runtime.env_server_url="$ENV_SERVER_URL" \
      rollout.gpu_memory_utilization="$EVAL_GPU_MEMORY_UTILIZATION" \
      rollout.max_num_batched_tokens=2048 \
      rollout.max_num_seqs=64 \
      multi_agent.invalid_output.max_retries=2 \
      multi_agent.invalid_output.retry_temperature=0.2 \
      multi_agent.invalid_output.retry_max_tokens=80 \
      multi_agent.debug.trace_executor_payload="$END_EVAL_TRACE" \
      multi_agent.debug.trace_dir="$EVAL_TRACE_DIR" \
      multi_agent.debug.trace_max_chars=800
  ) 2>&1 | tee "$EVAL_LOG"

  cp -f "$EVAL_LOG" "$REPORT_EVAL_LOG"
  cp -f "$TRAIN_LOG" "$REPORT_TRAIN_LOG"
  cp -f "$TRAIN_LOG" "$REPORT_STEP_LOG"
}

update_report_assets() {
  if [[ -f "$PROGRESS_REPORT_SCRIPT" ]]; then
    "$PYTHON_BIN" "$PROGRESS_REPORT_SCRIPT"
  fi
}

resume_mode=$(determine_resume_mode)
run_training "$resume_mode"
run_end_eval
update_report_assets

printf 'Resume run complete.\n'
printf '  %-18s %s\n' "train_log" "$TRAIN_LOG"
printf '  %-18s %s\n' "merge_log" "$MERGE_LOG"
printf '  %-18s %s\n' "eval_log" "$EVAL_LOG"
printf '  %-18s %s\n' "run_dir" "$RUN_DIR"
