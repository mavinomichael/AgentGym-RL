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
export EXP_NAME="${EXP_NAME:-babyai_resume_500_trace_8gpu}"
export SOURCE_EXP_NAME="${SOURCE_EXP_NAME:-babyai_diagnostic_100_8gpu_v2}"
export SOURCE_STEP="${SOURCE_STEP:-100}"
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
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-501}"
export CHECKPOINT_STEP="${CHECKPOINT_STEP:-500}"
export SAVE_FREQ="${SAVE_FREQ:-100}"
export RESUME_MODE="${RESUME_MODE:-$SAVE_ROOT/agentgym_multi_agent/$SOURCE_EXP_NAME/global_step_$SOURCE_STEP}"

RUN_DIR="$SAVE_ROOT/agentgym_multi_agent/$EXP_NAME"
LOG_DIR="$LOG_ROOT/$EXP_NAME"
TRACE_DIR="$LOG_DIR/trace_train"
REPORT_DIR="$REPO_ROOT/reports/babyai_multi_agent_diagnostics_2026-03-11"
TRACE_SUMMARY_PATH="$REPORT_DIR/resume_step500_trace_summary.txt"
TRAIN_LOG="$LOG_DIR/train_step500_resume.log"
MERGE_LOG="$LOG_DIR/merge_step500.log"
EVAL_LOG="$LOG_DIR/eval_step500.log"

mkdir -p "$LOG_DIR" "$TRACE_DIR" "$TMPDIR" "$RAY_TMPDIR" "$RAY_TMPDIR/spill" "$REPORT_DIR"

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
  active_sessions=$(
    ps aux | sed -n 's#.*\(/[^ ]*/ray/session_[^ /]*\).*#\1#p' | sort -u
  )

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
  local checkpoint_actor_dir="$RUN_DIR/global_step_${CHECKPOINT_STEP}/actor"
  if [[ ! -d "$checkpoint_actor_dir" ]]; then
    printf 'Missing checkpoint actor directory: %s\n' "$checkpoint_actor_dir" >&2
    exit 1
  fi

  (
    cd "$TRAIN_ROOT"
    "$PYTHON_BIN" scripts/model_merger.py --local_dir "$checkpoint_actor_dir"
  ) 2>&1 | tee "$MERGE_LOG"
}

generate_trace_summary() {
  "$PYTHON_BIN" - "$TRACE_DIR" "$TRACE_SUMMARY_PATH" <<'PY'
import collections
import glob
import json
import os
import sys

trace_dir = sys.argv[1]
report_path = sys.argv[2]
paths = sorted(glob.glob(os.path.join(trace_dir, "executor_payload_trace_rank*.jsonl")))
if not paths:
    raise SystemExit(f"No trace files found in {trace_dir}")

events = []
for path in paths:
    with open(path, encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))

events.sort(key=lambda item: (int(item.get("training_step") or -1), int(item.get("round") or 0), int(item.get("rank") or 0), int(item.get("item_id") or 0)))

def first_step(predicate):
    for event in events:
        if predicate(event):
            return event.get("training_step")
    return None

first_planner_invalid = first_step(lambda e: not bool(e.get("planner_validation_valid", True)))
first_planner_fallback = first_step(lambda e: bool(e.get("planner_fallback_used", False)))
first_executor_invalid_format = first_step(lambda e: not bool(e.get("executor_native_format_valid", True)))
first_invalid_action = first_step(lambda e: not bool(e.get("executor_action_valid", True)))

events_by_step = collections.OrderedDict()
for event in events:
    step = event.get("training_step")
    events_by_step.setdefault(step, []).append(event)

with open(report_path, "w", encoding="utf-8") as report:
    report.write("BabyAI 500-Step Resume Trace Summary\n")
    report.write(f"Trace dir: {trace_dir}\n")
    report.write(f"Total traced events: {len(events)}\n")
    report.write(f"Traced training steps: {', '.join(str(step) for step in events_by_step)}\n")
    report.write(f"First planner_invalid_format step: {first_planner_invalid}\n")
    report.write(f"First planner_fallback step: {first_planner_fallback}\n")
    report.write(f"First executor_invalid_format step: {first_executor_invalid_format}\n")
    report.write(f"First invalid_action step: {first_invalid_action}\n\n")

    for step, step_events in events_by_step.items():
        reasons = sorted({reason for event in step_events for reason in event.get("trace_reasons", [])})
        planner_invalid = [event for event in step_events if not bool(event.get("planner_validation_valid", True))]
        planner_fallback = [event for event in step_events if bool(event.get("planner_fallback_used", False))]
        executor_invalid_format = [event for event in step_events if not bool(event.get("executor_native_format_valid", True))]
        invalid_action = [event for event in step_events if not bool(event.get("executor_action_valid", True))]
        report.write("=" * 100 + "\n")
        report.write(f"training_step: {step}\n")
        report.write(f"trace_reasons: {', '.join(reasons) if reasons else 'none'}\n")
        report.write(f"events: {len(step_events)}\n")
        report.write(f"planner_invalid_count: {len(planner_invalid)}\n")
        report.write(f"planner_fallback_count: {len(planner_fallback)}\n")
        report.write(f"executor_invalid_format_count: {len(executor_invalid_format)}\n")
        report.write(f"invalid_action_count: {len(invalid_action)}\n")
        report.write(f"planner_retry_count: {sum(int(event.get('planner_retry_count_total', 0)) for event in step_events)}\n\n")

print(report_path)
PY
}

run_eval() {
  local model_path="$RUN_DIR/global_step_${CHECKPOINT_STEP}/actor/huggingface"
  if [[ ! -d "$model_path" ]]; then
    printf 'Missing merged model directory: %s\n' "$model_path" >&2
    exit 1
  fi

  (
    cd "$TRAIN_ROOT"
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "$PYTHON_BIN" -m verl.multi_agent.main_generation \
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
      multi_agent.invalid_output.max_retries=2 \
      multi_agent.invalid_output.retry_temperature=0.2 \
      multi_agent.invalid_output.retry_max_tokens=80 \
      multi_agent.invalid_output.planner_max_retries=2 \
      multi_agent.invalid_output.planner_retry_temperature=0.1 \
      multi_agent.invalid_output.planner_retry_max_tokens=20 \
      multi_agent.debug.trace_executor_payload=false \
      multi_agent.debug.trace_dir="$LOG_DIR/trace_eval_step500" \
      multi_agent.debug.trace_max_chars=800
  ) 2>&1 | tee "$EVAL_LOG"
}

multi_agent::print_run_header "resume-500-trace-8gpu" "$TASK_NAME"
printf '  %-18s %s\n' "TRACE_DIR" "$TRACE_DIR"
printf '  %-18s %s\n' "TRACE_SUMMARY" "$TRACE_SUMMARY_PATH"
printf '  %-18s %s\n' "TOTAL_STEPS" "$TOTAL_TRAINING_STEPS"
printf '  %-18s %s\n' "CHECKPOINT_STEP" "$CHECKPOINT_STEP"
printf '  %-18s %s\n' "SAVE_FREQ" "$SAVE_FREQ"
printf '  %-18s %s\n' "RESUME_MODE" "$RESUME_MODE"

cleanup_old_ray_sessions

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
  "$PYTHON_BIN" -m verl.multi_agent.main_ppo \
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
    algo.use_kl_loss=true \
    algo.kl_coef=0.001 \
    trainer.default_local_dir="$RUN_DIR" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.nnodes=$NNODES \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.resume_mode="$RESUME_MODE" \
    trainer.save_freq=$SAVE_FREQ \
    multi_agent.roles.planner.max_tokens=20 \
    multi_agent.roles.planner.temperature=0.2 \
    actor_rollout_ref.actor.planner_kl_weight=4.0 \
    multi_agent.invalid_output.policy=terminate_with_penalty \
    multi_agent.invalid_output.penalty=-0.2 \
    multi_agent.invalid_output.max_retries=2 \
    multi_agent.invalid_output.retry_temperature=0.2 \
    multi_agent.invalid_output.retry_max_tokens=80 \
    multi_agent.invalid_output.planner_max_retries=2 \
    multi_agent.invalid_output.planner_retry_temperature=0.1 \
    multi_agent.invalid_output.planner_retry_max_tokens=20 \
    multi_agent.debug.trace_executor_payload=true \
    multi_agent.debug.trace_dir="$TRACE_DIR" \
    multi_agent.debug.trace_max_chars=800 \
    multi_agent.debug.trace_first_training_steps=115 \
    multi_agent.debug.trace_every_training_steps=5 \
    multi_agent.debug.trace_on_planner_invalid=true \
    multi_agent.debug.trace_on_planner_fallback=true \
    multi_agent.debug.trace_on_executor_invalid_format=true \
    multi_agent.debug.trace_on_invalid_action=true
) 2>&1 | tee "$TRAIN_LOG"

merge_checkpoint
generate_trace_summary
run_eval

printf 'Resume run complete.\n'
printf '  %-18s %s\n' "train_log" "$TRAIN_LOG"
printf '  %-18s %s\n' "trace_dir" "$TRACE_DIR"
printf '  %-18s %s\n' "trace_summary" "$TRACE_SUMMARY_PATH"
printf '  %-18s %s\n' "eval_log" "$EVAL_LOG"
