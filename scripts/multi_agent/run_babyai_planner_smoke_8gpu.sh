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
export EXP_NAME="${EXP_NAME:-babyai_planner_stabilization_smoke_8gpu}"
export N_GPUS="${N_GPUS:-8}"
export NNODES="${NNODES:-1}"
export TP_SIZE="${TP_SIZE:-1}"
export TMPDIR="${TMPDIR:-/mnt/data/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-/mnt/data/raytmp}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-agentgym-rl}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"

RUN_DIR="$SAVE_ROOT/agentgym_multi_agent/$EXP_NAME"
LOG_DIR="$LOG_ROOT/$EXP_NAME"
TRACE_DIR="$LOG_DIR/trace_train"
REPORT_DIR="$REPO_ROOT/reports/babyai_multi_agent_diagnostics_2026-03-09"
REPORT_PATH="$REPORT_DIR/planner_stabilization_step5_messages.txt"
TRAIN_LOG="$LOG_DIR/train_step5_smoke.log"
MERGE_LOG="$LOG_DIR/merge_step5_smoke.log"

mkdir -p "$LOG_DIR" "$TRACE_DIR" "$TMPDIR" "$RAY_TMPDIR" "$REPORT_DIR"

if [[ -f /home/mavinomichael/miniconda/etc/profile.d/conda.sh ]]; then
  # Run under the same conda environment used for training/eval on the VM.
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

merge_checkpoint() {
  local checkpoint_actor_dir="$RUN_DIR/global_step_5/actor"
  if [[ ! -d "$checkpoint_actor_dir" ]]; then
    printf 'Missing checkpoint actor directory: %s\n' "$checkpoint_actor_dir" >&2
    exit 1
  fi

  (
    cd "$TRAIN_ROOT"
    "$PYTHON_BIN" scripts/model_merger.py --local_dir "$checkpoint_actor_dir"
  ) 2>&1 | tee "$MERGE_LOG"
}

generate_trace_report() {
  "$PYTHON_BIN" - "$TRACE_DIR" "$REPORT_PATH" <<'PY'
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

events.sort(key=lambda item: (item.get("rank", 0), item.get("item_id", 0), item.get("round", 0)))

planner_valid = sum(1 for item in events if item.get("planner_validation_valid"))
planner_fallback = sum(1 for item in events if item.get("planner_fallback_used"))
planner_tag_only = sum(1 for item in events if item.get("planner_validation_reason") == "tag_only")

with open(report_path, "w", encoding="utf-8") as report:
    report.write("BabyAI Planner Stabilization Step-5 Smoke Trace\n")
    report.write(f"Trace dir: {trace_dir}\n")
    report.write(f"Total events: {len(events)}\n")
    report.write(f"Planner valid events: {planner_valid}\n")
    report.write(f"Planner fallback events: {planner_fallback}\n")
    report.write(f"Planner tag-only events: {planner_tag_only}\n\n")

    for idx, event in enumerate(events, start=1):
        report.write("=" * 120 + "\n")
        report.write(
            f"event_idx: {idx} | rank: {event.get('rank')} | item_id: {event.get('item_id')} | round: {event.get('round')}\n"
        )
        report.write(
            "planner_validation_valid: "
            f"{event.get('planner_validation_valid')} | planner_validation_reason: {event.get('planner_validation_reason')} "
            f"| planner_fallback_used: {event.get('planner_fallback_used')}\n"
        )
        report.write(
            "executor_validation_valid: "
            f"{event.get('validation_valid')} | executor_validation_reason: {event.get('validation_reason')} "
            f"| extracted_action: {event.get('extracted_action')}\n"
        )
        report.write(
            f"env_step_called: {event.get('env_step_called')} | env_reward: {event.get('env_reward')} | env_done: {event.get('env_done')}\n\n"
        )
        report.write("[OBSERVATION]\n")
        report.write(f"{event.get('observation_excerpt', '')}\n\n")
        report.write("[PLANNER_PROMPT]\n")
        report.write(f"{event.get('planner_prompt', '')}\n\n")
        report.write("[PLANNER_RAW_OUTPUT]\n")
        report.write(f"{event.get('planner_raw_output', '')}\n\n")
        report.write("[PLANNER_NORMALIZED_OUTPUT]\n")
        report.write(f"{event.get('planner_normalized_output', '')}\n\n")
        report.write("[PLANNER_CONTEXT_USED_BY_EXECUTOR]\n")
        report.write(f"{event.get('planner_context_used_by_executor', '')}\n\n")
        report.write("[EXECUTOR_PROMPT]\n")
        report.write(f"{event.get('executor_prompt', '')}\n\n")
        report.write("[EXECUTOR_RAW_OUTPUT]\n")
        report.write(f"{event.get('executor_raw_output', '')}\n\n")
        report.write("[EXECUTOR_NORMALIZED_OUTPUT]\n")
        report.write(f"{event.get('executor_normalized_output', '')}\n\n")
        report.write("[ENV_STEP_PAYLOAD]\n")
        report.write(f"{event.get('env_step_payload', '')}\n\n")
        report.write("[ENV_STATE]\n")
        report.write(f"{event.get('env_state_excerpt', '')}\n\n")

print(report_path)
PY
}

multi_agent::print_run_header "planner-smoke-8gpu" "$TASK_NAME"
printf '  %-18s %s\n' "TRACE_DIR" "$TRACE_DIR"
printf '  %-18s %s\n' "REPORT_PATH" "$REPORT_PATH"

cleanup_old_ray_sessions

(
  cd "$TRAIN_ROOT"
  HYDRA_FULL_ERROR=1 \
  PYTHONUNBUFFERED=1 \
  TMPDIR="$TMPDIR" \
  RAY_TMPDIR="$RAY_TMPDIR" \
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
    trainer.total_training_steps=6 \
    trainer.resume_mode=disable \
    trainer.save_freq=5 \
    multi_agent.invalid_output.policy=terminate_with_penalty \
    multi_agent.invalid_output.penalty=-0.2 \
    multi_agent.invalid_output.max_retries=2 \
    multi_agent.invalid_output.retry_temperature=0.2 \
    multi_agent.invalid_output.retry_max_tokens=80 \
    multi_agent.debug.trace_executor_payload=true \
    multi_agent.debug.trace_dir="$TRACE_DIR" \
    multi_agent.debug.trace_max_chars=800
) 2>&1 | tee "$TRAIN_LOG"

merge_checkpoint
generate_trace_report

printf 'Smoke run complete.\n'
printf '  %-18s %s\n' "train_log" "$TRAIN_LOG"
printf '  %-18s %s\n' "trace_dir" "$TRACE_DIR"
printf '  %-18s %s\n' "report_path" "$REPORT_PATH"
