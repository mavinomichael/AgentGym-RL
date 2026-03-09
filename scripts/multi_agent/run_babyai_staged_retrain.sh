#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

TASK_NAME="babyai"
TRAIN_ROOT=$(multi_agent::train_root)

export MODEL_PATH="${MODEL_PATH:-/home/mavinomichael/AgentGym-RL/models/Qwen2.5-7B-Instruct}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$MODEL_PATH}"
export DATA_ROOT="${DATA_ROOT:-$(dirname "$TRAIN_ROOT")}"
export ENV_SERVER_URL="${ENV_SERVER_URL:-http://127.0.0.1:36005}"
export SAVE_ROOT="${SAVE_ROOT:-/mnt/data/saves}"
export LOG_ROOT="${LOG_ROOT:-/mnt/data/logs}"
export EXP_NAME="${EXP_NAME:-babyai_multi_retrain_v2}"
export N_GPUS="${N_GPUS:-8}"
export NNODES="${NNODES:-1}"
export TP_SIZE="${TP_SIZE:-1}"
export TMPDIR="${TMPDIR:-/mnt/data/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-/mnt/data/raytmp}"
export EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.72}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export TMUX_SESSION="${TMUX_SESSION:-train_babyai_staged}"

RUN_DIR="$SAVE_ROOT/agentgym_multi_agent/$EXP_NAME"
LOG_DIR="$LOG_ROOT/$EXP_NAME"
METRICS_FILE="$LOG_DIR/stage_metrics.tsv"

mkdir -p "$LOG_DIR" "$SAVE_ROOT" "$TMPDIR" "$RAY_TMPDIR"
if [[ -d "$RUN_DIR" ]] && [[ -n "$(ls -A "$RUN_DIR" 2>/dev/null || true)" ]]; then
  echo "Run directory already exists and is non-empty: $RUN_DIR"
  echo "Set a fresh EXP_NAME for a clean run."
  exit 1
fi

multi_agent::print_run_header "staged-train" "$TASK_NAME"
echo "  BASE_MODEL_PATH    $BASE_MODEL_PATH"
echo "  LOG_ROOT           $LOG_ROOT"
echo "  TMUX_SESSION       $TMUX_SESSION"

cat > "$METRICS_FILE" <<'EOF'
stage	avg_at_1	pass_at_1	executor_native_format_violations	invalid_format_termination_rate	invalid_action_termination_rate	trace_dir	eval_log	model_path
EOF

run_train_stage() {
  local stage="$1"
  local total_steps="$2"
  local resume_mode="$3"
  local train_log="$LOG_DIR/train_stage${stage}.log"

  echo "[stage $stage] training start (total_training_steps=$total_steps resume_mode=$resume_mode)"
  (
    cd "$TRAIN_ROOT"
    HYDRA_FULL_ERROR=1 \
    PYTHONUNBUFFERED=1 \
    TMPDIR="$TMPDIR" \
    RAY_TMPDIR="$RAY_TMPDIR" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    MODEL_PATH="$BASE_MODEL_PATH" \
    DATA_ROOT="$DATA_ROOT" \
    ENV_SERVER_URL="$ENV_SERVER_URL" \
    SAVE_ROOT="$SAVE_ROOT" \
    EXP_NAME="$EXP_NAME" \
    N_GPUS="$N_GPUS" \
    TP_SIZE="$TP_SIZE" \
    python3 -m verl.multi_agent.main_ppo \
      task=$TASK_NAME \
      task.train_file=AgentItemId/train/babyai_train.json \
      runtime=qwen2_5_7b_8gpu \
      runtime.nnodes=$NNODES \
      runtime.n_gpus_per_node=$N_GPUS \
      runtime.tensor_model_parallel_size=$TP_SIZE \
      algo=multi_agent_gae \
      trainer.default_local_dir="$RUN_DIR" \
      trainer.experiment_name="$EXP_NAME" \
      trainer.nnodes=$NNODES \
      trainer.n_gpus_per_node=$N_GPUS \
      trainer.total_training_steps="$total_steps" \
      trainer.resume_mode="$resume_mode" \
      trainer.save_freq=25 \
      multi_agent.invalid_output.policy=terminate_with_penalty \
      multi_agent.invalid_output.penalty=-0.2 \
      multi_agent.invalid_output.max_retries=2 \
      multi_agent.invalid_output.retry_temperature=0.2 \
      multi_agent.invalid_output.retry_max_tokens=80
  ) 2>&1 | tee "$train_log"
}

merge_stage_checkpoint() {
  local stage="$1"
  local checkpoint_actor_dir="$RUN_DIR/global_step_${stage}/actor"
  local merge_log="$LOG_DIR/merge_stage${stage}.log"

  if [[ ! -d "$checkpoint_actor_dir" ]]; then
    echo "Missing checkpoint actor directory: $checkpoint_actor_dir"
    exit 1
  fi

  echo "[stage $stage] merging actor checkpoint at $checkpoint_actor_dir"
  (
    cd "$TRAIN_ROOT"
    python3 scripts/model_merger.py --local_dir "$checkpoint_actor_dir"
  ) 2>&1 | tee "$merge_log"

  local merged_model_path="$checkpoint_actor_dir/huggingface"
  if [[ ! -d "$merged_model_path" ]]; then
    echo "Missing merged model path: $merged_model_path"
    exit 1
  fi
  MERGED_MODEL_PATH="$merged_model_path"
}

run_eval_stage() {
  local stage="$1"
  local model_path="$2"

  EVAL_LOG="$LOG_DIR/eval_stage${stage}.log"
  TRACE_DIR="$LOG_DIR/trace_stage${stage}"
  mkdir -p "$TRACE_DIR"

  echo "[stage $stage] eval start (model_path=$model_path)"
  (
    cd "$TRAIN_ROOT"
    HYDRA_FULL_ERROR=1 \
    PYTHONUNBUFFERED=1 \
    TMPDIR="$TMPDIR" \
    RAY_TMPDIR="$RAY_TMPDIR" \
    MODEL_PATH="$model_path" \
    DATA_ROOT="$DATA_ROOT" \
    ENV_SERVER_URL="$ENV_SERVER_URL" \
    SAVE_ROOT="$SAVE_ROOT" \
    EXP_NAME="${EXP_NAME}_eval_stage${stage}" \
    N_GPUS=1 \
    TP_SIZE=1 \
    python3 -m verl.multi_agent.main_generation \
      task=$TASK_NAME \
      runtime=qwen2_5_7b_1gpu_smoke \
      runtime.nnodes=1 \
      runtime.n_gpus_per_node=1 \
      runtime.tensor_model_parallel_size=1 \
      runtime.eval_gpu_memory_utilization="$EVAL_GPU_MEMORY_UTILIZATION" \
      runtime.model_path="$model_path" \
      runtime.env_server_url="$ENV_SERVER_URL" \
      rollout.gpu_memory_utilization="$EVAL_GPU_MEMORY_UTILIZATION" \
      rollout.max_num_batched_tokens=2048 \
      rollout.max_num_seqs=64 \
      multi_agent.invalid_output.max_retries=2 \
      multi_agent.invalid_output.retry_temperature=0.2 \
      multi_agent.invalid_output.retry_max_tokens=80 \
      multi_agent.debug.trace_executor_payload=true \
      multi_agent.debug.trace_dir="$TRACE_DIR" \
      multi_agent.debug.trace_max_chars=800
  ) 2>&1 | tee "$EVAL_LOG"
}

parse_eval_metrics() {
  local eval_log="$1"
  python3 - "$eval_log" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8", errors="ignore").read()

def first_metric(name: str) -> float:
    match = re.search(rf"{re.escape(name)}:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)
    if match is None:
        return float("nan")
    return float(match.group(1))

values = [
    first_metric("Avg@1"),
    first_metric("Pass@1"),
    first_metric("ExecutorNativeFormatViolations"),
    first_metric("InvalidFormatTerminationRate"),
    first_metric("InvalidActionTerminationRate"),
]

print(" ".join("nan" if value != value else f"{value:.15g}" for value in values))
PY
}

check_gate_stage_100() {
  local avg="$1"
  local invalid_format_rate="$2"
  python3 - "$avg" "$invalid_format_rate" <<'PY'
import sys

avg = float(sys.argv[1])
invalid = float(sys.argv[2])
ok = avg > 0.6728270654877027 and invalid <= 0.02
print(f"[gate stage100] pass={ok} avg={avg} invalid_format={invalid}")
raise SystemExit(0 if ok else 1)
PY
}

check_gate_stage_500() {
  local avg="$1"
  local invalid_format_rate="$2"
  local previous_avg="$3"
  python3 - "$avg" "$invalid_format_rate" "$previous_avg" <<'PY'
import sys

avg = float(sys.argv[1])
invalid = float(sys.argv[2])
previous_avg = float(sys.argv[3])
ok = avg > previous_avg and invalid <= 0.02
print(f"[gate stage500] pass={ok} avg={avg} previous_avg={previous_avg} invalid_format={invalid}")
raise SystemExit(0 if ok else 1)
PY
}

declare -A TOTAL_STEPS_BY_STAGE=(
  ["100"]=101
  ["500"]=501
  ["700"]=701
)

stage_100_avg=""

for stage in 100 500 700; do
  resume_mode="auto"
  if [[ "$stage" == "100" ]]; then
    resume_mode="disable"
  fi

  run_train_stage "$stage" "${TOTAL_STEPS_BY_STAGE[$stage]}" "$resume_mode"
  merge_stage_checkpoint "$stage"
  stage_model_path="$MERGED_MODEL_PATH"
  run_eval_stage "$stage" "$stage_model_path"
  read -r avg_at_1 pass_at_1 native_format_viol invalid_format_rate invalid_action_rate <<< "$(parse_eval_metrics "$EVAL_LOG")"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$stage" \
    "$avg_at_1" \
    "$pass_at_1" \
    "$native_format_viol" \
    "$invalid_format_rate" \
    "$invalid_action_rate" \
    "$TRACE_DIR" \
    "$EVAL_LOG" \
    "$stage_model_path" >> "$METRICS_FILE"

  echo "[stage $stage] Avg@1=$avg_at_1 Pass@1=$pass_at_1 ExecutorNativeFormatViolations=$native_format_viol InvalidFormatTerminationRate=$invalid_format_rate InvalidActionTerminationRate=$invalid_action_rate"

  if [[ "$stage" == "100" ]]; then
    check_gate_stage_100 "$avg_at_1" "$invalid_format_rate"
    stage_100_avg="$avg_at_1"
  elif [[ "$stage" == "500" ]]; then
    check_gate_stage_500 "$avg_at_1" "$invalid_format_rate" "$stage_100_avg"
  fi
done

echo "Staged retrain completed successfully."
echo "Metrics report: $METRICS_FILE"
