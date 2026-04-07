#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SAVE_ROOT="${SAVE_ROOT:-/mnt/data/saves}"
DATA_ROOT="${DATA_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
WARMSTART_EVAL_ROOT="${WARMSTART_EVAL_ROOT:-$SAVE_ROOT/improve_multi_agent/warmstart_eval}"
export ENV_SERVER_URL="${ENV_SERVER_URL:-http://127.0.0.1:36006}"

if [[ ! -f "$DATA_ROOT/improve_multi-agent/datasets/babyai_executor_warmstart_eval.json" || ! -f "$DATA_ROOT/improve_multi-agent/datasets/babyai_planner_warmstart_eval.json" ]]; then
  python3 "$SCRIPT_DIR/build_babyai_trace_datasets.py"
fi

select_eval_dataset() {
  local heldout_path="$1"
  local fallback_path="$2"
  local selected="$heldout_path"
  if [[ ! -f "$selected" ]]; then
    selected="$fallback_path"
  else
    local count
    count=$(python3 - "$selected" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
print(len(data))
PY
)
    if [[ "$count" == "0" ]]; then
      selected="$fallback_path"
    fi
  fi
  printf '%s\n' "$selected"
}

eval_report_is_usable() {
  local report_path="$1"
  local dataset_path="$2"
  if [[ ! -f "$report_path" ]]; then
    return 1
  fi
  python3 - "$report_path" "$dataset_path" <<'PY'
import json, sys
from pathlib import Path
report = Path(sys.argv[1])
dataset = sys.argv[2]
payload = json.loads(report.read_text(encoding="utf-8"))
metrics = payload.get("metrics", {})
total = int(metrics.get("total", 0))
same_dataset = payload.get("dataset") == dataset
raise SystemExit(0 if total > 0 and same_dataset else 1)
PY
}

run_executor_eval() {
  local ckpt_dir="$SAVE_ROOT/improve_multi_agent/improve_babyai_executor_warmstart_v1/global_step_200"
  local eval_dataset
  eval_dataset=$(select_eval_dataset \
    "$DATA_ROOT/improve_multi-agent/datasets/babyai_executor_warmstart_heldout.json" \
    "$DATA_ROOT/improve_multi-agent/datasets/babyai_executor_warmstart_eval.json")
  if [[ ! -d "$ckpt_dir" || ! -f "$eval_dataset" ]]; then
    return
  fi
  local report_dir="$WARMSTART_EVAL_ROOT/executor"
  if eval_report_is_usable "$report_dir/executor_warmstart_eval.json" "$eval_dataset"; then
    echo "[stage 1 eval] executor warm-start eval already exists"
    return
  fi
  python3 "$SCRIPT_DIR/eval_babyai_warmstart.py" \
    --stage executor \
    --checkpoint "$ckpt_dir" \
    --dataset "$eval_dataset" \
    --output-dir "$report_dir"
}

run_planner_eval() {
  local ckpt_dir="$SAVE_ROOT/improve_multi_agent/improve_babyai_planner_warmstart_v1/global_step_200"
  local eval_dataset
  eval_dataset=$(select_eval_dataset \
    "$DATA_ROOT/improve_multi-agent/datasets/babyai_planner_warmstart_heldout.json" \
    "$DATA_ROOT/improve_multi-agent/datasets/babyai_planner_warmstart_eval.json")
  if [[ ! -d "$ckpt_dir" || ! -f "$eval_dataset" ]]; then
    return
  fi
  local report_dir="$WARMSTART_EVAL_ROOT/planner"
  if eval_report_is_usable "$report_dir/planner_warmstart_eval.json" "$eval_dataset"; then
    echo "[stage 2 eval] planner warm-start eval already exists"
    return
  fi
  python3 "$SCRIPT_DIR/eval_babyai_warmstart.py" \
    --stage planner \
    --checkpoint "$ckpt_dir" \
    --dataset "$eval_dataset" \
    --output-dir "$report_dir"
}

echo "[stage 1] executor warm-start"
if [[ -d "$SAVE_ROOT/improve_multi_agent/improve_babyai_executor_warmstart_v1/global_step_200" ]]; then
  echo "[stage 1] skipping existing executor warm-start checkpoint"
else
  bash "$SCRIPT_DIR/run_babyai_executor_warmstart.sh"
fi
run_executor_eval
echo "[stage 2] planner warm-start"
if [[ -d "$SAVE_ROOT/improve_multi_agent/improve_babyai_planner_warmstart_v1/global_step_200" ]]; then
  echo "[stage 2] skipping existing planner warm-start checkpoint"
else
  bash "$SCRIPT_DIR/run_babyai_planner_warmstart.sh"
fi
run_planner_eval
echo "[rollout harness] recovery ladder before RL"
bash "$SCRIPT_DIR/run_babyai_rollout_recovery_ladder.sh"
echo "[stage 3] planner RL with frozen executor"
bash "$SCRIPT_DIR/run_babyai_planner_frozen_rl.sh"
echo "[stage 4] light joint RL with milestones"
bash "$SCRIPT_DIR/run_babyai_joint_rl.sh"
