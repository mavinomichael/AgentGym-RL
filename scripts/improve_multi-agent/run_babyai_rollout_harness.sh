#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

export SAVE_ROOT="${SAVE_ROOT:-/mnt/data/saves}"
export ENV_SERVER_URL="${ENV_SERVER_URL:-http://127.0.0.1:36006}"
export HARNESS_ROOT="${HARNESS_ROOT:-$SAVE_ROOT/improve_multi_agent/rollout_harness/manual}"
export EPISODE_COUNT="${EPISODE_COUNT:-1}"
export START_ITEM_ID="${START_ITEM_ID:-0}"
export MAX_STEPS="${MAX_STEPS:-6}"
export PLANNER_INTERVAL="${PLANNER_INTERVAL:-3}"
export ENV_TIMEOUT="${ENV_TIMEOUT:-20}"
export RESET_TIMEOUT="${RESET_TIMEOUT:-20}"
export OBSERVE_TIMEOUT="${OBSERVE_TIMEOUT:-15}"
export AVAILABLE_ACTIONS_TIMEOUT="${AVAILABLE_ACTIONS_TIMEOUT:-2}"
export PLANNER_TIMEOUT="${PLANNER_TIMEOUT:-45}"
export EXECUTOR_TIMEOUT="${EXECUTOR_TIMEOUT:-45}"
export ENV_STEP_TIMEOUT="${ENV_STEP_TIMEOUT:-30}"
export PLANNER_MODE="${PLANNER_MODE:-scripted}"
export EXECUTOR_MODE="${EXECUTOR_MODE:-scripted}"
export PLANNER_CHECKPOINT="${PLANNER_CHECKPOINT:-$SAVE_ROOT/improve_multi_agent/improve_babyai_planner_warmstart_v1/global_step_200}"
export EXECUTOR_CHECKPOINT="${EXECUTOR_CHECKPOINT:-$SAVE_ROOT/improve_multi_agent/improve_babyai_executor_warmstart_v1/global_step_200}"

mkdir -p "$HARNESS_ROOT"

python3 "$SCRIPT_DIR/run_babyai_rollout_harness.py" \
  --env-server-url "$ENV_SERVER_URL" \
  --output-dir "$HARNESS_ROOT" \
  --summary-path "$HARNESS_ROOT/summary.json" \
  --episode-count "$EPISODE_COUNT" \
  --start-item-id "$START_ITEM_ID" \
  --max-steps "$MAX_STEPS" \
  --planner-interval "$PLANNER_INTERVAL" \
  --env-timeout "$ENV_TIMEOUT" \
  --reset-timeout "$RESET_TIMEOUT" \
  --observe-timeout "$OBSERVE_TIMEOUT" \
  --available-actions-timeout "$AVAILABLE_ACTIONS_TIMEOUT" \
  --planner-timeout "$PLANNER_TIMEOUT" \
  --executor-timeout "$EXECUTOR_TIMEOUT" \
  --env-step-timeout "$ENV_STEP_TIMEOUT" \
  --planner-mode "$PLANNER_MODE" \
  --executor-mode "$EXECUTOR_MODE" \
  --planner-checkpoint "$PLANNER_CHECKPOINT" \
  --executor-checkpoint "$EXECUTOR_CHECKPOINT"
