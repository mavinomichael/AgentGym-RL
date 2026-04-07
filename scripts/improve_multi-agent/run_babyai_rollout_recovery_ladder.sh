#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

export SAVE_ROOT="${SAVE_ROOT:-/mnt/data/saves}"
export ENV_SERVER_URL="${ENV_SERVER_URL:-http://127.0.0.1:36006}"
export HARNESS_ROOT="${HARNESS_ROOT:-$SAVE_ROOT/improve_multi_agent/rollout_harness}"
export EPISODE_COUNT="${EPISODE_COUNT:-3}"
export START_ITEM_ID="${START_ITEM_ID:-0}"
export MAX_STEPS="${MAX_STEPS:-6}"
export PLANNER_INTERVAL="${PLANNER_INTERVAL:-3}"
export PLANNER_CHECKPOINT="${PLANNER_CHECKPOINT:-$SAVE_ROOT/improve_multi_agent/improve_babyai_planner_warmstart_v1/global_step_200}"
export EXECUTOR_CHECKPOINT="${EXECUTOR_CHECKPOINT:-$SAVE_ROOT/improve_multi_agent/improve_babyai_executor_warmstart_v1/global_step_200}"

run_rung() {
  local name="$1"
  local planner_mode="$2"
  local executor_mode="$3"
  echo "[rollout-ladder] running rung: $name"
  HARNESS_ROOT="$HARNESS_ROOT/$name" \
  PLANNER_MODE="$planner_mode" \
  EXECUTOR_MODE="$executor_mode" \
  EPISODE_COUNT="$EPISODE_COUNT" \
  START_ITEM_ID="$START_ITEM_ID" \
  MAX_STEPS="$MAX_STEPS" \
  PLANNER_INTERVAL="$PLANNER_INTERVAL" \
  PLANNER_CHECKPOINT="$PLANNER_CHECKPOINT" \
  EXECUTOR_CHECKPOINT="$EXECUTOR_CHECKPOINT" \
  bash "$SCRIPT_DIR/run_babyai_rollout_harness.sh"
}

run_rung "scripted_planner_scripted_executor" "scripted" "scripted"
run_rung "scripted_planner_learned_executor" "scripted" "model"
run_rung "learned_planner_frozen_executor" "model" "model"

echo "[rollout-ladder] all rungs passed"
