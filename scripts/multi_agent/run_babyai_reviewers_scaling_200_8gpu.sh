#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

export EXP_NAME="${EXP_NAME:-babyai_reviewers_scaling_200_8gpu}"
export ROUNDS_CTRL_TYPE="${ROUNDS_CTRL_TYPE:-scaling_inter_stepwise}"
export ROUNDS_CTRL_ROUNDS="${ROUNDS_CTRL_ROUNDS:-[6,13,20]}"
export ROUNDS_CTRL_STEPS="${ROUNDS_CTRL_STEPS:-100}"

exec "$SCRIPT_DIR/run_babyai_reviewers_200_8gpu.sh"
