#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

export EXP_NAME="${EXP_NAME:-webarena_2agent_fixed15_8gpu_tagged_v1}"
export PROMPT_STYLE="${PROMPT_STYLE:-tagged}"
export ROUNDS_CTRL_TYPE="${ROUNDS_CTRL_TYPE:-fixed}"
export ROUNDS_CTRL_ROUNDS="${ROUNDS_CTRL_ROUNDS:-15}"
export ROUNDS_CTRL_STEPS="${ROUNDS_CTRL_STEPS:-80}"

exec "$SCRIPT_DIR/run_webarena_2agent_8gpu.sh"
