#!/usr/bin/env bash
set -euo pipefail

export EXP_NAME="${EXP_NAME:-improve_babyai_protocol_only_v1}"
export ROLE_LOCAL_OPTIMIZATION=false
export ROLE_LOCAL_ADVANTAGE=false
export ROLE_AUX_REWARDS=false
export MILESTONE_REWARDS=false

bash "$(cd "$(dirname "$0")" && pwd)/run_babyai_joint_rl.sh"
