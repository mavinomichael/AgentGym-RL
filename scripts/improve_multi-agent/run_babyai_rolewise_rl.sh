#!/usr/bin/env bash
set -euo pipefail

export EXP_NAME="${EXP_NAME:-improve_babyai_rolewise_rl_v1}"
export ROLE_LOCAL_OPTIMIZATION=true
export ROLE_LOCAL_ADVANTAGE=true
export ROLE_AUX_REWARDS=true
export MILESTONE_REWARDS=false

bash "$(cd "$(dirname "$0")" && pwd)/run_babyai_joint_rl.sh"
