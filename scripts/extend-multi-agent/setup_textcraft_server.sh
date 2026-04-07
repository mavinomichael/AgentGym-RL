#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

REPO_ROOT=$(multi_agent::repo_root)
SERVER_ROOT="$REPO_ROOT/AgentGym/agentenv-textcraft"

python3 -m pip install -e "$REPO_ROOT/AgentGym/agentenv"
python3 -m pip install -e "$SERVER_ROOT"
