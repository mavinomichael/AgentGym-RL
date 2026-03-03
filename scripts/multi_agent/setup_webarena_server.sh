#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

REPO_ROOT=$(multi_agent::repo_root)
SERVER_ROOT="$REPO_ROOT/AgentGym/agentenv-webarena"
WEB_ARENA_ENV_FILE="${WEB_ARENA_ENV_FILE:-$SERVER_ROOT/.env}"

multi_agent::source_env_file "$WEB_ARENA_ENV_FILE"
python3 -m pip install -e "$REPO_ROOT/AgentGym/agentenv"
(
  cd "$SERVER_ROOT"
  bash ./setup.sh
)
