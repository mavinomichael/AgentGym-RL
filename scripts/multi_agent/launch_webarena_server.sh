#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

REPO_ROOT=$(multi_agent::repo_root)
SERVER_ROOT="$REPO_ROOT/AgentGym/agentenv-webarena"
WEB_ARENA_ENV_FILE="${WEB_ARENA_ENV_FILE:-$SERVER_ROOT/.env}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-36005}"

multi_agent::source_env_file "$WEB_ARENA_ENV_FILE"
for required_var in SHOPPING SHOPPING_ADMIN REDDIT GITLAB MAP WIKIPEDIA HOMEPAGE OPENAI_API_KEY; do
  multi_agent::require_var "$required_var"
done
multi_agent::print_server_header "launch" "webarena" "$HOST" "$PORT"
exec webarena --host "$HOST" --port "$PORT"
