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

export PATH="$HOME/.local/bin:$PATH"

multi_agent::source_env_file "$WEB_ARENA_ENV_FILE"
multi_agent::set_webarena_default_env
if [[ ! -f "$WEB_ARENA_ENV_FILE" ]]; then
  multi_agent::write_webarena_env_file "$WEB_ARENA_ENV_FILE"
fi
for required_var in SHOPPING SHOPPING_ADMIN REDDIT GITLAB MAP WIKIPEDIA HOMEPAGE; do
  multi_agent::require_var "$required_var"
done
multi_agent::print_server_header "launch" "webarena" "$HOST" "$PORT"
exec webarena --host "$HOST" --port "$PORT"
