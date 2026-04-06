#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

REPO_ROOT=$(multi_agent::repo_root)
SERVER_ROOT="$REPO_ROOT/AgentGym/agentenv-webarena"
WEB_ARENA_ENV_FILE="${WEB_ARENA_ENV_FILE:-$SERVER_ROOT/.env}"

multi_agent::source_env_file "$WEB_ARENA_ENV_FILE"
multi_agent::set_webarena_default_env
if [[ ! -f "$WEB_ARENA_ENV_FILE" ]]; then
  multi_agent::write_webarena_env_file "$WEB_ARENA_ENV_FILE"
fi

python3 -m pip install "$REPO_ROOT/AgentGym/agentenv"
python3 -m pip install -r "$SERVER_ROOT/webarena/requirements.txt"
python3 -m playwright install-deps
python3 -m playwright install
python3 -m pip install -e "$SERVER_ROOT/webarena"
python3 -m pip install gunicorn
(
  cd "$SERVER_ROOT/webarena"
  python3 scripts/generate_test_data.py
  mkdir -p ./.auth
  if [[ "${WEB_ARENA_SKIP_AUTO_LOGIN:-0}" == "1" ]]; then
    echo "Skipping WebArena auto_login because WEB_ARENA_SKIP_AUTO_LOGIN=1"
  else
    python3 browser_env/auto_login.py
  fi
  python3 agent/prompts/to_json.py
)
python3 -m pip install -e "$SERVER_ROOT"
