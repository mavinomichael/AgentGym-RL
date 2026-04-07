#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-36005}"
multi_agent::print_server_header "launch" "sciworld" "$HOST" "$PORT"
exec sciworld --host "$HOST" --port "$PORT"
