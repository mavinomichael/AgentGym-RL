#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

REPO_ROOT=$(multi_agent::repo_root)
TRAIN_ROOT=$(multi_agent::train_root)
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
FLASH_ATTENTION_WHEEL_URL="${FLASH_ATTENTION_WHEEL_URL:-}"
FLASH_ATTENTION_WHEEL_PATH="${FLASH_ATTENTION_WHEEL_PATH:-}"

printf 'Bootstrapping training environment\n'
printf '  %-24s %s\n' "repo_root" "$REPO_ROOT"
printf '  %-24s %s\n' "train_root" "$TRAIN_ROOT"
printf '  %-24s %s\n' "INSTALL_FLASH_ATTN" "$INSTALL_FLASH_ATTN"

python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r "$TRAIN_ROOT/requirements.txt"
python3 -m pip install -e "$TRAIN_ROOT"
python3 -m pip install -e "$REPO_ROOT/AgentGym/agentenv"
python3 -m pip install transformers==4.51.3

if [ "$INSTALL_FLASH_ATTN" = "1" ]; then
  if [ -n "$FLASH_ATTENTION_WHEEL_PATH" ]; then
    python3 -m pip install "$FLASH_ATTENTION_WHEEL_PATH"
  elif [ -n "$FLASH_ATTENTION_WHEEL_URL" ]; then
    tmp_wheel=$(mktemp /tmp/flash_attn.XXXXXX.whl)
    curl -L "$FLASH_ATTENTION_WHEEL_URL" -o "$tmp_wheel"
    python3 -m pip install "$tmp_wheel"
    rm -f "$tmp_wheel"
  else
    printf 'INSTALL_FLASH_ATTN=1 but no FLASH_ATTENTION_WHEEL_PATH or FLASH_ATTENTION_WHEEL_URL provided.\n' >&2
    exit 1
  fi
fi

printf '\nTraining environment bootstrap complete.\n'
printf 'Next steps:\n'
printf '  1. Set MODEL_PATH and DATA_ROOT.\n'
printf '  2. Run scripts/multi_agent/setup_<task>_server.sh for the target environment.\n'
printf '  3. Run scripts/multi_agent/preflight.py before training or evaluation.\n'
