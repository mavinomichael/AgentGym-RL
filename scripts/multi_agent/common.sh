#!/usr/bin/env bash

multi_agent::repo_root() {
  local script_dir
  script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  cd "${script_dir}/../.." >/dev/null 2>&1 && pwd
}

multi_agent::train_root() {
  printf '%s\n' "$(multi_agent::repo_root)/AgentGym-RL"
}

multi_agent::require_var() {
  local var_name="$1"
  if [ -z "${!var_name:-}" ]; then
    printf 'Missing required environment variable: %s\n' "$var_name" >&2
    exit 1
  fi
}

multi_agent::source_env_file() {
  local env_file="${1:-}"
  if [ -n "$env_file" ] && [ -f "$env_file" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$env_file"
    set +a
  fi
}

multi_agent::print_run_header() {
  local mode="$1"
  local task_name="$2"
  printf 'Multi-agent %s configuration\n' "$mode"
  printf '  %-18s %s\n' "task" "$task_name"
  printf '  %-18s %s\n' "MODEL_PATH" "${MODEL_PATH:-}"
  printf '  %-18s %s\n' "DATA_ROOT" "${DATA_ROOT:-}"
  printf '  %-18s %s\n' "ENV_SERVER_URL" "${ENV_SERVER_URL:-}"
  printf '  %-18s %s\n' "SAVE_ROOT" "${SAVE_ROOT:-}"
  printf '  %-18s %s\n' "EXP_NAME" "${EXP_NAME:-}"
  printf '  %-18s %s\n' "N_GPUS" "${N_GPUS:-}"
  printf '  %-18s %s\n' "TP_SIZE" "${TP_SIZE:-}"
  if [ -n "${CHECKPOINT_DIR:-}" ]; then
    printf '  %-18s %s\n' "CHECKPOINT_DIR" "$CHECKPOINT_DIR"
  fi
}

multi_agent::maybe_merge_checkpoint() {
  local train_root="$1"
  if [ -n "${CHECKPOINT_DIR:-}" ]; then
    python3 "$train_root/scripts/model_merger.py" --local_dir "$CHECKPOINT_DIR"
    if [ -z "${MODEL_PATH:-}" ]; then
      export MODEL_PATH="$CHECKPOINT_DIR/huggingface"
    fi
  fi
}

multi_agent::print_server_header() {
  local action="$1"
  local task_name="$2"
  local host="$3"
  local port="$4"
  printf 'Multi-agent server %s\n' "$action"
  printf '  %-18s %s\n' "task" "$task_name"
  printf '  %-18s %s\n' "host" "$host"
  printf '  %-18s %s\n' "port" "$port"
}
