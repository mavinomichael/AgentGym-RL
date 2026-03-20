#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/AgentGym-RL}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-agentgym-rl}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/models/Qwen2.5-7B-Instruct}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT}"
ENV_SESSION="${ENV_SESSION:-agentgym_env}"
TRAIN_SESSION="${TRAIN_SESSION:-agentgym_train}"
ENV_LOG="${ENV_LOG:-/tmp/agentgym_env.log}"
TRAIN_LOG="${TRAIN_LOG:-/tmp/agentgym_train.launch.log}"
TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-$REPO_ROOT/scripts/multi_agent/run_babyai_2agent_scaling_200_8gpu.sh}"

# shellcheck source=/dev/null
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

tmux has-session -t "$ENV_SESSION" 2>/dev/null && tmux kill-session -t "$ENV_SESSION"
tmux has-session -t "$TRAIN_SESSION" 2>/dev/null && tmux kill-session -t "$TRAIN_SESSION"

tmux new-session -d -s "$ENV_SESSION" "cd '$REPO_ROOT' && bash scripts/multi_agent/launch_babyai_server.sh > '$ENV_LOG' 2>&1"
sleep 10

tmux new-session -d -s "$TRAIN_SESSION" "cd '$REPO_ROOT' && export MODEL_PATH='$MODEL_PATH' DATA_ROOT='$DATA_ROOT' && bash '$TRAIN_LAUNCHER' > '$TRAIN_LOG' 2>&1"

touch "$HOME/.agentgym_training_started"
