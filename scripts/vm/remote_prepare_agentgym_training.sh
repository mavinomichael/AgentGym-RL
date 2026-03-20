#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/AgentGym-RL}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-agentgym-rl}"
MODEL_HUB_ID="${MODEL_HUB_ID:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/models/Qwen2.5-7B-Instruct}"
DATASET_HUB_ID="${DATASET_HUB_ID:-AgentGym/AgentGym-RL-Data-ID}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT}"
FLASH_ATTENTION_URL="${FLASH_ATTENTION_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"
FLASH_ATTENTION_WHL="${FLASH_ATTENTION_WHL:-/tmp/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"

mkdir -p "$REPO_ROOT/models" "$DATA_ROOT"

if [[ ! -x "$CONDA_DIR/bin/conda" ]]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
  rm -f /tmp/miniconda.sh
fi

# shellcheck source=/dev/null
source "$CONDA_DIR/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
  conda create -n "$CONDA_ENV_NAME" python=3.10 -y
fi
conda activate "$CONDA_ENV_NAME"

python -m pip install --upgrade pip
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install transformers==4.51.3 huggingface_hub

if ! python - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec('flash_attn') else 1)
PY
then
  wget -q "$FLASH_ATTENTION_URL" -O "$FLASH_ATTENTION_WHL"
  python -m pip install "$FLASH_ATTENTION_WHL"
  rm -f "$FLASH_ATTENTION_WHL"
fi

python -m pip install -e "$REPO_ROOT/AgentGym-RL"
python -m pip install -e "$REPO_ROOT/AgentGym/agentenv"

python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${MODEL_HUB_ID}',
    local_dir='${MODEL_PATH}',
    local_dir_use_symlinks=False,
)
snapshot_download(
    repo_id='${DATASET_HUB_ID}',
    repo_type='dataset',
    local_dir='${DATA_ROOT}',
    local_dir_use_symlinks=False,
    allow_patterns=['AgentItemId/*', 'AgentItemId/**/*', 'AgentEval/*', 'AgentEval/**/*'],
)
PY

touch "$HOME/.agentgym_training_env_ready"
