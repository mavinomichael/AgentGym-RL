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
import os
import shutil
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
    allow_patterns=[
        'train/*', 'train/**/*',
        'eval/*', 'eval/**/*',
        'AgentItemId/*', 'AgentItemId/**/*',
        'AgentEval/*', 'AgentEval/**/*',
    ],
)

# Current Hugging Face dataset layout uses top-level train/eval folders, while
# the training scripts in this project still expect AgentItemId plus nested
# AgentEval/<task>/<task>_test.json directories.
for src_name, dst_name in [('train', 'AgentItemId/train'), ('eval', 'AgentEval')]:
    src = os.path.join('${DATA_ROOT}', src_name)
    dst = os.path.join('${DATA_ROOT}', dst_name)
    if not os.path.isdir(src):
        continue
    os.makedirs(dst, exist_ok=True)
    for entry in os.listdir(src):
        src_path = os.path.join(src, entry)
        dst_path = os.path.join(dst, entry)
        if os.path.isdir(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)

# Mirror flat eval files such as AgentEval/babyai_test.json into the nested
# task-specific directories consumed by the multi-agent generation config.
agent_eval_root = os.path.join('${DATA_ROOT}', 'AgentEval')
if os.path.isdir(agent_eval_root):
    for entry in os.listdir(agent_eval_root):
        if not entry.endswith('_test.json'):
            continue
        task_name = entry[:-10]  # strip "_test.json"
        nested_dir = os.path.join(agent_eval_root, task_name)
        os.makedirs(nested_dir, exist_ok=True)
        shutil.copy2(
            os.path.join(agent_eval_root, entry),
            os.path.join(nested_dir, entry),
        )
PY

touch "$HOME/.agentgym_training_env_ready"
