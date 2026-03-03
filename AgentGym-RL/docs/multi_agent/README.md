# Multi-Agent Pull-To-Run Guide

This runbook documents the isolated planner-executor extension under `AgentGym-RL/verl/multi_agent`.

## Supported environments
- `webarena`
- `sciworld`
- `searchqa`
- `babyai`
- `textcraft`

## Primary training target
- model family: `Qwen2.5-Instruct`
- primary training profile: `8 GPUs`
- first validation target: `webarena`

## Required environment variables
- `MODEL_PATH`: local merged Hugging Face model directory used for training or evaluation.
- `DATA_ROOT`: directory containing `AgentItemId/` and `AgentEval/`.
- `ENV_SERVER_URL`: environment server base URL, default `http://127.0.0.1:36005`.
- `SAVE_ROOT`: output directory for checkpoints and rollout logs.
- `EXP_NAME`: run name.
- `N_GPUS`: GPU count.
- `TP_SIZE`: tensor parallel size.

## Pull-to-run workflow
1. Create the training environment.
2. Run `bash scripts/multi_agent/bootstrap_training_env.sh`.
3. Set `MODEL_PATH`, `DATA_ROOT`, and `ENV_SERVER_URL`.
4. Set up and launch the environment server for the selected task.
5. Run `python3 scripts/multi_agent/preflight.py --task webarena --mode train --gpus 8`.
6. Run `bash examples/train/MultiAgent/webarena_train.sh`.
7. Merge the checkpoint if needed with `python3 AgentGym-RL/scripts/model_merger.py --local_dir <actor-checkpoint-dir>`.
8. Run `python3 scripts/multi_agent/preflight.py --task webarena --mode eval --gpus 1`.
9. Run `bash examples/eval/MultiAgent/webarena_eval.sh`.

## WebArena notes
- Website containers are external prerequisites.
- Copy `AgentGym/agentenv-webarena/.env.example` to `.env` and replace the URLs with your deployment.
- `OPENAI_API_KEY` must be set for judge-backed evaluations.
