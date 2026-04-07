#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

TASK_NAME="webarena"
TRAIN_ROOT=$(multi_agent::train_root)
REPO_ROOT=$(multi_agent::repo_root)

export MODEL_PATH="${MODEL_PATH:-/home/mavinomichael/AgentGym-RL/models/Qwen2.5-7B-Instruct}"
export DATA_ROOT="${DATA_ROOT:-$(dirname "$TRAIN_ROOT")}"
export ENV_SERVER_URL="${ENV_SERVER_URL:-http://127.0.0.1:36005}"
export WEB_ARENA_ENV_FILE="${WEB_ARENA_ENV_FILE:-$REPO_ROOT/AgentGym/agentenv-webarena/.env}"
export SAVE_ROOT="${SAVE_ROOT:-/home/mavinomichael/agentgym_runs/saves}"
export LOG_ROOT="${LOG_ROOT:-/home/mavinomichael/agentgym_runs/logs}"
export EXP_NAME="${EXP_NAME:-webarena_3agent_executor_reviewer_scaling_600_8gpu_no_tags_v1}"
export N_GPUS="${N_GPUS:-4}"
export NNODES="${NNODES:-1}"
export TP_SIZE="${TP_SIZE:-1}"
export TMPDIR="${TMPDIR:-/tmp/agentgym_tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/rayw}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-agentgym-rl}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export MULTI_AGENT_PROMPT_STYLE="${MULTI_AGENT_PROMPT_STYLE:-plain_split}"
export TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-0.2}"
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.2}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-601}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export EVAL_MAX_PROMPT_LENGTH="${EVAL_MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
export EVAL_MAX_RESPONSE_LENGTH="${EVAL_MAX_RESPONSE_LENGTH:-1024}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export SAVE_STEPS="${SAVE_STEPS:-[]}"
export PLANNER_MAX_TOKENS="${PLANNER_MAX_TOKENS:-128}"
export EXECUTOR_MAX_TOKENS="${EXECUTOR_MAX_TOKENS:-192}"
export EXECUTOR_REVIEWER_MAX_TOKENS="${EXECUTOR_REVIEWER_MAX_TOKENS:-48}"
export EXECUTOR_TEMPERATURE="${EXECUTOR_TEMPERATURE:-0.2}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export ROLLOUT_N="${ROLLOUT_N:-1}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.30}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-256}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export ROLLOUT_MAX_TOKENS="${ROLLOUT_MAX_TOKENS:-192}"
export ACTOR_MAX_TOKEN_LEN_PER_GPU="${ACTOR_MAX_TOKEN_LEN_PER_GPU:-3072}"
export CRITIC_MAX_TOKEN_LEN_PER_GPU="${CRITIC_MAX_TOKEN_LEN_PER_GPU:-3072}"
export ROUNDS_CTRL_TYPE="${ROUNDS_CTRL_TYPE:-scaling_inter_stepwise}"
export ROUNDS_CTRL_ROUNDS="${ROUNDS_CTRL_ROUNDS:-[4,6,8,10,12,15]}"
export ROUNDS_CTRL_STEPS="${ROUNDS_CTRL_STEPS:-100}"
export RESUME_MODE="${RESUME_MODE:-disable}"

RUN_DIR="$SAVE_ROOT/agentgym_multi_agent/$EXP_NAME"
LOG_DIR="$LOG_ROOT/$EXP_NAME"
TRACE_DIR="$LOG_DIR/trace_train"
mkdir -p "$RUN_DIR" "$LOG_DIR" "$TRACE_DIR" "$TMPDIR" "$RAY_TMPDIR" "$RAY_TMPDIR/spill"

if [[ -f /home/mavinomichael/miniconda/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  set +u
  source /home/mavinomichael/miniconda/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV_NAME"
  set -u
  PYTHON_BIN=$(command -v python)
fi

multi_agent::source_env_file "$WEB_ARENA_ENV_FILE"

cleanup_old_ray_sessions() {
  local -a roots=("/tmp/ray" "$RAY_TMPDIR" "$RAY_TMPDIR/ray")
  local active_sessions
  active_sessions=$(ps aux | sed -n 's#.*\(/[^ ]*/ray/session_[^ /]*\).*#\1#p' | sort -u)
  for ray_root in "${roots[@]}"; do
    [[ -d "$ray_root" ]] || continue
    for session_dir in "$ray_root"/session_*; do
      [[ -d "$session_dir" ]] || continue
      if [[ -n "$active_sessions" ]] && grep -Fxq "$session_dir" <<<"$active_sessions"; then
        continue
      fi
      rm -rf "$session_dir"
    done
  done
  rm -rf "$RAY_TMPDIR/spill"/*
}

cleanup_old_ray_sessions
multi_agent::print_run_header "webarena-3agent-executor-reviewer-scaling-600-8gpu" "$TASK_NAME"
printf '  %-22s %s\n' "RUN_DIR" "$RUN_DIR"
printf '  %-22s %s\n' "TRACE_DIR" "$TRACE_DIR"
printf '  %-22s %s\n' "TOTAL_STEPS" "$TOTAL_TRAINING_STEPS"
printf '  %-22s %s\n' "MAX_PROMPT_LENGTH" "$MAX_PROMPT_LENGTH"
printf '  %-22s %s\n' "EVAL_MAX_PROMPT_LEN" "$EVAL_MAX_PROMPT_LENGTH"
printf '  %-22s %s\n' "MAX_RESPONSE_LEN" "$MAX_RESPONSE_LENGTH"
printf '  %-22s %s\n' "EVAL_MAX_RESP_LEN" "$EVAL_MAX_RESPONSE_LENGTH"
printf '  %-22s %s\n' "SAVE_FREQ" "$SAVE_FREQ"
printf '  %-22s %s\n' "SAVE_STEPS" "$SAVE_STEPS"
printf '  %-22s %s\n' "RESUME_MODE" "$RESUME_MODE"
printf '  %-22s %s\n' "PLANNER_TOKENS" "$PLANNER_MAX_TOKENS"
printf '  %-22s %s\n' "EXECUTOR_TOKENS" "$EXECUTOR_MAX_TOKENS"
printf '  %-22s %s\n' "EXEC_REVIEW_TOKENS" "$EXECUTOR_REVIEWER_MAX_TOKENS"
printf '  %-22s %s\n' "TRAIN_BATCH_SIZE" "$TRAIN_BATCH_SIZE"
printf '  %-22s %s\n' "ROLLOUT_N" "$ROLLOUT_N"
printf '  %-22s %s\n' "PPO_MINI_BATCH" "$PPO_MINI_BATCH_SIZE"
printf '  %-22s %s\n' "GPU_MEM_UTIL" "$GPU_MEMORY_UTILIZATION"
printf '  %-22s %s\n' "MAX_BATCHED_TOKENS" "$MAX_NUM_BATCHED_TOKENS"
printf '  %-22s %s\n' "MAX_NUM_SEQS" "$MAX_NUM_SEQS"
printf '  %-22s %s\n' "ROLLOUT_MAX_TOKENS" "$ROLLOUT_MAX_TOKENS"
printf '  %-22s %s\n' "ACTOR_MAX_TOKENS_GPU" "$ACTOR_MAX_TOKEN_LEN_PER_GPU"
printf '  %-22s %s\n' "CRITIC_MAX_TOKENS_GPU" "$CRITIC_MAX_TOKEN_LEN_PER_GPU"
printf '  %-22s %s\n' "ROUNDS_CTRL" "$ROUNDS_CTRL_TYPE"
printf '  %-22s %s\n' "ROUNDS" "$ROUNDS_CTRL_ROUNDS"
printf '  %-22s %s\n' "ROUND_STEP" "$ROUNDS_CTRL_STEPS"

(
  cd "$TRAIN_ROOT"
  HYDRA_FULL_ERROR=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONNOUSERSITE=1 \
  TMPDIR="$TMPDIR" \
  RAY_TMPDIR="$RAY_TMPDIR" \
  RAY_object_spilling_config="{\"type\":\"filesystem\",\"params\":{\"directory_path\":\"$RAY_TMPDIR/spill\"}}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  MULTI_AGENT_PROMPT_STYLE="$MULTI_AGENT_PROMPT_STYLE" \
  TRAIN_TEMPERATURE="$TRAIN_TEMPERATURE" \
  EVAL_TEMPERATURE="$EVAL_TEMPERATURE" \
  ACTOR_MAX_TOKEN_LEN_PER_GPU="$ACTOR_MAX_TOKEN_LEN_PER_GPU" \
  CRITIC_MAX_TOKEN_LEN_PER_GPU="$CRITIC_MAX_TOKEN_LEN_PER_GPU" \
  MODEL_PATH="$MODEL_PATH" \
  DATA_ROOT="$DATA_ROOT" \
  ENV_SERVER_URL="$ENV_SERVER_URL" \
  SAVE_ROOT="$SAVE_ROOT" \
  EXP_NAME="$EXP_NAME" \
  N_GPUS="$N_GPUS" \
  TP_SIZE="$TP_SIZE" \
  "$PYTHON_BIN" -m verl.extend_multi_agent.main_ppo \
    task=$TASK_NAME \
    task.train_file=AgentItemId/train/webarena_train.json \
    task.max_prompt_length=$MAX_PROMPT_LENGTH \
    task.max_response_length=$MAX_RESPONSE_LENGTH \
    task.eval_max_prompt_length=$EVAL_MAX_PROMPT_LENGTH \
    task.eval_max_response_length=$EVAL_MAX_RESPONSE_LENGTH \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    task.executor_max_tokens=$EXECUTOR_MAX_TOKENS \
    task.rollout_max_tokens=$ROLLOUT_MAX_TOKENS \
    task.train_batch_size=$TRAIN_BATCH_SIZE \
    task.rollout_n=$ROLLOUT_N \
    task.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    runtime=qwen2_5_7b_8gpu \
    runtime.nnodes=$NNODES \
    runtime.n_gpus_per_node=$N_GPUS \
    runtime.tensor_model_parallel_size=$TP_SIZE \
    runtime.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    runtime.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    runtime.max_num_seqs=$MAX_NUM_SEQS \
    runtime.actor_ppo_max_token_len_per_gpu=$ACTOR_MAX_TOKEN_LEN_PER_GPU \
    runtime.critic_ppo_max_token_len_per_gpu=$CRITIC_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.rollout.prompt_length=$MAX_PROMPT_LENGTH \
    actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH \
    algo=multi_agent_gae \
    algorithm.rounds_ctrl.type=$ROUNDS_CTRL_TYPE \
    algorithm.rounds_ctrl.rounds=$ROUNDS_CTRL_ROUNDS \
    algorithm.rounds_ctrl.steps_scaling_inter=$ROUNDS_CTRL_STEPS \
    trainer.default_local_dir="$RUN_DIR" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.nnodes=$NNODES \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.resume_mode=$RESUME_MODE \
    trainer.save_freq=$SAVE_FREQ \
    trainer.save_steps=$SAVE_STEPS \
    algo.use_kl_loss=true \
    algo.kl_coef=0.001 \
    multi_agent.topology=planner_executor_reviewer \
    multi_agent.roles.planner.max_tokens=$PLANNER_MAX_TOKENS \
    multi_agent.roles.planner.temperature=0.2 \
    multi_agent.roles.executor.max_tokens=$EXECUTOR_MAX_TOKENS \
    multi_agent.roles.executor.temperature=$EXECUTOR_TEMPERATURE \
    multi_agent.roles.executor_reviewer.max_tokens=$EXECUTOR_REVIEWER_MAX_TOKENS \
    multi_agent.roles.executor_reviewer.temperature=0.2 \
    multi_agent.roles.executor_reviewer.ppo_weight=0.25 \
    multi_agent.roles.executor_reviewer.kl_weight=6.0 \
    actor_rollout_ref.model.use_remove_padding=true \
    critic.model.use_remove_padding=true \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    critic.use_dynamic_bsz=true \
    actor_rollout_ref.actor.planner_kl_weight=4.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    multi_agent.invalid_output.policy=terminate_with_penalty \
    multi_agent.invalid_output.penalty=-0.2 \
    multi_agent.invalid_output.max_retries=3 \
    multi_agent.invalid_output.retry_temperature=0.2 \
    multi_agent.invalid_output.retry_max_tokens=128 \
    multi_agent.invalid_output.planner_max_retries=0 \
    multi_agent.invalid_output.planner_retry_temperature=0.1 \
    multi_agent.invalid_output.planner_retry_max_tokens=$PLANNER_MAX_TOKENS \
    multi_agent.debug.trace_executor_payload=true \
    multi_agent.debug.trace_dir="$TRACE_DIR" \
    multi_agent.debug.trace_max_chars=800 \
    multi_agent.debug.trace_first_training_steps=10 \
    multi_agent.debug.trace_every_training_steps=10 \
    multi_agent.debug.trace_on_planner_invalid=true \
    multi_agent.debug.trace_on_planner_fallback=true \
    multi_agent.debug.trace_on_executor_invalid_format=true \
    multi_agent.debug.trace_on_invalid_action=true
) 2>&1 | tee "$LOG_DIR/train_step200.log"

printf '3-agent WebArena executor-reviewer scaling run complete.\n'
