set -x
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS

task_name="sciworld"

cd AgentGym-RL
source activate
conda activate agentgym-rl

env_server_url="http://127.0.0.1:36005"
pure_agent_model_name="Qwen2.5-7B-Instruct"
agent_model_path="models/${pure_agent_model_name}"
model_save_dir="saves"
exp_name="multiagent-sciworld"
model_save_path=${model_save_dir}/${exp_name}
mkdir -p ${model_save_path}

HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m verl.multi_agent.main_ppo \
    data.train_file=AgentItemId/${task_name}_train.json \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    actor_rollout_ref.agentgym.task_name=${task_name} \
    actor_rollout_ref.agentgym.env_addr=${env_server_url} \
    actor_rollout_ref.agentgym.timeout=600 \
    actor_rollout_ref.model.path=${agent_model_path} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.max_tokens=200 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.rollout_log_dir=${model_save_path}/executer_logs \
    multi_agent.roles.planner.max_tokens=128 \
    multi_agent.roles.executor.max_tokens=200 \
    trainer.default_local_dir=${model_save_path} \
    trainer.project_name=multiagent \
    trainer.experiment_name=${exp_name} \
    trainer.save_freq=25 \
    trainer.total_epochs=10
