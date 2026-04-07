# Multi-agent extension.
# Derived from: /Users/mavinomichael/PycharmProjects/AgentGym-RL/AgentGym-RL/verl/agent_trainer/main_ppo.py
# Original file left untouched for comparison.

from verl.extend_multi_agent.ppo.ray_trainer import RayPPOTrainer

import hydra
import ray


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config):
    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})
    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    from omegaconf import OmegaConf, open_dict
    from pprint import pprint
    from verl.extend_multi_agent.ppo.ray_trainer import ResourcePoolManager, Role
    from verl.extend_multi_agent.workers.agent_fsdp_workers import ActorRolloutRefWorker, CriticWorker
    from verl.single_controller.ray import RayWorkerGroup
    from verl.utils import hf_tokenizer
    from verl.utils.fs import copy_local_path_from_hdfs

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)
    if str(config.actor_rollout_ref.agentgym.task_name).lower() != "babyai":
        raise ValueError(
            "extend_multi_agent currently supports only BabyAI. "
            "Use verl.multi_agent for other tasks."
        )
    train_role = str(config.extend_multi_agent.train_role).lower()
    with open_dict(config.actor_rollout_ref):
        config.actor_rollout_ref.multi_agent = config.multi_agent
        config.actor_rollout_ref.extend_multi_agent = config.extend_multi_agent
    with open_dict(config.multi_agent):
        if train_role == "planner":
            config.multi_agent.roles.executor.ppo_weight = 0.0
            config.multi_agent.roles.executor.kl_weight = 0.0
        elif train_role == "executor":
            config.multi_agent.roles.planner.ppo_weight = 0.0
            config.multi_agent.roles.planner.kl_weight = 0.0

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path)

    if config.actor_rollout_ref.actor.strategy != "fsdp":
        raise NotImplementedError

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = "global_pool"
    resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=RayWorkerGroup,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
