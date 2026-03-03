# Multi-agent extension.
# Derived from: /Users/mavinomichael/PycharmProjects/AgentGym-RL/AgentGym-RL/verl/workers/agent_fsdp_workers.py
# Original file left untouched for comparison.

from verl.workers.agent_fsdp_workers import ActorRolloutRefWorker as BaseActorRolloutRefWorker
from verl.workers.agent_fsdp_workers import CriticWorker as BaseCriticWorker
from verl.utils.debug import log_gpu_memory_usage


class ActorRolloutRefWorker(BaseActorRolloutRefWorker):
    def _build_rollout(self):
        from torch.distributed.device_mesh import init_device_mesh
        from verl.multi_agent.workers.rollout.agent_vllm_rollout import vLLMRollout
        from verl.workers.sharding_manager import FSDPVLLMShardingManager
        import torch

        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        )
        rollout_device_mesh = init_device_mesh(
            "cuda", mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )

        log_gpu_memory_usage("Before building multi-agent vllm rollout", logger=None)
        rollout = vLLMRollout(
            actor_module=self.actor_module_fsdp,
            rollout_config=self.config.rollout,
            agentgym_config=self.config.agentgym,
            tokenizer=self.tokenizer,
            model_hf_config=self.actor_model_config,
            multi_agent_config=self.config.get("multi_agent", None),
        )
        log_gpu_memory_usage("After building multi-agent vllm rollout", logger=None)
        if torch.distributed.get_world_size() == 1:
            self.config.rollout.load_format = "dummy_hf"
        rollout_sharding_manager = FSDPVLLMShardingManager(
            module=self.actor_module_fsdp,
            inference_engine=rollout.inference_engine,
            model_config=self.actor_model_config,
            full_params="hf" in self.config.rollout.load_format,
            device_mesh=rollout_device_mesh,
        )
        log_gpu_memory_usage("After building multi-agent sharding manager", logger=None)
        return rollout, rollout_sharding_manager


class CriticWorker(BaseCriticWorker):
    pass
