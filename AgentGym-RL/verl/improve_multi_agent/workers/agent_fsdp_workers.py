from __future__ import annotations

from verl.multi_agent.workers.agent_fsdp_workers import CriticWorker as BaseCriticWorker
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.agent_fsdp_workers import ActorRolloutRefWorker as BaseActorRolloutRefWorker
from verl.utils.debug import log_gpu_memory_usage


class ActorRolloutRefWorker(BaseActorRolloutRefWorker):
    def _build_rollout(self):
        from torch.distributed.device_mesh import init_device_mesh
        from verl.improve_multi_agent.workers.rollout.agent_vllm_rollout import vLLMRollout
        from verl.workers.sharding_manager import FSDPVLLMShardingManager
        import torch

        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0
        rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])
        rollout = vLLMRollout(
            actor_module=self.actor_module_fsdp,
            rollout_config=self.config.rollout,
            agentgym_config=self.config.agentgym,
            tokenizer=self.tokenizer,
            model_hf_config=self.actor_model_config,
            improve_config=self.config.get("improve_multi_agent", None),
        )
        if torch.distributed.get_world_size() == 1:
            self.config.rollout.load_format = "dummy_hf"
        rollout_sharding_manager = FSDPVLLMShardingManager(
            module=self.actor_module_fsdp,
            inference_engine=rollout.inference_engine,
            model_config=self.actor_model_config,
            full_params="hf" in self.config.rollout.load_format,
            device_mesh=rollout_device_mesh,
        )
        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.improve_multi_agent.actor import RoleAwareDataParallelPPOActor
        from verl.workers.agent_actor import DataParallelPPOActor
        from verl.utils.fsdp_utils import offload_fsdp_grad, offload_fsdp_optimizer
        from verl.utils.flops_counter import FlopsCounter
        from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
        from verl.utils.import_utils import import_external_libs
        from omegaconf import OmegaConf, open_dict
        import torch

        import_external_libs(self.config.model.get("external_lib", None))
        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        use_remove_padding = self.config.model.get("use_remove_padding", False)

        if self._is_actor or self._is_rollout:
            optim_config = self.config.actor.optim
            fsdp_config = self.config.actor.fsdp_config
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
            )
            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module
            if self._is_offload_param:
                offload_fsdp_grad(module=self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor grad during init", logger=None)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=None)

        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
            self.actor = RoleAwareDataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout()

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                tokenizer=self.tokenizer,
            )
        torch.cuda.empty_cache()


class CriticWorker(BaseCriticWorker):
    pass
