# Multi-agent extension.
# Derived from: /Users/mavinomichael/PycharmProjects/AgentGym-RL/AgentGym-RL/verl/agent_trainer/ppo
# Original files left untouched for comparison.

from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role

__all__ = ["RayPPOTrainer", "ResourcePoolManager", "Role"]
