# Multi-agent extension.
# Derived from: /Users/mavinomichael/PycharmProjects/AgentGym-RL/AgentGym-RL/verl/utils/agent_dataset
# Original files left untouched for comparison.

from .rl_dataset import RLHFDataset, collate_fn, build_multi_agent_bootstrap

__all__ = ["RLHFDataset", "collate_fn", "build_multi_agent_bootstrap"]
