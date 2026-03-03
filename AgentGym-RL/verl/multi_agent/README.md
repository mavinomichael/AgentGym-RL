# Multi-Agent Extension

This package contains an isolated planner-executor multi-agent extension for AgentGym-RL.

Design goals:
- additive only
- no edits required to the original single-agent path
- detachable by removing `verl/multi_agent` and the corresponding `examples/*/MultiAgent` folders

Current scope:
- planner + executor topology
- shared actor policy
- centralized critic over the joint transcript
- five RL environments: `webarena`, `sciworld`, `searchqa`, `babyai`, `textcraft`
- Qwen-first runnable configs for 8-GPU training and 1-GPU smoke evaluation
