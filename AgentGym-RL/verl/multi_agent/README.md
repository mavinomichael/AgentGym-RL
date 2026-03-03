# Multi-Agent Extension

This package contains an isolated planner-executor multi-agent extension for AgentGym-RL.

Design goals:
- additive only
- no edits required to the original single-agent path
- detachable by removing `verl/multi_agent` and the corresponding `examples/*/MultiAgent` folders

Initial scope:
- planner + executor topology
- shared actor policy
- centralized critic over the joint transcript
- SciWorld pilot environment
