# Multi-agent extension.
# Derived from: /Users/mavinomichael/PycharmProjects/AgentGym-RL/AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/__init__.py
# Original file left untouched for comparison.

__all__ = ["vLLMRollout"]


def __getattr__(name):
    if name == "vLLMRollout":
        from .vllm_rollout import vLLMRollout

        return vLLMRollout
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
