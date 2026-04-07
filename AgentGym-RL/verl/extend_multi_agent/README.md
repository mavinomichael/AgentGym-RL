# extend_multi_agent

This package is a BabyAI-first fork of `verl.multi_agent`.

Key changes in this fork:
- strict planner/executor JSON prompts and validators
- executor JSON adapts to the single BabyAI environment action in one place
- fail-fast invalid JSON handling in the copied rollout path
- alternating-role PPO helper utilities for planner-only, executor-only, and joint phases
- minimal SFT bootstrap helpers for JSON obedience

Use `/Users/mavinomichael/PycharmProjects/AgentGym-RL/extend-multi-agent/README.md` for run commands.
