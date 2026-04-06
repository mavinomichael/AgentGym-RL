# Improve Multi-Agent v1 Implementation Status

## Implemented now
- Isolated BabyAI-first research stack under `AgentGym-RL/verl/improve_multi_agent`
- Strict JSON planner schema and JSON executor action-index schema
- Deterministic BabyAI action rendering from `action_id`
- Planner interval scheduling with forced replanning hooks
- Role-specific rollout bookkeeping for planner/executor masks and metrics
- Planner/executor auxiliary reward functions and milestone detection
- Collapse monitoring with warning and hard-collapse states
- Role-local advantage normalization and role-local PPO/KL budgets
- Trace-driven warm-start dataset builders for planner and executor
- Stage launchers for:
  - executor warm-start
  - planner warm-start
  - protocol-only RL
  - role-wise RL
  - planner-frozen RL
  - joint RL

## Current approximation
- The v1 PPO implementation still uses a **shared backbone with role-local optimization**
  rather than two fully separate planner/executor actor backbones.
- This means planner-only training is approximated by zeroing executor PPO/KL weights
  rather than instantiating a completely frozen executor policy module.

## Planned next upgrade
- Split planner and executor into separate actor modules with independent optimizer
  state and checkpointing while reusing the same base initialization checkpoint.
- Add stage-level report generation for collapse summaries and plots.
- Port the structured protocol to WebArena only after BabyAI gates pass.
