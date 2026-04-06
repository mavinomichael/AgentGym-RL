# Paper To Code Mapping

## Dr. MAS
- role-wise reward normalization: `AgentGym-RL/verl/improve_multi_agent/advantage.py`
- role-local advantage whitening: `AgentGym-RL/verl/improve_multi_agent/advantage.py`
- role-local PPO/KL budgets: `AgentGym-RL/verl/improve_multi_agent/actor.py`

## MALT / MARFT
- planner vs executor auxiliary rewards: `AgentGym-RL/verl/improve_multi_agent/rewarding.py`
- role-specific rollout bookkeeping: `AgentGym-RL/verl/improve_multi_agent/workers/rollout/agent_vllm_rollout/vllm_rollout.py`

## HiMAC
- planner acts every 3 executor steps: `AgentGym-RL/verl/improve_multi_agent/workers/rollout/agent_vllm_rollout/vllm_rollout.py`

## Subgoal-driven
- milestone rewards and subgoal completion: `AgentGym-RL/verl/improve_multi_agent/rewarding.py`

## DAMCS / Communicating Plans, Not Percepts
- planner JSON schema and executor JSON action index: `AgentGym-RL/verl/improve_multi_agent/protocol.py`
