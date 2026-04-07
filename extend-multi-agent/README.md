# Extend Multi-Agent v1

This workspace is a BabyAI-first experimental fork of `/Users/mavinomichael/PycharmProjects/AgentGym-RL/AgentGym-RL/verl/multi_agent`.

## What Was Copied
- `/Users/mavinomichael/PycharmProjects/AgentGym-RL/AgentGym-RL/verl/multi_agent` -> `/Users/mavinomichael/PycharmProjects/AgentGym-RL/AgentGym-RL/verl/extend_multi_agent`
- `/Users/mavinomichael/PycharmProjects/AgentGym-RL/scripts/multi_agent` -> `/Users/mavinomichael/PycharmProjects/AgentGym-RL/scripts/extend-multi-agent`

## What Changed
- strict planner/executor JSON protocol for BabyAI
- fail-fast invalid JSON and invalid action handling in the copied rollout path
- new role-freezing utilities for alternating planner/executor PPO phases
- minimal BabyAI SFT dataset builder that emits planner/executor JSON-obedience datasets
- extend-specific tests and example scripts

## Run
1. Build and SFT both roles
   - `/Users/mavinomichael/PycharmProjects/AgentGym-RL/scripts/extend-multi-agent/run_babyai_sft_both.sh`
2. Freeze executor, train planner
   - `/Users/mavinomichael/PycharmProjects/AgentGym-RL/scripts/extend-multi-agent/run_babyai_train_planner_phase.sh`
3. Freeze planner, train executor
   - `/Users/mavinomichael/PycharmProjects/AgentGym-RL/scripts/extend-multi-agent/run_babyai_train_executor_phase.sh`
4. Optional short joint PPO fine-tune
   - `/Users/mavinomichael/PycharmProjects/AgentGym-RL/scripts/extend-multi-agent/run_babyai_joint_phase.sh`

## Notes
- `extend_multi_agent` supports only BabyAI in v1.
- The original `verl.multi_agent` path is left untouched and remains the baseline.
- The centralized critic stays shared in alternating phases.
