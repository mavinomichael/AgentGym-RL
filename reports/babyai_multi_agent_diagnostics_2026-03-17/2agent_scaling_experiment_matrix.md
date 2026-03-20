# Planned Run: BabyAI 2-Agent + Aligned Prompts + ScalingInter `[6,13,20]`

## Intent
This is the next planned run.
It is not a comparison study. It is the next primary training configuration:
- 2-agent topology
- aligned prompts
- ScalingInter-style turn curriculum
- no launch yet

## Top-level configuration
| Field | Value |
|---|---|
| Run name | `babyai_2agent_scaling_200_8gpu` |
| Topology | `planner_executor` |
| Prompt regime | aligned to the original single-agent BabyAI contract |
| Round control | `scaling_inter_stepwise` |
| Round schedule | `[6,13,20]` |
| Step interval | `100` |
| Total training steps | `201` |
| Save frequency | `50` |
| Eval checkpoints | `50, 100, 150, 200` |
| GPUs | `8` |

## Exact training settings
| Group | Setting | Value |
|---|---|---|
| Trainer | `trainer.total_training_steps` | `201` |
| Trainer | `trainer.save_freq` | `50` |
| Trainer | `trainer.resume_mode` | `disable` |
| Algorithm | `algo` | `multi_agent_gae` |
| Algorithm | `algo.use_kl_loss` | `true` |
| Algorithm | `algo.kl_coef` | `0.001` |
| Round control | `algorithm.rounds_ctrl.type` | `scaling_inter_stepwise` |
| Round control | `algorithm.rounds_ctrl.rounds` | `[6,13,20]` |
| Round control | `algorithm.rounds_ctrl.steps_scaling_inter` | `100` |
| Task | `task.train_file` | `AgentItemId/train/babyai_train.json` |
| Task | `task.train_batch_size` | `8` |
| Task | `task.rollout_n` | `2` |
| Task | `task.ppo_mini_batch_size` | `8` |
| Runtime | `runtime.gpu_memory_utilization` | `0.55` |
| Runtime | `runtime.max_num_batched_tokens` | `1024` |
| Runtime | `runtime.max_num_seqs` | `64` |
| Multi-agent | `multi_agent.topology` | `planner_executor` |
| Planner | `multi_agent.roles.planner.max_tokens` | `16` |
| Planner | `multi_agent.roles.planner.temperature` | `0.2` |
| Actor | `actor_rollout_ref.actor.planner_kl_weight` | `4.0` |
| Invalid-output | `multi_agent.invalid_output.max_retries` | `2` |
| Invalid-output | `multi_agent.invalid_output.retry_temperature` | `0.2` |
| Invalid-output | `multi_agent.invalid_output.retry_max_tokens` | `80` |
| Planner retry | `multi_agent.invalid_output.planner_max_retries` | `3` |
| Planner retry | `multi_agent.invalid_output.planner_retry_temperature` | `0.1` |
| Planner retry | `multi_agent.invalid_output.planner_retry_max_tokens` | `16` |
| Tracing | `trace_first_training_steps` | `15` |
| Tracing | `trace_every_training_steps` | `5` |
| Tracing | planner/executor invalid hooks | `true` |

## Expected turn schedule
| Training step range | Max interaction rounds |
|---|---:|
| `0-99` | `6` |
| `100-199` | `13` |
| `200+` | `20` |

## Why this schedule
- It matches the repo's BabyAI ScalingInter example rather than the generic README example.
- It respects the current BabyAI task shape better than `[10,20,30]`.
- It should reduce early long-horizon pressure while still reaching the full 20-round regime by the end.

## Launch script
- `/Users/mavinomichael/PycharmProjects/AgentGym-RL/scripts/multi_agent/run_babyai_2agent_scaling_200_8gpu.sh`

## Exact command to use later
```bash
cd /home/mavinomichael/AgentGym-RL
bash /home/mavinomichael/AgentGym-RL/scripts/multi_agent/run_babyai_2agent_scaling_200_8gpu.sh
```

## Explicit override form
```bash
cd /home/mavinomichael/AgentGym-RL
EXP_NAME=babyai_2agent_scaling_200_8gpu \
ROUNDS_CTRL_TYPE=scaling_inter_stepwise \
ROUNDS_CTRL_ROUNDS='[6,13,20]' \
ROUNDS_CTRL_STEPS=100 \
TOTAL_TRAINING_STEPS=201 \
SAVE_FREQ=50 \
CHECKPOINT_STEPS='50 100 150 200' \
bash /home/mavinomichael/AgentGym-RL/scripts/multi_agent/run_babyai_2agent_scaling_200_8gpu.sh
```

## Evaluation plan
| Checkpoint | Purpose |
|---|---|
| `50` | early signal under the `6`-round regime |
| `100` | end of first curriculum stage |
| `150` | mid-run under the `13`-round regime |
| `200` | final state after entering the `20`-round regime |

## What to inspect first after launch
| Phase | Metrics to check |
|---|---|
| Early (`<=50`) | `planner_invalid_format_rate`, `planner_fallback_rate`, `executor_invalid_action_rate`, `executor_native_format_violations` |
| Mid (`100-150`) | reward/task score trend, planner retry load, executor invalid rates |
| Late (`>=150`) | whether zero-reward collapse appears, whether planner rewrite/fallback appears, whether actor stats remain finite |

## Success condition
A good run is one where:
- planner fallback stays near zero
- executor format errors remain materially below the failed 4-agent reviewer regime
- reward does not enter an extended zero plateau
- checkpoint quality does not collapse monotonically after the first save

## Related files
- Single-agent BabyAI instruction: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/AgentGym/agentenv/agentenv/envs/babyai.py`
- Multi-agent aligned prompt policy: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/AgentGym-RL/verl/multi_agent/envs/prompt_policy.py`
- Prompt alignment notes: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/babyai_multi_agent_diagnostics_2026-03-17/prompt_alignment_single_vs_multi.md`
