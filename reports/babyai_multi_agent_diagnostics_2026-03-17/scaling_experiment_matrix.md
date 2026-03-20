# Planned Experiment Matrix: Aligned Prompts + ScalingInter-RL for BabyAI

## Scope
This matrix defines the next planned BabyAI experiment using:
- prompt-aligned multi-agent roles
- 4-agent topology: `planner_executor_reviewers`
- ScalingInter-style round curriculum: `[6,13,20]`
- no launch yet

Related implementation files:
- Prompt policy: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/AgentGym-RL/verl/multi_agent/envs/prompt_policy.py`
- Base reviewer launcher: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/scripts/multi_agent/run_babyai_reviewers_200_8gpu.sh`
- Scaling launcher: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/scripts/multi_agent/run_babyai_reviewers_scaling_200_8gpu.sh`
- Prompt comparison report: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/babyai_multi_agent_diagnostics_2026-03-17/prompt_alignment_single_vs_multi.md`

## Primary Run
| Field | Value |
|---|---|
| Run name | `babyai_reviewers_scaling_200_8gpu` |
| Task | `babyai` |
| Topology | `planner_executor_reviewers` |
| Prompting | aligned to original single-agent BabyAI contract |
| Round control type | `scaling_inter_stepwise` |
| Round schedule | `[6,13,20]` |
| Scaling step interval | `100` |
| Total training steps | `201` |
| Save frequency | `50` |
| Checkpoints to evaluate | `50, 100, 150, 200` |
| GPUs | `8` |
| Runtime | `qwen2_5_7b_8gpu` |
| Resume mode | `disable` |

## Exact Training Configuration
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
| Multi-agent | `multi_agent.topology` | `planner_executor_reviewers` |
| Planner | `multi_agent.roles.planner.max_tokens` | `64` |
| Planner | `multi_agent.roles.planner.temperature` | `0.7` |
| Planner reviewer | `multi_agent.roles.planner_reviewer.max_tokens` | `96` |
| Planner reviewer | `multi_agent.roles.planner_reviewer.temperature` | `0.2` |
| Executor reviewer | `multi_agent.roles.executor_reviewer.max_tokens` | `64` |
| Executor reviewer | `multi_agent.roles.executor_reviewer.temperature` | `0.2` |
| Invalid-output | `multi_agent.invalid_output.max_retries` | `5` |
| Invalid-output | `multi_agent.invalid_output.retry_temperature` | `0.2` |
| Invalid-output | `multi_agent.invalid_output.retry_max_tokens` | `80` |
| Planner retry | `multi_agent.invalid_output.planner_max_retries` | `5` |
| Planner retry | `multi_agent.invalid_output.planner_retry_temperature` | `0.1` |
| Planner retry | `multi_agent.invalid_output.planner_retry_max_tokens` | `64` |
| Tracing | `trace_first_training_steps` | `15` |
| Tracing | `trace_every_training_steps` | `5` |
| Tracing | planner invalid/fallback hooks | `true` |
| Tracing | executor invalid/action hooks | `true` |

## Expected Horizon Schedule
| Training step range | Max interaction rounds |
|---|---:|
| `0-99` | `6` |
| `100-199` | `13` |
| `200+` | `20` |

## Checkpoint Evaluation Matrix
| Checkpoint | Purpose | What to inspect first |
|---|---|---|
| `50` | early curriculum phase under `6` rounds | whether reward improves without early role drift |
| `100` | boundary checkpoint before `13`-round expansion | whether executor format stays stable through the first phase |
| `150` | mid second phase under `13` rounds | whether reviewer load grows slower than in the fixed-round run |
| `200` | final checkpoint after entering `20`-round regime | whether scaling delays or avoids late collapse |

## Success Criteria
| Metric family | Target |
|---|---|
| Planner health | `planner_fallback_rate = 0` or near-zero through training |
| Planner repair load | lower than the previous fixed-round 4-agent run in the `100-150` region |
| Executor health | materially lower `executor_native_format_violations` and `executor_invalid_action_rate` than the fixed-round 4-agent run |
| Reward trajectory | no extended zero-reward regime like the prior late-stage collapse |
| Eval trend | flatter degradation from `50 -> 100 -> 150 -> 200` than the prior fixed-round reviewer run |

## Failure Criteria
| Condition | Interpretation |
|---|---|
| `planner_rewrite_rate` approaches `1.0` for multiple consecutive steps | planner no longer provides usable raw drafts |
| `planner_reviewer_retry_mean` reaches cap repeatedly | planner channel is overloaded |
| `executor_reviewer_pass_rate` falls below `0.5` for sustained windows | executor path is no longer recoverable |
| reward/task score stays at `0` over many steps | effective behavioral collapse |
| any `nan` in actor/critic metrics | numerical instability |

## Exact Launch Command
Run this on the VM when resources are available:

```bash
cd /home/mavinomichael/AgentGym-RL
bash /home/mavinomichael/AgentGym-RL/scripts/multi_agent/run_babyai_reviewers_scaling_200_8gpu.sh
```

## Exact Environment Overrides
If you want the command fully explicit:

```bash
cd /home/mavinomichael/AgentGym-RL
EXP_NAME=babyai_reviewers_scaling_200_8gpu \
ROUNDS_CTRL_TYPE=scaling_inter_stepwise \
ROUNDS_CTRL_ROUNDS='[6,13,20]' \
ROUNDS_CTRL_STEPS=100 \
TOTAL_TRAINING_STEPS=201 \
SAVE_FREQ=50 \
CHECKPOINT_STEPS='50 100 150 200' \
bash /home/mavinomichael/AgentGym-RL/scripts/multi_agent/run_babyai_reviewers_scaling_200_8gpu.sh
```

## Post-Run Comparison Target
The main comparison baseline is the prior fixed-round 4-agent reviewer run documented in:
- `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/babyai_multi_agent_diagnostics_2026-03-14/reviewer_eval_summary.txt`

That baseline degraded as follows:
- `step 50`: `Avg@1 0.739644`, `Pass@1 0.777778`
- `step 100`: `Avg@1 0.510516`, `Pass@1 0.533333`
- `step 150`: `Avg@1 0.134307`, `Pass@1 0.144444`

The scaling run should be judged primarily on whether it reduces that degradation rate.
