# babyai_2agent_scaling_100_8gpu_plain_split_retry_v2 checkpoint eval comparison

- Generated: 2026-03-22 21:22:22 Africa/Lagos
- VM: `odion-agentgym-sweep-w3-h100-as1c` (`34.87.90.34`, zone `asia-southeast1-c`)
- Training status: reached `global_step_500`
- Eval action taken: no new evals launched, because every saved checkpoint under `/home/mavinomichael/agentgym_runs/saves/agentgym_multi_agent/babyai_2agent_scaling_100_8gpu_plain_split_retry_v2` already had a completed `eval_step*.log`
- Eval setup source: reused the existing experiment eval logs in `/home/mavinomichael/agentgym_runs/logs/babyai_2agent_scaling_100_8gpu_plain_split_retry_v2/`
- Standard eval configuration verified from existing logs:
  - `task_name: babyai`
  - `max_rounds: 20`
  - `eval_batch_size: 32`
  - `eval_n_samples: 1`
  - `executor_max_tokens: 200`
  - `planner max_tokens: 96`
  - `executor max_tokens: 200`
  - `planner_reviewer max_tokens: 96`
  - `executor_reviewer max_tokens: 64`

## Summary

The strongest checkpoint by both `Avg@1` and `Pass@1` is `step 200` with `Avg@1=0.755230` and `Pass@1=0.811111`. `step 300` remains competitive (`Avg@1=0.723644`, `Pass@1=0.788889`) but shows a higher `InvalidActionTerminationRate` than `step 200`.

The run improves steadily from `step 50` through `step 200`, then degrades after `step 300`. `step 350` drops on action validity (`InvalidActionTerminationRate=0.200000`), `step 400` degrades further (`InvalidActionTerminationRate=0.244444`), and both `step 450` and `step 500` collapse completely into executor format failure with `Avg@1=-0.200000`, `Pass@1=0.000000`, `ExecutorNativeFormatViolations=1.000000`, and `InvalidFormatTerminationRate=1.000000`.

`PlannerInvalidFormatRate`, `PlannerFallbackRate`, and `PlannerTagOnlyRate` are all `0.000000` from `step 200` onward, so the late collapse is concentrated in the executor path rather than planner formatting.

## Best checkpoints

| Rank | Step | Avg@1 | Pass@1 | ExecutorNativeFormatViolations | InvalidFormatTerminationRate | InvalidActionTerminationRate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 200 | 0.755230 | 0.811111 | 0.044444 | 0.044444 | 0.022222 |
| 2 | 300 | 0.723644 | 0.788889 | 0.033333 | 0.033333 | 0.077778 |
| 3 | 250 | 0.690505 | 0.744444 | 0.044444 | 0.044444 | 0.011111 |
| 4 | 350 | 0.634342 | 0.711111 | 0.000000 | 0.000000 | 0.200000 |
| 5 | 150 | 0.624248 | 0.688889 | 0.133333 | 0.133333 | 0.011111 |

## Full table

See [`checkpoint_metrics.tsv`](/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/20260322_212222_babyai_2agent_scaling_100_8gpu_plain_split_retry_v2/checkpoint_metrics.tsv).
