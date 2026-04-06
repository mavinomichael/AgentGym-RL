# babyai_3agent_executor_reviewer_scaling_600_8gpu_no_tags_v1 checkpoint eval comparison

- Generated: 2026-04-03 13:06:17 Africa/Lagos
- VM: `odion-agentgym-sweep-h100-apac-20260331-asia-southeast1-b` (zone `asia-southeast1-b`)
- Training status: reached `global_step_600`
- Eval action taken: no new evals launched, because every saved checkpoint under `/home/mavinomichael/agentgym_runs/saves/agentgym_multi_agent/babyai_3agent_executor_reviewer_scaling_600_8gpu_no_tags_v1` already had a completed `eval_step*.log`
- Eval setup source: reused the existing experiment eval logs in `/home/mavinomichael/agentgym_runs/logs/babyai_3agent_executor_reviewer_scaling_600_8gpu_no_tags_v1/`

## Summary

The strongest checkpoint by both `Avg@1` and `Pass@1` is `step 100` with `Avg@1=0.691801` and `Pass@1=0.722222`. `step 350` is the closest later checkpoint (`Avg@1=0.668654`, `Pass@1=0.711111`) and has substantially lower executor-format failure than `step 100`, but it still trails on headline task success.

The run is unstable rather than monotonically improving. Early checkpoints swing sharply from `step 100` to `step 200`, then recover through `step 300` and `step 350`. By `step 400`, planner invalid-format errors rise to `0.700000`, and checkpoints `450`, `500`, `550`, and `600` collapse completely with `Avg@1=-0.200000`, `Pass@1=0.000000`, `ExecutorNativeFormatViolations=1.000000`, `InvalidFormatTerminationRate=1.000000`, and `PlannerTagOnlyRate=1.000000`.

The late-run failure is broader than a pure executor regression: from `step 450` onward both executor-format metrics and planner-format metrics saturate. The best recovery candidate for follow-up analysis is therefore the `step 100` to `step 350` band, especially `step 100`, `step 300`, and `step 350`.

## Best checkpoints

| Rank | Step | Avg@1 | Pass@1 | ExecutorNativeFormatViolations | InvalidFormatTerminationRate | PlannerInvalidFormatRate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 100 | 0.691801 | 0.722222 | 0.000000 | 0.000000 | 0.477778 |
| 2 | 350 | 0.668654 | 0.711111 | 0.011111 | 0.011111 | 0.511111 |
| 3 | 300 | 0.632004 | 0.677778 | 0.022222 | 0.022222 | 0.500000 |
| 4 | 250 | 0.542972 | 0.611111 | 0.088889 | 0.088889 | 0.344444 |
| 5 | 50 | 0.542954 | 0.566667 | 0.033333 | 0.044444 | 0.522222 |

## Full table

See [`checkpoint_metrics.tsv`](/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/20260403_130617_babyai_3agent_executor_reviewer_scaling_600_8gpu_no_tags_v1/checkpoint_metrics.tsv).
