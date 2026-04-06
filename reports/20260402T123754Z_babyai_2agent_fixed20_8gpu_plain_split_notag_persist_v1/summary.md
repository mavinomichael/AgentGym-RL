# babyai_2agent_fixed20_8gpu_plain_split_notag_persist_v1 checkpoint eval comparison

- Generated: 2026-04-02 13:39:56 WAT
- VM: `odion-agentgym-sweep-h100-apac-20260331-asia-southeast1-b` (zone `asia-southeast1-b`)
- Training status: reached `global_step_600`
- Eval action taken: no new evals launched, because every saved checkpoint under `/home/mavinomichael/agentgym_runs/saves/agentgym_multi_agent/babyai_2agent_fixed20_8gpu_plain_split_notag_persist_v1` already had a completed `eval_step*.log`
- Eval setup verified from `eval_step600.log`:
  - `task_name: babyai`
  - `topology: planner_executor`
  - `max_rounds: 20`
  - `eval_batch_size: 32`
  - `eval_n_samples: 1`
  - `executor_max_tokens: 200`
  - `planner_max_tokens: 96`
  - `ScalingRL disabled during eval` (fixed-round setup only)
  - `plain-split no-tag planner/executor` configuration reused from the existing experiment eval logs

## Summary

The strongest checkpoint by both `Avg@1` and `Pass@1` is `step 100` with `Avg@1=0.639864` and `Pass@1=0.677778`.

The run peaks early, then drifts into total collapse. `step 150` shows a temporary action-validity regression with `InvalidActionTerminationRate=0.188889` despite intact formatting, while the `250-350` window stays middling but stable. Starting at `step 400`, every evaluated checkpoint collapses to floor reward with `Avg@1=-0.200000`, `Pass@1=0.000000`, `ExecutorNativeFormatViolations=1.000000`, `InvalidFormatTerminationRate=1.000000`, `PlannerInvalidFormatRate=1.000000`, and `PlannerTagOnlyRate=1.000000`.

Among non-collapsed checkpoints, the top 5 by `Pass@1` are:

| Rank | Step | Avg@1 | Pass@1 | ExecutorNativeFormatViolations | InvalidFormatTerminationRate | InvalidActionTerminationRate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 100 | 0.639864 | 0.677778 | 0.011111 | 0.011111 | 0.011111 |
| 2 | 150 | 0.528700 | 0.588889 | 0.000000 | 0.000000 | 0.188889 |
| 3 | 350 | 0.489064 | 0.511111 | 0.000000 | 0.000000 | 0.000000 |
| 4 | 250 | 0.478522 | 0.500000 | 0.000000 | 0.000000 | 0.000000 |
| 5 | 300 | 0.469112 | 0.488889 | 0.000000 | 0.000000 | 0.000000 |

## Full table

See [`checkpoint_metrics.tsv`](/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/20260402T123754Z_babyai_2agent_fixed20_8gpu_plain_split_notag_persist_v1/checkpoint_metrics.tsv).

## Notes

- Raw remote save root: `/home/mavinomichael/agentgym_runs/saves/agentgym_multi_agent/babyai_2agent_fixed20_8gpu_plain_split_notag_persist_v1`
- Raw remote eval logs: `/home/mavinomichael/agentgym_runs/logs/babyai_2agent_fixed20_8gpu_plain_split_notag_persist_v1`
- Collapse steps: `[400, 450, 500, 550, 600]`
