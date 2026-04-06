# BabyAI 2-Agent Scaling Retry v2 Eval Summary

- Generated: 2026-03-22T22:22:41
- VM: `odion-agentgym-sweep-w3-h100-as1c` (`asia-southeast1-c`)
- Experiment: `babyai_2agent_scaling_100_8gpu_plain_split_retry_v2`
- Training status: checkpoint `global_step_500` exists.
- Evaluation source: existing remote eval logs in `/home/mavinomichael/agentgym_runs/logs/babyai_2agent_scaling_100_8gpu_plain_split_retry_v2/`.
- Eval setup reused by the run matches the existing 2-agent scaling eval path: `python -m verl.multi_agent.main_generation` with `task=babyai`, `runtime=qwen2_5_7b_1gpu_smoke`, `data.batch_size=8`, `rollout.max_num_seqs=16`, planner/executor topology, and the same invalid-output retry settings visible in the experiment logs/scripts.

## Coverage

- Completed eval logs found for 14 checkpoints: `50`, `60`, `70`, `80`, `90`, `100`, `150`, `200`, `250`, `300`, `350`, `400`, `450`, `500`.
- No additional evals were launched in this run because every saved non-corrupt checkpoint already had a complete metric block.

## Key Findings

- Best checkpoint by both `Avg@1` and `Pass@1`: step `200` with `Avg@1=0.755230` and `Pass@1=0.811111`.
- Strong late-stage checkpoint before collapse: step `300` with `Avg@1=0.723644` and `Pass@1=0.788889`.
- Planner formatting stabilized after step `150`: `PlannerInvalidFormatRate` and `PlannerFallbackRate` are `0.0` from step `200` onward.
- Executor-side degradation appears late: step `400` has `InvalidActionTerminationRate=0.244444`, then steps `450` and `500` fully collapse with `Avg@1=-0.200000`, `Pass@1=0.000000`, and `ExecutorNativeFormatViolations=1.000000`.
- Latest checkpoint: step `500` is tied for worst `Avg@1` with step `450` at `-0.200000`.

## Checkpoint Table

| Step | Avg@1 | Pass@1 | ExecFmtViol | InvalidFmtTerm | InvalidActionTerm | PlannerInvalidFmt | PlannerFallback | PlannerTagOnly |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 0.460722 | 0.488889 | 0.044444 | 0.044444 | 0.022222 | 0.133333 | 0.133333 | 0.000000 |
| 60 | 0.460576 | 0.477778 | 0.011111 | 0.011111 | 0.000000 | 0.266667 | 0.266667 | 0.000000 |
| 70 | 0.229416 | 0.255556 | 0.000000 | 0.000000 | 0.044444 | 0.244444 | 0.244444 | 0.000000 |
| 80 | 0.332182 | 0.344444 | 0.011111 | 0.011111 | 0.044444 | 0.466667 | 0.466667 | 0.000000 |
| 90 | 0.374486 | 0.411111 | 0.000000 | 0.000000 | 0.044444 | 0.444444 | 0.444444 | 0.000000 |
| 100 | 0.404596 | 0.433333 | 0.000000 | 0.000000 | 0.000000 | 0.377778 | 0.377778 | 0.000000 |
| 150 | 0.624248 | 0.688889 | 0.133333 | 0.133333 | 0.011111 | 0.011111 | 0.011111 | 0.000000 |
| 200 | 0.755230 | 0.811111 | 0.044444 | 0.044444 | 0.022222 | 0.000000 | 0.000000 | 0.000000 |
| 250 | 0.690505 | 0.744444 | 0.044444 | 0.044444 | 0.011111 | 0.000000 | 0.000000 | 0.000000 |
| 300 | 0.723644 | 0.788889 | 0.033333 | 0.033333 | 0.077778 | 0.000000 | 0.000000 | 0.000000 |
| 350 | 0.634342 | 0.711111 | 0.000000 | 0.000000 | 0.200000 | 0.000000 | 0.000000 | 0.000000 |
| 400 | 0.546904 | 0.633333 | 0.066667 | 0.066667 | 0.244444 | 0.000000 | 0.000000 | 0.000000 |
| 450 | -0.200000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 500 | -0.200000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Source Paths

- Checkpoints: `/home/mavinomichael/agentgym_runs/saves/agentgym_multi_agent/babyai_2agent_scaling_100_8gpu_plain_split_retry_v2/`
- Eval logs: `/home/mavinomichael/agentgym_runs/logs/babyai_2agent_scaling_100_8gpu_plain_split_retry_v2/`
