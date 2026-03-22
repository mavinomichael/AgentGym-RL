# BabyAI 2-agent scaling 100 plain split retry v2

- Experiment: `babyai_2agent_scaling_100_8gpu_plain_split_retry_v2`
- VM: `odion-agentgym-sweep-w3-h100-as1c` (`asia-southeast1-c`)
- Training status at check time: reached `global_step_500`
- Evaluated checkpoints completed for all saved steps with logs at `/home/mavinomichael/agentgym_runs/logs/babyai_2agent_scaling_100_8gpu_plain_split_retry_v2/`
- Rerun work performed in this pass: `step 300` was allowed to finish in-flight; `350/400/450/500` were completed afterward. A GPU 0 OOM from unrelated occupancy was avoided by pinning the remaining evals to GPU 1 with the same experiment eval flags.

## Key findings

- Best checkpoint by both `Avg@1` and `Pass@1` is `step 200`: `Avg@1=0.755230`, `Pass@1=0.811111`.
- `step 300` remained strong: `Avg@1=0.723644`, `Pass@1=0.788889`, with low planner/executor format failure rates.
- Performance falls after `step 300`: `step 350` drops to `Avg@1=0.634342`, `Pass@1=0.711111`, driven mostly by `InvalidActionTerminationRate=0.200000`.
- `step 400` degrades further: `Avg@1=0.546904`, `Pass@1=0.633333`, `InvalidActionTerminationRate=0.244444`, `ExecutorNativeFormatViolations=0.066667`.
- `step 450` and `step 500` collapse completely with `Avg@1=-0.200000`, `Pass@1=0.000000`, `ExecutorNativeFormatViolations=1.000000`, and `InvalidFormatTerminationRate=1.000000`.
- Planner-side validity improves after `step 150`: `PlannerInvalidFormatRate`, `PlannerFallbackRate`, and `PlannerTagOnlyRate` are all `0.0` from `step 200` onward, including the collapsed `450/500` checkpoints. The late-stage failure is therefore executor-side rather than planner fallback.

## Full comparison

| Step | Avg@1 | Pass@1 | ExecFormatViol | InvalidFormatTerm | InvalidActionTerm | PlannerInvalid | PlannerFallback | PlannerTagOnly |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
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

## Artifacts

- TSV metrics: `checkpoint_metrics.tsv`
- Raw remote eval logs: `/home/mavinomichael/agentgym_runs/logs/babyai_2agent_scaling_100_8gpu_plain_split_retry_v2/`
