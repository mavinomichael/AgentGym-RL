# BabyAI 2-Agent Scaling 100 Plain Split Retry v2

Date: 2026-03-22 20:20:51 +0100

## Context

- VM: `odion-agentgym-sweep-w3-h100-as1c`
- Zone: `asia-southeast1-c`
- Experiment: `babyai_2agent_scaling_100_8gpu_plain_split_retry_v2`
- Remote save root: `/home/mavinomichael/agentgym_runs/saves/agentgym_multi_agent/babyai_2agent_scaling_100_8gpu_plain_split_retry_v2`
- Remote log root: `/home/mavinomichael/agentgym_runs/logs/babyai_2agent_scaling_100_8gpu_plain_split_retry_v2`

## Training Status

- Highest completed checkpoint found: `500`
- Numeric checkpoints present: `50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500`
- Extra non-numeric save artifact detected and excluded from comparison: `global_step_350_corrupt_20260322111159`

## Evaluation Status

- Completed eval logs already existed for every numeric checkpoint listed above.
- No new evaluations were required for this automation run.
- The eval configuration in the existing logs used the same `verl.multi_agent.main_generation` setup already established for this experiment family, including:
  - `task=babyai`
  - `runtime=qwen2_5_7b_1gpu_smoke`
  - `data.batch_size=8`
  - invalid-output retry policy with planner retries enabled

## Checkpoint Comparison

Best checkpoint:
- `step 200` with `Avg@1=0.755230` and `Pass@1=0.811111`

Trend summary:
- `50 -> 100`: unstable early phase dominated by planner invalid-format and fallback rates.
- `150 -> 300`: strongest region; planner failure rates drop to near zero and task metrics peak at `step 200`.
- `350 -> 400`: quality degrades, with invalid action terminations rising sharply.
- `450 -> 500`: complete executor-format collapse, with `ExecutorNativeFormatViolations=1.0` and `Pass@1=0.0`.

Most important rows:

| Step | Avg@1 | Pass@1 | ExecFmtViol | InvalidFmtTerm | InvalidActionTerm | PlannerInvalidFmt | PlannerFallback | PlannerTagOnly |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 150 | 0.624248 | 0.688889 | 0.133333 | 0.133333 | 0.011111 | 0.011111 | 0.011111 | 0.000000 |
| 200 | 0.755230 | 0.811111 | 0.044444 | 0.044444 | 0.022222 | 0.000000 | 0.000000 | 0.000000 |
| 300 | 0.723644 | 0.788889 | 0.033333 | 0.033333 | 0.077778 | 0.000000 | 0.000000 | 0.000000 |
| 400 | 0.546904 | 0.633333 | 0.066667 | 0.066667 | 0.244444 | 0.000000 | 0.000000 | 0.000000 |
| 450 | -0.200000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 500 | -0.200000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Files

- Machine-readable metrics: `stage_metrics.tsv`
- This summary: `summary.md`
