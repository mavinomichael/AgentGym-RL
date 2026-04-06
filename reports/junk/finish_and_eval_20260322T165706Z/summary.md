# BabyAI 2-agent scaling eval status

- Experiment: `babyai_2agent_scaling_100_8gpu_plain_split_retry_v2`
- VM: `odion-agentgym-sweep-w3-h100-as1c` (`asia-southeast1-c`)
- Training threshold check: reached `step500`
- Saved checkpoints found: `50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500`
- Historical evals already complete before this run: `50, 60, 70, 80, 90, 100`
- Additional work completed in this run:
  - merged checkpoints `150, 200, 250, 300, 350, 400, 450, 500`
  - completed eval for checkpoint `150`
  - started sequential eval backlog from `200` upward; `200` is currently running on the VM

## Current comparison

Completed checkpoints so far, sorted by `Avg@1`:

1. `150`: `Avg@1 0.624248`, `Pass@1 0.688889`, `ExecutorNativeFormatViolations 0.133333`, `PlannerInvalidFormatRate 0.011111`
2. `50`: `Avg@1 0.460722`, `Pass@1 0.488889`, `ExecutorNativeFormatViolations 0.044444`, `PlannerInvalidFormatRate 0.133333`
3. `60`: `Avg@1 0.460576`, `Pass@1 0.477778`, `ExecutorNativeFormatViolations 0.011111`, `PlannerInvalidFormatRate 0.266667`
4. `100`: `Avg@1 0.404596`, `Pass@1 0.433333`, `ExecutorNativeFormatViolations 0.000000`, `PlannerInvalidFormatRate 0.377778`
5. `90`: `Avg@1 0.374486`, `Pass@1 0.411111`, `ExecutorNativeFormatViolations 0.000000`, `PlannerInvalidFormatRate 0.444444`
6. `80`: `Avg@1 0.332182`, `Pass@1 0.344444`, `ExecutorNativeFormatViolations 0.011111`, `PlannerInvalidFormatRate 0.466667`
7. `70`: `Avg@1 0.229416`, `Pass@1 0.255556`, `ExecutorNativeFormatViolations 0.000000`, `PlannerInvalidFormatRate 0.244444`

## Notes

- `step150` is the current best checkpoint by both `Avg@1` and `Pass@1` among completed evals.
- `step150` also sharply reduced planner invalid-format and fallback rates relative to `100`.
- The sequential eval loop for `200, 250, 300, 350, 400, 450, 500` was not finished within this run window.
- The active launcher session during this snapshot was still progressing through `step200`.
