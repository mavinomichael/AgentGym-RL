# BabyAI Fixed-20 No-Tag Checkpoint Comparison

Experiment: `babyai_2agent_fixed20_8gpu_plain_split_notag_persist_v1`

Report scope:
- Reused completed eval logs for checkpoints `50`, `100`, `150`, and `200`.
- Completed missing evals for checkpoints `250`, `300`, `350`, `400`, `450`, `500`, `550`, and `600`.
- Evaluation setup matched the experiment family: 2-agent BabyAI planner/executor, plain-split no-tag prompting, no ScalingRL curriculum, fixed 20-round interaction budget.

## Summary

Best checkpoint by `Pass@1` was `step 100` with `Avg@1=0.6398642710513539` and `Pass@1=0.6777777777777778`.

Recovery after the dip at `step 200` was limited:
- `step 250`: `Avg@1=0.47852161195543075`, `Pass@1=0.5`
- `step 300`: `Avg@1=0.4691121591462029`, `Pass@1=0.4888888888888889`
- `step 350`: `Avg@1=0.489063972234726`, `Pass@1=0.5111111111111111`

Late checkpoints collapsed sharply beginning at `step 400`:
- `steps 400/450/500/550/600` all hit floor reward with `Avg@1=-0.20000000298023224` and `Pass@1=0.0`.
- Those same checkpoints also show total format failure: `ExecutorNativeFormatViolations=1.0`, `InvalidFormatTerminationRate=1.0`, `PlannerInvalidFormatRate=1.0`, and `PlannerTagOnlyRate=1.0`.

## Notable Patterns

- Early checkpoints still had some planner/executor formatting noise, but they remained functional.
- `step 100` is the clear peak among all evaluated checkpoints.
- `steps 250-350` are cleaner than earlier checkpoints on formatting metrics, but they do not recover the `step 100` performance peak.
- The terminal collapse from `step 400` onward is not just executor-side; planner validity also fails completely.

## Artifacts

- TSV metrics: `checkpoint_metrics.tsv`
- Remote eval logs live under `/home/mavinomichael/agentgym_runs/logs/babyai_2agent_fixed20_8gpu_plain_split_notag_persist_v1/`
