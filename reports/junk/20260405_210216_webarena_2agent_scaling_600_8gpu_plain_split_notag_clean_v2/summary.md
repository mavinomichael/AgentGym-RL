# WebArena 2-agent scaling-600 checkpoint comparison

- Generated: 2026-04-05 21:05:21 WAT
- VM: `odion-agentgym-sweep-h100-apac-20260331-asia-southeast1-b` (`asia-southeast1-b`)
- Experiment: `webarena_2agent_scaling_600_8gpu_plain_split_notag_clean_v2`
- Training status: reached `global_step_600`; no training process was active when checked
- Eval action taken: no new evals launched, because every saved checkpoint from `50` through `600` already had a completed `eval_step*.log`
- Remote eval logs: `/home/mavinomichael/agentgym_runs/logs/webarena_2agent_scaling_600_8gpu_plain_split_notag_clean_v2/eval_logs`
- Remote trace directory: `/home/mavinomichael/agentgym_runs/logs/webarena_2agent_scaling_600_8gpu_plain_split_notag_clean_v2/trace_train`
- Local copied artifacts: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/20260405_210216_webarena_2agent_scaling_600_8gpu_plain_split_notag_clean_v2/artifacts/extracted`

## Summary

All evaluated checkpoints are tied at the floor. The best checkpoint by `Pass@1` is `step 50`, but it still has `Avg@1=0.0` and `Pass@1=0.0`.

Collapse is already present at the first evaluated checkpoint, `step 50`. Every checkpoint from `50` to `600` has the same aggregate eval signature: `ExecutorNativeFormatViolations=0.9`, `InvalidFormatTerminationRate=0.9`, `PlannerInvalidFormatRate=0.9`, `PlannerFallbackRate=0.9`, and `PlannerTagOnlyRate=0.9`, with `InvalidActionTerminationRate=0.0`.

The copied training traces support that diagnosis. Out of `3983` traced interactions, only `28` have a planner output that passes validation, and only `24` end with a final action that passes validation. The dominant failure mode in training traces is planner-side degeneration and fallback; no executor-native-format-invalid training trace was found in the copied sample.

In practical terms, the run shows a few early valid fragments, but those do not survive into any functional saved checkpoint. By the first checkpoint, the policy is already flatlined on WebArena and stays there through `step 600`.

## Checkpoint comparison

| Step | Avg@1 | Pass@1 | Exec fmt viol | Invalid fmt term | Invalid action term | Planner invalid | Planner fallback | Planner tag-only |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.0 | 0.0 | 0.9 | 0.9 | 0.0 | 0.9 | 0.9 | 0.9 |
| 100 | 0.0 | 0.0 | 0.9 | 0.9 | 0.0 | 0.9 | 0.9 | 0.9 |
| 150 | 0.0 | 0.0 | 0.9 | 0.9 | 0.0 | 0.9 | 0.9 | 0.9 |
| 200 | 0.0 | 0.0 | 0.9 | 0.9 | 0.0 | 0.9 | 0.9 | 0.9 |
| 250 | 0.0 | 0.0 | 0.9 | 0.9 | 0.0 | 0.9 | 0.9 | 0.9 |
| 300 | 0.0 | 0.0 | 0.9 | 0.9 | 0.0 | 0.9 | 0.9 | 0.9 |
| 350 | 0.0 | 0.0 | 0.9 | 0.9 | 0.0 | 0.9 | 0.9 | 0.9 |
| 400 | 0.0 | 0.0 | 0.9 | 0.9 | 0.0 | 0.9 | 0.9 | 0.9 |
| 450 | 0.0 | 0.0 | 0.9 | 0.9 | 0.0 | 0.9 | 0.9 | 0.9 |
| 500 | 0.0 | 0.0 | 0.9 | 0.9 | 0.0 | 0.9 | 0.9 | 0.9 |
| 550 | 0.0 | 0.0 | 0.9 | 0.9 | 0.0 | 0.9 | 0.9 | 0.9 |
| 600 | 0.0 | 0.0 | 0.9 | 0.9 | 0.0 | 0.9 | 0.9 | 0.9 |

## Paper-analysis artifacts

- `checkpoint_metrics.tsv` and `checkpoint_metrics.json` in this folder for the comparison table source
- Copied remote `eval_logs/` under `artifacts/extracted/eval_logs/`, including per-checkpoint eval logs, failed attempts, and sweep driver logs
- Copied remote `trace_train/` under `artifacts/extracted/trace_train/` for deeper trace inspection
- Copied `train_step200.log` under `artifacts/extracted/train_step200.log` to preserve the run config and debug settings
- Trace-focused examples are summarized in `trace_examples.md`
