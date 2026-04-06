# WebArena 2-agent scaling 600 plain split no-tag clean v2

- Experiment: `webarena_2agent_scaling_600_8gpu_plain_split_notag_clean_v2`
- VM: `odion-agentgym-sweep-h100-apac-20260331-asia-southeast1-b` (`asia-southeast1-b`)
- Training status at check time: reached `global_step_600`
- Training checkpoint save status: `global_step_600` saved successfully
- Remote training trace directory: `/home/mavinomichael/agentgym_runs/logs/webarena_2agent_scaling_600_8gpu_plain_split_notag_clean_v2/trace_train`
- Local artifact bundle: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/webarena_2agent_scaling_600_8gpu_plain_split_notag_clean_v2_eval_20260405_185205`

## What was done in this pass

- Verified the VM was still up and the run had already reached step `600`.
- Confirmed the boot disk did not need resizing. The boot disk was already `8000 GB`, and checkpoint saving at `global_step_600` succeeded.
- Identified missing eval coverage for later checkpoints and repaired the missing actor `huggingface/` metadata for resumed evals by seeding tokenizer/config files from the merged step `600` actor export.
- Worked around a DeepSpeed-related merge failure by running a one-off merger on the remote VM that disables DeepSpeed detection during `save_pretrained`.
- Completed missing evals for checkpoints `250`, `350`, `400`, `450`, `500`, and `550`, while preserving the existing completed evals for `50`, `100`, `150`, `200`, and `600`.
- Observed that the `step 300` eval stayed live but looped on `Multi-agent rounds 1/15` without reaching final aggregation during this run.
- Copied the remote `trace_train` and `eval_logs` trees into the local report bundle for downstream paper analysis.

## Key findings

- Every completed checkpoint from `50` through `600` is flat at `Avg@1=0.0` and `Pass@1=0.0`.
- Every completed checkpoint also shares the same failure profile: `ExecutorNativeFormatViolations=0.9`, `InvalidFormatTerminationRate=0.9`, `PlannerInvalidFormatRate=0.9`, `PlannerFallbackRate=0.9`, and `PlannerTagOnlyRate=0.9`.
- The collapse onset is therefore not a late-training event. On the completed evidence, the run is already collapsed by the earliest evaluated checkpoint (`50`).
- Training traces show a brief structurally valid window at training step `1`, but degeneration appears almost immediately after that through planner fallback and executor format failure, followed by punctuation-only outputs by training steps `3-4`.
- The stalled `step 300` eval does not alter the conclusion materially because the adjacent completed checkpoints `250` and `350` are identical across all aggregate metrics.

## Full comparison

| Step | Status | Avg@1 | Pass@1 | ExecFormatViol | InvalidFormatTerm | InvalidActionTerm | PlannerInvalid | PlannerFallback | PlannerTagOnly |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | complete | 0.0 | 0.0 | 0.9 | 0.9 | 0.0 | 0.9 | 0.9 | 0.9 |
| 100 | complete | 0.0 | 0.0 | 0.9 | 0.9 | 0.0 | 0.9 | 0.9 | 0.9 |
| 150 | complete | 0.0 | 0.0 | 0.9 | 0.9 | 0.0 | 0.9 | 0.9 | 0.9 |
| 200 | complete | 0.0 | 0.0 | 0.9 | 0.9 | 0.0 | 0.9 | 0.9 | 0.9 |
| 250 | complete | 0.0 | 0.0 | 0.9 | 0.9 | 0.0 | 0.9 | 0.9 | 0.9 |
| 300 | incomplete_stuck | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 350 | complete | 0.0 | 0.0 | 0.9 | 0.9 | 0.0 | 0.9 | 0.9 | 0.9 |
| 400 | complete | 0.0 | 0.0 | 0.9 | 0.9 | 0.0 | 0.9 | 0.9 | 0.9 |
| 450 | complete | 0.0 | 0.0 | 0.9 | 0.9 | 0.0 | 0.9 | 0.9 | 0.9 |
| 500 | complete | 0.0 | 0.0 | 0.9 | 0.9 | 0.0 | 0.9 | 0.9 | 0.9 |
| 550 | complete | 0.0 | 0.0 | 0.9 | 0.9 | 0.0 | 0.9 | 0.9 | 0.9 |
| 600 | complete | 0.0 | 0.0 | 0.9 | 0.9 | 0.0 | 0.9 | 0.9 | 0.9 |

## Artifacts

- Metrics TSV: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/webarena_2agent_scaling_600_8gpu_plain_split_notag_clean_v2_eval_20260405_185205/checkpoint_metrics.tsv`
- Metrics JSON: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/webarena_2agent_scaling_600_8gpu_plain_split_notag_clean_v2_eval_20260405_185205/checkpoint_metrics.json`
- Selected trace examples: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/webarena_2agent_scaling_600_8gpu_plain_split_notag_clean_v2_eval_20260405_185205/selected_trace_examples.md`
- Copied remote artifacts: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/webarena_2agent_scaling_600_8gpu_plain_split_notag_clean_v2_eval_20260405_185205/artifacts`

## Next analysis focus

- Inspect the copied `artifacts/eval_logs/rollout_step250` tree if you want per-episode failure traces around the first repaired eval.
- Use the copied `artifacts/trace_train` directory together with `selected_trace_examples.md` to compare the short-lived structurally valid outputs against the immediate fallback and punctuation-collapse traces.
- If `step 300` needs a completed aggregate for completeness, restart only that eval in isolation and capture whether the loop at `rounds 1/15` is deterministic for that checkpoint.
