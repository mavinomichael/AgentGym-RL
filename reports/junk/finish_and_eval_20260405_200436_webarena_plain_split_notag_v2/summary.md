# WebArena 2-agent scaling 600 plain split no-tag clean v2

- Experiment: `webarena_2agent_scaling_600_8gpu_plain_split_notag_clean_v2`
- VM: `odion-agentgym-sweep-h100-apac-20260331-asia-southeast1-b` (`asia-southeast1-b`)
- Training status at check time: reached `global_step_600`
- Remote train log: `/home/mavinomichael/agentgym_runs/logs/webarena_2agent_scaling_600_8gpu_plain_split_notag_clean_v2/train_step200.log`
- Remote trace directory: `/home/mavinomichael/agentgym_runs/logs/webarena_2agent_scaling_600_8gpu_plain_split_notag_clean_v2/trace_train`
- Local artifact mirror: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/finish_and_eval_20260405_200436_webarena_plain_split_notag_v2/artifacts_full`
- Eval coverage: `11/12` checkpoints have completed eval logs; `step 300` produced all 50 rollout JSONs but its eval log is still missing the final summary block.

## Key findings

- Every evaluated checkpoint is collapsed: `Avg@1=0.0` and `Pass@1=0.0` at `50/100/150/200/250/350/400/450/500/550/600`, and `step 300` rollout rewards are also all zero.
- The dominant eval failure mode is planner-side collapse with executor follow-on failure: completed evals all report `ExecutorNativeFormatViolations=0.9`, `InvalidFormatTerminationRate=0.9`, `PlannerInvalidFormatRate=0.9`, `PlannerFallbackRate=0.9`, and `PlannerTagOnlyRate=0.9`.
- Collapse onset is visible in the training traces almost immediately: the earliest traced `planner_validation_reason=tag_only` packet appears at `training_step 2`.
- By the first saved checkpoint (`step 50`), both training and evaluation are already in the collapsed regime.
- Training-side metrics show a brief pocket of less-severe saturation around `step 200-300` (`planner_tag_only_rate` drops from `0.875` to `0.625` and `executor_first_pass_valid_rate` rises to `0.375` at `step 300`), but this never translates into non-zero eval reward.
- The trace corpus is overwhelmingly tag-only: `3954/3983` packets are `planner_validation_reason=tag_only`, with only `28` traced `ok` packets and those concentrated at `training_step 1`.

## Training snapshots

| Step | PlannerTagOnly | PlannerInvalid | PlannerFallback | ExecFormatViol | ExecFirstPassValid | ResponseLenMean | PlannerAvgTokenLen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.875000 | 0.875000 | 0.875000 | 0.875000 | 0.125000 | 0.875000 | 9.625000 |
| 100 | 0.875000 | 0.875000 | 0.875000 | 0.875000 | 0.125000 | 0.875000 | 9.625000 |
| 150 | 0.875000 | 0.875000 | 0.875000 | 0.875000 | 0.125000 | 0.875000 | 9.625000 |
| 200 | 0.750000 | 0.750000 | 0.750000 | 0.750000 | 0.250000 | 0.750000 | 8.250000 |
| 250 | 0.875000 | 0.875000 | 0.875000 | 0.875000 | 0.125000 | 0.875000 | 9.625000 |
| 300 | 0.625000 | 0.625000 | 0.625000 | 0.625000 | 0.375000 | 0.625000 | 6.875000 |
| 350 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 11.000000 |
| 400 | 0.875000 | 0.875000 | 0.875000 | 0.875000 | 0.125000 | 0.875000 | 9.625000 |
| 450 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 11.000000 |
| 500 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 11.000000 |
| 550 | 0.875000 | 0.875000 | 0.875000 | 0.875000 | 0.125000 | 0.875000 | 9.625000 |
| 600 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 11.000000 |

## Checkpoint comparison

| Step | Status | Avg@1 | Pass@1 | ExecFormatViol | InvalidFormatTerm | InvalidActionTerm | PlannerInvalid | PlannerFallback | PlannerTagOnly |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | complete | 0.000000 | 0.000000 | 0.900000 | 0.900000 | 0.000000 | 0.900000 | 0.900000 | 0.900000 |
| 100 | complete | 0.000000 | 0.000000 | 0.900000 | 0.900000 | 0.000000 | 0.900000 | 0.900000 | 0.900000 |
| 150 | complete | 0.000000 | 0.000000 | 0.900000 | 0.900000 | 0.000000 | 0.900000 | 0.900000 | 0.900000 |
| 200 | complete | 0.000000 | 0.000000 | 0.900000 | 0.900000 | 0.000000 | 0.900000 | 0.900000 | 0.900000 |
| 250 | complete | 0.000000 | 0.000000 | 0.900000 | 0.900000 | 0.000000 | 0.900000 | 0.900000 | 0.900000 |
| 300 | rollout_complete_log_hung | 0.000000 | 0.000000 | NA | NA | NA | NA | NA | NA |
| 350 | complete | 0.000000 | 0.000000 | 0.900000 | 0.900000 | 0.000000 | 0.900000 | 0.900000 | 0.900000 |
| 400 | complete | 0.000000 | 0.000000 | 0.900000 | 0.900000 | 0.000000 | 0.900000 | 0.900000 | 0.900000 |
| 450 | complete | 0.000000 | 0.000000 | 0.900000 | 0.900000 | 0.000000 | 0.900000 | 0.900000 | 0.900000 |
| 500 | complete | 0.000000 | 0.000000 | 0.900000 | 0.900000 | 0.000000 | 0.900000 | 0.900000 | 0.900000 |
| 550 | complete | 0.000000 | 0.000000 | 0.900000 | 0.900000 | 0.000000 | 0.900000 | 0.900000 | 0.900000 |
| 600 | complete | 0.000000 | 0.000000 | 0.900000 | 0.900000 | 0.000000 | 0.900000 | 0.900000 | 0.900000 |

## Artifacts

- Metrics TSV: `checkpoint_metrics.tsv`
- Metrics JSON: `checkpoint_metrics.json`
- Trace packet dossier: `trace_examples.md`
- Local eval/train artifacts: `artifacts_full/`
- Remote eval root: `/home/mavinomichael/agentgym_runs/logs/webarena_2agent_scaling_600_8gpu_plain_split_notag_clean_v2/eval_logs`
