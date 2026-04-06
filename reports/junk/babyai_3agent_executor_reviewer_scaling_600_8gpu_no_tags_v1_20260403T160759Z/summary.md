# BabyAI 3-Agent Executor Reviewer Scaling 600 Report

Generated: 2026-04-03 16:07:59 UTC

Source run: `/home/mavinomichael/agentgym_runs/saves/agentgym_multi_agent/babyai_3agent_executor_reviewer_scaling_600_8gpu_no_tags_v1`
Remote eval logs: `/home/mavinomichael/agentgym_runs/logs/babyai_3agent_executor_reviewer_scaling_600_8gpu_no_tags_v1`

## Outcome

- BabyAI training reached checkpoint `global_step_600`.
- Completed eval logs exist for every saved checkpoint from `50` through `600`.
- Best checkpoint by `Avg@1` and `Pass@1` is `step 100` with `Avg@1=0.691801` and `Pass@1=0.722222`.
- `step 350` is the strongest late-stage checkpoint with `Avg@1=0.668654` and `Pass@1=0.711111`.
- The run collapses fully starting at `step 450`; steps `450`, `500`, `550`, and `600` all produce `Avg@1=-0.200000`, `Pass@1=0.0`, `ExecutorNativeFormatViolations=1.0`, and `PlannerInvalidFormatRate=1.0`.
- The stale `/home/mavinomichael/agentgym_runs/logs/babyai_3agent_executor_reviewer_scaling_600_8gpu_no_tags_v1/finish_eval_state.tsv` does not reflect the completed late-step eval logs, but the log files themselves are complete.

## Checkpoint Table

| Step | Avg@1 | Pass@1 | ExecFmtViol | InvalidFmt | InvalidAction | PlannerInvalid | PlannerFallback | PlannerTagOnly |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.542954 | 0.566667 | 0.033333 | 0.044444 | 0.000000 | 0.522222 | 0.000000 | 0.000000 |
| 100 | 0.691801 | 0.722222 | 0.000000 | 0.000000 | 0.000000 | 0.477778 | 0.000000 | 0.000000 |
| 150 | 0.057160 | 0.211111 | 0.677778 | 0.744444 | 0.000000 | 0.066667 | 0.000000 | 0.000000 |
| 200 | 0.179398 | 0.311111 | 0.577778 | 0.644444 | 0.011111 | 0.244444 | 0.000000 | 0.000000 |
| 250 | 0.542972 | 0.611111 | 0.088889 | 0.088889 | 0.133333 | 0.344444 | 0.000000 | 0.000000 |
| 300 | 0.632004 | 0.677778 | 0.022222 | 0.022222 | 0.011111 | 0.500000 | 0.000000 | 0.000000 |
| 350 | 0.668654 | 0.711111 | 0.011111 | 0.011111 | 0.000000 | 0.511111 | 0.000000 | 0.000000 |
| 400 | 0.385532 | 0.422222 | 0.177778 | 0.177778 | 0.000000 | 0.700000 | 0.000000 | 0.000000 |
| 450 | -0.200000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 |
| 500 | -0.200000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 |
| 550 | -0.200000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 |
| 600 | -0.200000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 |

## WebArena Follow-up Status

- The follow-up run `webarena_3agent_executor_reviewer_scaling_600_8gpu_no_tags_v1` has no valid checkpoint yet and is not running.
- The first launch failed before checkpointing because the WebArena env server was not provisioned correctly.
- Current blockers on the VM: `/home/mavinomichael/AgentGym-RL/AgentGym/agentenv-webarena/.env` is missing, `OPENAI_API_KEY` is unset in the shell environment, and none of the required WebArena service endpoints are reachable on either localhost or the fallback `metis.lti.cs.cmu.edu` URLs.
- Relaunching WebArena now would fail with the same environment bootstrap error until those services and variables are restored.
