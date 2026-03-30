# Dense500 Eval And Trace Archive

Artifacts copied locally for the run `babyai_2agent_scaling_500_8gpu_plain_split_retry_dense_v1`.

## Contents
- Training launcher log: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/2026-03-23_dense500_eval_and_traces/agentgym_train_dense500.launch.log`
- Training/eval/merge logs: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/2026-03-23_dense500_eval_and_traces/logs`
- Full training traces: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/2026-03-23_dense500_eval_and_traces/trace_train`
- Checkpoint metrics TSV: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/2026-03-23_dense500_eval_and_traces/checkpoint_metrics.tsv`
- Checkpoint metrics JSON: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/2026-03-23_dense500_eval_and_traces/checkpoint_metrics.json`
- Selected trace examples: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/2026-03-23_dense500_eval_and_traces/selected_trace_examples.md` and `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/2026-03-23_dense500_eval_and_traces/selected_trace_examples.json`

## Eval Summary

| Step | Avg@1 | Pass@1 | Exec Format Violations | Invalid Format | Invalid Action | Planner Invalid | Planner Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.1123 | 0.1222 | 0.0000 | 0.0000 | 0.0111 | 0.1333 | 0.0000 |
| 100 | 0.0872 | 0.1000 | 0.2444 | 0.2444 | 0.0333 | 0.4111 | 0.0000 |
| 150 | 0.2661 | 0.2889 | 0.1444 | 0.1444 | 0.0444 | 0.5556 | 0.0111 |
| 200 | 0.4405 | 0.4778 | 0.1444 | 0.1444 | 0.0111 | 0.3556 | 0.0000 |
| 250 | 0.5340 | 0.5889 | 0.2111 | 0.2111 | 0.1333 | 0.4333 | 0.0000 |
| 300 | 0.7743 | 0.8222 | 0.0333 | 0.0333 | 0.0000 | 0.0000 | 0.0000 |
| 350 | 0.6813 | 0.7222 | 0.0000 | 0.0000 | 0.0111 | 0.0111 | 0.0000 |
| 400 | 0.3312 | 0.4000 | 0.2222 | 0.2222 | 0.0333 | 0.1889 | 0.0000 |
| 450 | -0.1757 | 0.0222 | 0.9778 | 0.9778 | 0.0000 | 0.0000 | 0.0000 |
| 500 | -0.2000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |

- Best checkpoint: `step 300` with `Pass@1=0.8222` and `Avg@1=0.7743`.
- Worst checkpoint: `step 500` with `Pass@1=0.0000` and `Avg@1=-0.2000`.
- Performance rises through `250-350`, peaks at `300`, degrades at `400`, and collapses at `450-500`.
- At `450-500`, planner outputs remain formally valid while executor format failure dominates, so the terminal failure is executor-side.

## Recommended Analysis Focus
- Compare `selected_trace_examples.md` across `300`, `400`, `450`, and `500`.
- Inspect whether executor retries repair early invalid outputs at `400` and why they fail completely at `450-500`.
- Cross-reference `checkpoint_metrics.tsv` with the full `trace_train/*.jsonl` traces.
