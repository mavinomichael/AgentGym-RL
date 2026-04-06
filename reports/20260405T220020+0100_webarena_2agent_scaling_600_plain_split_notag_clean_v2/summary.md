# WebArena 2-Agent 600-Step Eval Summary

Run: `webarena_2agent_scaling_600_8gpu_plain_split_notag_clean_v2`


## Outcome
- Training had already reached `step 600`; no resume or relaunch was needed.
- Every saved checkpoint already had a completed eval log under `/home/mavinomichael/agentgym_runs/logs/webarena_2agent_scaling_600_8gpu_plain_split_notag_clean_v2/eval_logs`; no new evals were launched in this run.
- All twelve evaluated checkpoints (`50` through `600`) are identically collapsed: `Avg@1=0.0`, `Pass@1=0.0`, `ExecutorNativeFormatViolations=0.9`, `InvalidFormatTerminationRate=0.9`, `PlannerInvalidFormatRate=0.9`, `PlannerFallbackRate=0.9`, `PlannerTagOnlyRate=0.9`.
- There is no meaningful checkpoint ranking because the eval frontier is flat at zero across the entire save set.

## Collapse Timing
- First fully collapsed training step in copied traces: `step 2`.
- First selected concrete tag-only failure example: `step 3`.
- Sustained terminal-collapse regime in traces begins by `step 500` and remains present through `step 600`.
- The one clearly non-collapsed trace example in the copied selection is from `step 1`, where the planner gives grounded navigation advice and the executor emits a valid WebArena action block.

## Step 600 Health Signals
- `executor_invalid_action_rate=n/a`
- `executor_native_format_violations=1.000`
- `planner_invalid_format_rate=1.000`
- `planner_fallback_rate=1.000`
- `planner_tag_only_rate=1.000`
- `reward_events/mean=0.000`
- `critic/task_score/mean=0.000`
- `timing_s/save_checkpoint=38.067`

## Local Artifacts
- Summary: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/20260405T220020+0100_webarena_2agent_scaling_600_plain_split_notag_clean_v2/summary.md`
- Checkpoint metrics TSV: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/20260405T220020+0100_webarena_2agent_scaling_600_plain_split_notag_clean_v2/checkpoint_metrics.tsv`
- Checkpoint metrics JSON: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/20260405T220020+0100_webarena_2agent_scaling_600_plain_split_notag_clean_v2/checkpoint_metrics.json`
- Trace-focused markdown: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/20260405T220020+0100_webarena_2agent_scaling_600_plain_split_notag_clean_v2/selected_trace_examples.md`
- Copied key eval artifacts: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/20260405T220020+0100_webarena_2agent_scaling_600_plain_split_notag_clean_v2/key_eval_artifacts`
- Copied training trace snapshot: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/20260405T220020+0100_webarena_2agent_scaling_600_plain_split_notag_clean_v2/trace_train`
- Remote trace directory reference: `/home/mavinomichael/agentgym_runs/logs/webarena_2agent_scaling_600_8gpu_plain_split_notag_clean_v2/trace_train`

## Interpretation
- This run appears to be collapsed almost immediately in training rather than exhibiting a late-emerging degradation after an extended healthy regime.
- The copied traces show the dominant failure mode is planner tag-only garbage (`!!!!!!!!!!!!!!!!...`) triggering fallback guidance, followed by executor outputs that never satisfy the WebArena native action contract.
- Because eval is flat-zero at every checkpoint, the paper-facing comparison is about collapse persistence and failure mode, not about selecting a best-performing checkpoint.
