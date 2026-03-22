# Mavino Collapse Report

## Run
- Run: `babyai_2agent_scaling_100_8gpu`
- Topology: `planner_executor`
- Intended schedule: `scaling_inter_stepwise` with rounds `[6,13,20]` and `steps_scaling_inter=100`
- Checkpoints evaluated: `step 50`, `step 100`

## Main Finding
This run is a clean example of what we can label the **Mavino Collapse**: the model remains competent through the mid-run checkpoint, then degrades sharply late in training as planner outputs start leaking control/prompt scaffolding into executor prompts and the executor begins copying that scaffolding instead of producing BabyAI-native `Thought:` / `Action:` responses.

## Critical Confound
The run never advanced beyond the first scaling phase. `max_rounds` stayed at `6.0` through `step 100`. So this experiment does **not** test transitions to `13` or `20` rounds. The collapse happened while still operating at the 6-round regime.

## Eval Results
- `step 50`: `Avg@1=0.7442706210745705`, `Pass@1=0.8`, `ExecutorNativeFormatViolations=0.0`, `PlannerInvalidFormatRate=0.0`
- `step 100`: `Avg@1=-0.16204471704032686`, `Pass@1=0.03333333333333333`, `ExecutorNativeFormatViolations=0.8`, `PlannerInvalidFormatRate=0.9888888888888889`

The step-50 checkpoint is healthy. The step-100 checkpoint is collapsed.

## Collapse Timing
Using simple threshold heuristics on training metrics:
- Warning onset: `step 90`
  - criterion: planner invalid rate `>= 0.5` and executor invalid rate `>= 0.25` or executor format violations `> 0`
- Hard collapse: `step 94`
  - criterion: task score `< 0` and executor format violations `>= 0.5`
- Terminal collapse: `step 98`
  - criterion: planner invalid rate `>= 1.0`, executor invalid rate `>= 0.9`, executor format violations `>= 0.8`

These thresholds are heuristic, but they line up with the trace evidence and with the step-50 vs step-100 eval gap.

## Phase Summary
| Phase | Steps | Mean Task Score | Mean Exec Invalid | Mean Exec Format Violations | Mean Planner Invalid | Mean Planner Rewrite |
|---|---:|---:|---:|---:|---:|---:|
| healthy_1_50 | 1-50 | 0.561 | 0.021 | 0.002 | 0.180 | 0.180 |
| stable_51_80 | 51-80 | 0.599 | 0.035 | 0.000 | 0.060 | 0.060 |
| warning_81_89 | 81-89 | 0.477 | 0.090 | 0.007 | 0.396 | 0.396 |
| onset_90_93 | 90-93 | 0.319 | 0.250 | 0.093 | 0.563 | 0.563 |
| collapse_94_100 | 94-100 | -0.054 | 0.804 | 0.741 | 0.714 | 0.714 |

## Mechanism Suggested By The Traces
1. Planner drift exists early, but is tolerable.
   - Planner invalid/rewrite pressure is present even in the healthy phase, but executor format remains clean and the environment still gets legal actions.
2. The first clear warning is planner prompt leakage.
   - Around steps 81-93, planner outputs begin to include control tags like `[Planner]`, `[Executor Check]`, and truncated scaffolding.
   - Rewrites increase sharply in this window.
3. Hard collapse begins when the executor starts copying the leaked scaffolding.
   - Around steps 94-100, executor outputs switch from BabyAI-native `Thought:` / `Action:` responses to strings like `[Executor Review] ...`, `[Executor Turn] ...`, or repeated prompt fragments.
4. Terminal collapse is recursive prompt contamination.
   - By steps 98-100, both planner and executor are echoing role headers and reviewer scaffolding back into the generation stream, producing invalid-format outputs and invalid actions at very high rates.

## Representative Trace Evidence
> The earliest observed prompt-header leak occurs at `step 55`. In that trace, the planner emits `[Planner] Check room for ball.` even though the planner prompt forbids role labels. The normalization layer strips the header before the message is passed to the executor, so executor behavior remains valid at this point. This example is important because it marks the first visible contamination event, while still preceding the later collapse regime where prompt scaffolding propagates into executor outputs. See `first_header_leak_step55` in `representative_trace_examples.md`.

See `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/babyai_multi_agent_diagnostics_2026-03-21/mavino_collapse_2agent_scaling_100/representative_trace_examples.md`.

Key examples:
- `step 50`: healthy executor output
  - planner: `Approach the grey closed door.`
  - executor: `Thought: ... Action: toggle and go through grey closed door 1`
- `step 90`: early warning
  - planner emits `[Planner]` / `[Executor Check]` style scaffolding and requires rewrite
  - executor starts echoing prompt scaffolding instead of only task-grounded content
- `step 94`: hard collapse
  - executor emits reviewer-style tags such as `[Executor Review] ...` instead of BabyAI native format
- `step 98` and `step 100`: terminal collapse
  - executor output contains repeated `[Executor Turn]` / `[Reviewers Turn]` fragments, indicating recursive prompt contamination

## Additional Notes
- `executor_reviewer_pass_rate` is not informative in this 2-agent run because there is no reviewer role active; it remains `1.0`.
- `planner_fallback_rate` remained `0.0`. This was not a fallback-driven collapse. It was a raw-output contamination and format-drift collapse.
- The corrected eval rerun used the 90-item BabyAI test set under `AgentEval/babyai/babyai_test.json`, not the stale 8-item legacy file.

## Saved Artifacts
- Raw logs: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/babyai_multi_agent_diagnostics_2026-03-21/mavino_collapse_2agent_scaling_100/raw_logs/babyai_2agent_scaling_100_8gpu`
- Per-step metrics: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/babyai_multi_agent_diagnostics_2026-03-21/mavino_collapse_2agent_scaling_100/step_metrics.tsv`
- Collapse window metrics: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/babyai_multi_agent_diagnostics_2026-03-21/mavino_collapse_2agent_scaling_100/collapse_window_steps_81_100.tsv`
- Eval summary: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/babyai_multi_agent_diagnostics_2026-03-21/mavino_collapse_2agent_scaling_100/eval_summary.tsv`
- Phase summary: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/babyai_multi_agent_diagnostics_2026-03-21/mavino_collapse_2agent_scaling_100/phase_summary.tsv`
- Trace phase summary: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/babyai_multi_agent_diagnostics_2026-03-21/mavino_collapse_2agent_scaling_100/trace_phase_summary.tsv`
- Representative traces: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/babyai_multi_agent_diagnostics_2026-03-21/mavino_collapse_2agent_scaling_100/representative_trace_examples.md`
- Representative traces JSON: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/babyai_multi_agent_diagnostics_2026-03-21/mavino_collapse_2agent_scaling_100/representative_trace_examples.json`
