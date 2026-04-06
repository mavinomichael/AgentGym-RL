# 3-Agent BabyAI Reviewer Run: Collapse Analysis

## Run
- Experiment: `babyai_3agent_executor_reviewer_scaling_600_8gpu_no_tags_v1`
- Metrics table: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/20260403_130617_babyai_3agent_executor_reviewer_scaling_600_8gpu_no_tags_v1/checkpoint_metrics.tsv`
- Summary: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/20260403_130617_babyai_3agent_executor_reviewer_scaling_600_8gpu_no_tags_v1/summary.md`
- Selected traces: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/20260403_130617_babyai_3agent_executor_reviewer_scaling_600_8gpu_no_tags_v1/selected_trace_examples.md`
- Full local trace copy: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/20260403_130617_babyai_3agent_executor_reviewer_scaling_600_8gpu_no_tags_v1/trace_train`
- Eval/train logs: `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/20260403_130617_babyai_3agent_executor_reviewer_scaling_600_8gpu_no_tags_v1/logs`

## Headline Results
- Best checkpoint: `step 100`
  - `Avg@1 = 0.691801`
  - `Pass@1 = 0.722222`
- Best later recovery checkpoint: `step 350`
  - `Avg@1 = 0.668654`
  - `Pass@1 = 0.711111`
- Terminal collapse checkpoints: `450`, `500`, `550`, `600`
  - `Avg@1 = -0.200000`
  - `Pass@1 = 0.000000`
  - `ExecutorNativeFormatViolations = 1.000000`
  - `InvalidFormatTerminationRate = 1.000000`
  - `PlannerInvalidFormatRate = 1.000000`
  - `PlannerTagOnlyRate = 1.000000`

## What The Reviewer Helped With
- The reviewer did not produce a monotonically better curve, but it appears to have delayed the terminal collapse relative to the 2-agent dense no-tag run.
- The run had a good early checkpoint at `100` and a meaningful recovery band at `250-350`.
- At `300-350`, executor format errors were very low even though planner invalid-format remained high.

## What Went Wrong
The reviewer introduced a new failure surface instead of cleanly fixing the executor.

### 1. Early instability appears at step 150
At `step 150`, the run falls sharply:
- `Pass@1 = 0.211111`
- `ExecutorNativeFormatViolations = 0.677778`
- `PlannerInvalidFormatRate = 0.066667`

This means the collapse is not initially planner-driven. The reviewer/executor interface is already unstable.

Representative trace:
- `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/20260403_130617_babyai_3agent_executor_reviewer_scaling_600_8gpu_no_tags_v1/selected_trace_examples.md`
- Section: `Step 150 reviewer-triggered false retry / early instability`

Key pattern:
- the reviewer emits malformed schema (`PASS\n\nReason: Valid format` instead of strict `Verdict:` / `Reason:`)
- the system treats this as retry-worthy reviewer output
- the executor output is still semantically extractable, but the episode is terminated anyway

### 2. The run partially recovers
At `250-350`, the run recovers strongly:
- `step 250 Pass@1 = 0.611111`
- `step 300 Pass@1 = 0.677778`
- `step 350 Pass@1 = 0.711111`

This is important because it shows the run can recover from early instability.

### 3. Reviewer schema begins leaking into executor output by step 350
Representative trace:
- `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/20260403_130617_babyai_3agent_executor_reviewer_scaling_600_8gpu_no_tags_v1/selected_trace_examples.md`
- Section: `Step 350 recovered performance with schema leakage in executor output`

Key pattern:
- executor output contains both valid BabyAI `Thought:` / `Action:` and leaked reviewer text:
  - `Verdict:`
  - `Reason:`
- deterministic validation still extracts the action, so task success stays high
- but this is a clear contamination warning sign

### 4. Terminal collapse begins at step 400
At `step 400`:
- `Pass@1 = 0.422222`
- `PlannerInvalidFormatRate = 0.700000`
- `ExecutorNativeFormatViolations = 0.177778`

Representative trace:
- `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/20260403_130617_babyai_3agent_executor_reviewer_scaling_600_8gpu_no_tags_v1/selected_trace_examples.md`
- Section: `Step 400 onset of terminal collapse`

Key pattern:
- planner validation reason becomes `contains_role_or_schema_tokens`
- planner messages shrink into brittle imperative fragments like `Check options now`
- reviewer still often passes the executor output

Interpretation:
- the reviewer is no longer a reliable guardrail once planner contamination rises

### 5. Full collapse at step 450+
At `450`, `500`, `550`, and `600`:
- `Pass@1 = 0.0`
- `ExecutorNativeFormatViolations = 1.0`
- `PlannerInvalidFormatRate = 1.0`
- `PlannerTagOnlyRate = 1.0`

Representative trace:
- `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/20260403_130617_babyai_3agent_executor_reviewer_scaling_600_8gpu_no_tags_v1/selected_trace_examples.md`
- Section: `Step 450 terminal collapse example`

Key pattern:
- planner output degrades into tag-only / degenerate fragments
- executor emits punctuation spam
- reviewer emits punctuation spam too
- all three channels are now collapsed

## Answers To The Paper Questions

### When does collapse begin?
- Earliest instability: `step 150`
- Terminal-collapse onset: `step 400`
- Full collapse: `step 450`

### How do we detect it has started?
Watch for this sequence:
1. reviewer schema failures (`invalid_reviewer_schema`)
2. executor outputs that include reviewer schema text
3. planner invalid-format rising sharply
4. planner tag-only or garbage outputs
5. executor invalid-format saturating to `1.0`

### Can you recover from it?
- Yes, from early instability: this run recovered from the `150-200` crash band to strong checkpoints at `300-350`
- No evidence of recovery after full collapse: once `450+` is reached, the remaining checkpoints stay at `Pass@1 = 0.0`

### How do we prevent it?
This run suggests:
- reviewer prompts must be schema-stable and simpler than the current version
- reviewer outputs need deterministic post-validation before they can influence retry decisions
- reviewer-schema leakage into executor output should be treated as an explicit collapse warning signal
- checkpoint selection should stop training near the last stable band (`100` or `350`) rather than assuming later steps will improve

## Practical Paper Claim
The 3-agent reviewer ablation does **not** solve collapse. It changes the failure mode:
- early collapse becomes reviewer/executor schema disagreement
- later collapse becomes full three-channel contamination
