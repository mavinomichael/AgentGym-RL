# Selected Trace Examples

Experiment: `webarena_2agent_scaling_600_8gpu_plain_split_notag_clean_v2`

Remote training trace directory: `/home/mavinomichael/agentgym_runs/logs/webarena_2agent_scaling_600_8gpu_plain_split_notag_clean_v2/trace_train`

## Reading guide

- "Non-collapse" here means structurally valid planner and executor outputs in the trace, not successful evaluation outcomes.
- The eval metrics remain collapsed at every completed checkpoint, so these examples are useful for failure-shape analysis rather than success-case analysis.

## Relative non-collapse examples from training traces

### Example A: coherent early account-navigation plan

- Source: `executor_payload_trace_rank0.jsonl`
- Training step: `1`
- Item: `330`
- Round: `1`
- Planner valid: `true`
- Planner fallback used: `false`
- Executor native format valid: `true`
- Planner message: `Check the navigation menu for any option related to order history or customer account`
- Executor output excerpt: `In summary, the next action I will perform is click [1277].`
- Why it matters: the planner and executor can still produce structurally compliant actions at the very start of training.

### Example B: coherent order-history follow-up

- Source: `executor_payload_trace_rank0.jsonl`
- Training step: `1`
- Item: `330`
- Round: `2`
- Planner valid: `true`
- Planner fallback used: `false`
- Executor native format valid: `true`
- Planner message: `Click on 'My Orders' to view order history`
- Executor output excerpt: `In summary, the next action I will perform is click [4311].`
- Why it matters: multi-round role coordination is briefly intact before degeneration dominates.

### Example C: coherent item-selection step

- Source: `executor_payload_trace_rank0.jsonl`
- Training step: `1`
- Item: `330`
- Round: `3`
- Planner valid: `true`
- Planner fallback used: `false`
- Executor native format valid: `true`
- Planner message: `Find and click on an order from March 2023 to view its details`
- Executor output excerpt: `In summary, the next action I will perform is click [6447].`
- Why it matters: the trace still contains locally sensible action selection, but that does not translate into eval success later.

## Collapse examples from training traces

### Example D: planner fallback despite natural-language content

- Source: `executor_payload_trace_rank0.jsonl`
- Training step: `1`
- Item: `330`
- Round: `4`
- Planner valid: `false`
- Planner fallback used: `true`
- Planner raw output: `The order total is clearly displayed in the shopping cart. The amount spent in March 2023 is $285.29.`
- Planner injected fallback: `Planner guidance unavailable. Infer the next step from the observation only.`
- Executor native format valid: `true`
- Why it matters: the planner is already failing its own output contract even when it emits plausible English.

### Example E: executor natural language around a nominal action

- Source: `executor_payload_trace_rank0.jsonl`
- Training step: `1`
- Item: `330`
- Round: `7`
- Planner valid: `true`
- Planner fallback used: `false`
- Executor native format valid: `false`
- Planner message: `Wait for the page to load and then check the order details for the total spent`
- Executor output excerpt: `In summary, the next action I will perform is wait [3].`
- Validation reason: `invalid_format`
- Why it matters: the executor wraps actions in explanatory prose that violates the native action format, which is consistent with the high eval-time format-termination rate.

### Example F: fully degenerate punctuation-only outputs

- Source: `executor_payload_trace_rank0.jsonl`
- Training step: `3`
- Item: `35`
- Round: `1`
- Planner valid: `false`
- Planner fallback used: `true`
- Planner raw output: `!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`
- Executor native format valid: `false`
- Executor raw output: `!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!...`
- Validation reason: `invalid_format`
- Why it matters: by step `3`, the run already shows total output collapse rather than a gradual late-stage degradation.

### Example G: repeated punctuation-only collapse on another item

- Source: `executor_payload_trace_rank0.jsonl`
- Training step: `4`
- Item: `49`
- Round: `1`
- Planner valid: `false`
- Planner fallback used: `true`
- Planner raw output: `!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`
- Executor native format valid: `false`
- Executor raw output: `!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!...`
- Validation reason: `invalid_format`
- Why it matters: the degenerate pattern repeats across items immediately, which argues against a task-specific failure.

## Eval artifact note

- Completed eval checkpoints from `50` through `600` all share the same aggregate profile in this run: `Avg@1=0.0`, `Pass@1=0.0`, `ExecutorNativeFormatViolations=0.9`, `InvalidFormatTerminationRate=0.9`, `PlannerInvalidFormatRate=0.9`, `PlannerFallbackRate=0.9`, `PlannerTagOnlyRate=0.9`.
- `step 300` did not produce a completed aggregation log during this run. Its worker remained alive but kept repeating `Multi-agent rounds 1/15`.
- Because `250` and `350` are identical on every aggregate metric, the stalled `300` run does not change the collapse-onset conclusion from the completed checkpoints.
