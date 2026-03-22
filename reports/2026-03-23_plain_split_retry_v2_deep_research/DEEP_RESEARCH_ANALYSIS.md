# BabyAI 2-Agent Plain-Split Retry v2: Training and Collapse Analysis

Generated: 2026-03-23

## Scope
This report consolidates the locally saved evaluation results for `babyai_2agent_scaling_100_8gpu_plain_split_retry_v2`, supplements them with selected trace examples pulled from the training trace directory, and frames the main findings in a form suitable for external deep research.

Primary local artifacts used:
- `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/babyai_2agent_scaling_100_8gpu_plain_split_retry_v2_eval_20260322_222241/checkpoint_metrics.tsv`
- `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/babyai_2agent_scaling_100_8gpu_plain_split_retry_v2_eval_20260322_222241/key_trace_examples_selected.json`
- `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/babyai_multi_agent_diagnostics_2026-03-21/mavino_collapse_2agent_scaling_100/MAVINO_COLLAPSE_REPORT.md`
- `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/babyai_multi_agent_diagnostics_2026-03-21/mavino_collapse_2agent_scaling_100/representative_trace_examples.md`

## Executive Summary
- The run does **not** reproduce the March 21 prompt-scaffolding failure mode early in training. `PlannerTagOnlyRate` stays at `0.0` for every evaluated checkpoint.
- The run improves substantially after the unstable 50-100 range and reaches its best checkpoint at `step 200` with `Avg@1=0.755230` and `Pass@1=0.811111`.
- The strongest band is `150-300`, where planner invalid/fallback rates drop to zero and task performance is high.
- Degradation begins after `300`, first as **invalid action selection** (`step 350-400`) rather than planner tag leakage.
- The terminal failure at `450-500` is a new executor-side collapse: the executor emits plain planner-style text such as `use interact with wall 1` instead of the BabyAI `Thought:/Action:` schema. This yields `ExecutorNativeFormatViolations=1.0` and `Pass@1=0.0`.
- Relative to the earlier “Mavino Collapse”, the prompt cleanup appears to have removed the original recursive header-copying mode, but it did **not** solve long-horizon stability.

## Quantitative Checkpoint Results
| Step | Avg@1 | Pass@1 | ExecFmtViol | InvalidFmtTerm | InvalidActionTerm | PlannerInvalidFmt | PlannerFallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.460722 | 0.488889 | 0.044444 | 0.044444 | 0.022222 | 0.133333 | 0.133333 |
| 60 | 0.460576 | 0.477778 | 0.011111 | 0.011111 | 0.000000 | 0.266667 | 0.266667 |
| 70 | 0.229416 | 0.255556 | 0.000000 | 0.000000 | 0.044444 | 0.244444 | 0.244444 |
| 80 | 0.332182 | 0.344444 | 0.011111 | 0.011111 | 0.044444 | 0.466667 | 0.466667 |
| 90 | 0.374486 | 0.411111 | 0.000000 | 0.000000 | 0.044444 | 0.444444 | 0.444444 |
| 100 | 0.404596 | 0.433333 | 0.000000 | 0.000000 | 0.000000 | 0.377778 | 0.377778 |
| 150 | 0.624248 | 0.688889 | 0.133333 | 0.133333 | 0.011111 | 0.011111 | 0.011111 |
| 200 | 0.755230 | 0.811111 | 0.044444 | 0.044444 | 0.022222 | 0.000000 | 0.000000 |
| 250 | 0.690505 | 0.744444 | 0.044444 | 0.044444 | 0.011111 | 0.000000 | 0.000000 |
| 300 | 0.723644 | 0.788889 | 0.033333 | 0.033333 | 0.077778 | 0.000000 | 0.000000 |
| 350 | 0.634342 | 0.711111 | 0.000000 | 0.000000 | 0.200000 | 0.000000 | 0.000000 |
| 400 | 0.546904 | 0.633333 | 0.066667 | 0.066667 | 0.244444 | 0.000000 | 0.000000 |
| 450 | -0.200000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| 500 | -0.200000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |

## Phase-Level Interpretation
| Phase | Steps | Avg@1 range | Pass@1 range | Dominant issue |
| --- | --- | --- | --- | --- |
| Early unstable warmup | 50-100 | 0.229-0.461 | 0.256-0.489 | Planner invalid/fallback remains high despite no tag leakage |
| Mid-stage recovery | 150-300 | 0.624-0.755 | 0.689-0.811 | Best performance; planner metrics reach zero, executor mostly stable |
| Late degradation | 350-400 | 0.547-0.634 | 0.633-0.711 | Invalid action terminations rise sharply |
| Terminal collapse | 450-500 | -0.200 | 0.000 | Executor format collapses completely |

## Main Quantitative Findings
- Best checkpoint overall: `step 200` with `Avg@1=0.755230`, `Pass@1=0.811111`.
- Best checkpoint after the initial 100-step phase: `step 200` with `Avg@1=0.755230`, `Pass@1=0.811111`.
- Worst checkpoint overall: `step 450` with `Avg@1=-0.200000`, `Pass@1=0.000000`.
- Planner formatting/fallback is the main early issue:
  - `step 80`: `PlannerInvalidFormatRate=0.466667`, `PlannerFallbackRate=0.466667`.
  - `step 100`: still `0.377778` on both metrics.
- Planner metrics become clean by the strong region:
  - `step 200`, `250`, `300`: `PlannerInvalidFormatRate=0.0`, `PlannerFallbackRate=0.0`.
- Late-stage deterioration is executor-led:
  - `step 350`: `InvalidActionTerminationRate=0.200000`
  - `step 400`: `InvalidActionTerminationRate=0.244444`
  - `step 450` and `500`: `ExecutorNativeFormatViolations=1.0`, `InvalidFormatTerminationRate=1.0`

## Comparison Against The Earlier Mavino Collapse
The March 21 collapse report established a specific failure signature:
- visible prompt-header leakage beginning around `step 55`
- planner contamination with tags like `[Planner]`, `[Executor Turn]`, `[Environment Observation]`
- recursive executor contamination by `step 98-100`
- planner invalid/fallback and executor invalid-format errors driven by prompt scaffolding reproduction

This new run differs in two important ways:
1. **No observable header leakage in aggregate metrics**. `PlannerTagOnlyRate=0.0` at every checkpoint.
2. **Planner stabilization succeeds mid-run**. Unlike the old run, planner invalid/fallback rates reach zero from `200` through `500`.

The new failure mode is therefore not “header copying”. It is a later executor/policy collapse in which the planner starts issuing terse imperative suggestions (`red ball 1`, `use interact with wall 1`, `use pickup green box 1`) and the executor increasingly copies these suggestions directly instead of wrapping them in the required `Thought:/Action:` structure.

## Trace Evidence At Crucial Points
### Step 150: Healthy multi-step reasoning
- Training step: `150`
- Item: `1235`
- Round: `2`
- Validation result: `ok`
- Extracted action: `toggle and go through green closed door 1`

Planner message:
```text
The green closed door 1 is blocking the way to the yellow ball 1 and the green ball 1 The agent needs to open the green closed door 1 to find the yellow ball 1 and the green ball 1 The agent can use the toggle and go through green closed door 1 action to open the green closed door 1 and find the yellow ball 1 and the green ball 1
```

Executor output:
```text
Thought:
The green closed door 1 is blocking the way to the yellow ball 1 and the green ball 1 The agent needs to open the green closed door 1 to find the yellow ball 1 and the green ball 1

Action:
toggle and go through green closed door 1
```

Analysis:
- Planner reasoning is long-form but grounded in the available actions.
- Executor preserves the BabyAI schema and chooses a valid action.
- This is representative of the mid-stage recovery window where the team setup is functioning as intended.

### Step 200: Peak-performance valid example
- Training step: `200`
- Item: `1289`
- Round: `13`
- Validation result: `ok`
- Extracted action: `go through yellow open door 1`

Planner message:
```text
The yellow open door 1 is located 3 steps in front of the player and 2 steps to the left The purple open door 1 is located right in front of the player and 6 steps away We can start by trying to go through the yellow open door 1 using the go through yellow open door 1 action
```

Executor output:
```text
Thought:
We can start by trying to go through the yellow open door 1 using the go through yellow open door 1 action

Action:
go through yellow open door 1
```

Analysis:
- This is a clean example from the best checkpoint.
- The planner names a concrete route through an open door, and the executor converts that into a valid BabyAI action.
- This supports the claim that the prompt split can work well when the model remains aligned to the executor schema.

### Step 200: Peak checkpoint still contains mixed-format failures
- Training step: `200`
- Item: `1648`
- Round: `1`
- Validation result: `invalid_format`
- Extracted action: `None`

Planner message:
```text
The yellow key 1 is located 2 steps in front of the player and can be picked up directly using the pickup yellow key 1 action The green box 1 is located 2 steps in front of the player and can be picked up directly using the pickup green box 1 action The purple key 1 is located 4 steps in front of the player and can be picked up directly using the pickup purple key 1 action The purple box 1 is located 5
```

Executor output:
```text
Thought:
The yellow key 1 is located 2 steps in front of the player and can be picked up directly using the pickup yellow key 1 action The green box 1 is located 2 steps in front of the player and can be picked up directly using the pickup green box 1 action The purple key 1 is located 4 steps in front of the player and can be picked up directly using the pickup purple key 1 action The purple box 1 is located 5
```

Analysis:
- Even at the best checkpoint, failures still exist at the sample level.
- Here the executor emits only a `Thought:` body and omits the `Action:` line entirely, producing `invalid_format`.
- This explains why `step 200` is strong but not perfect: the aggregate success is high, yet format robustness is not absolute.

### Step 300: Strong late-mid checkpoint with correct execution
- Training step: `300`
- Item: `7195`
- Round: `3`
- Validation result: `ok`
- Extracted action: `toggle and go through green closed door 1`

Planner message:
```text
The yellow ball is being carried by the player. There are several objects in the room, including a blue key 1 and a green closed door 1. The blue key 1 is located 0 steps in front and 1 step to the right. The green closed door 1 is located 2 steps in front and 3 steps to the right
```

Executor output:
```text
Thought: The green closed door 1 needs to be opened to continue

Action:
toggle and go through green closed door 1
```

Analysis:
- The model is still capable of grounded execution at `step 300`.
- Planner text remains verbose but usable.
- Executor maps the reasoning into a correct environment action and preserves the native format.

### Step 300: Same checkpoint, parallel sign of executor fragility
- Training step: `300`
- Item: `7195`
- Round: `3`
- Validation result: `invalid_format`
- Extracted action: `None`

Planner message:
```text
The yellow ball is being carried by an agent in the room. There are several objects in the room, including a blue key 1 and a green closed door 1. The blue key 1 is located 0 steps in front and 1 step to the right. The green closed door 1 is located 2 steps in front and 3 steps to the right. There is also a wall
```

Executor output:
```text
Thought: The green closed door 1 needs to be opened to continue
```

Analysis:
- At the same checkpoint there are also format failures: the executor emits only a `Thought:` line without an `Action:` line.
- This suggests the late collapse is preceded by increasing executor schema brittleness before the aggregate metrics fully break.

### Step 400: Degradation shifts to invalid-action selection
- Training step: `400`
- Item: `401`
- Round: `3`
- Validation result: `action_not_in_available`
- Extracted action: `unknown`

Planner message:
```text
red ball 1
```

Executor output:
```text
Thought: red ball 1 is not available

Action:
"unknown"
```

Analysis:
- The planner message here is drastically under-informative: `red ball 1`.
- The executor keeps the native `Thought:/Action:` format but invents an unavailable action (`unknown`).
- This is qualitatively different from the earlier tag-leak collapse: syntax is mostly preserved, but semantics/action grounding are deteriorating.

### Step 450: Terminal executor format collapse begins
- Training step: `450`
- Item: `3098`
- Round: `1`
- Validation result: `invalid_format`
- Extracted action: `None`

Planner message:
```text
use interact with wall 1
```

Executor output:
```text
use interact with wall 1
```

Analysis:
- The planner message is now imperative but still accepted by the planner validator.
- The executor no longer produces `Thought:/Action:` at all; it simply copies the planner-like string verbatim.
- This exactly matches the evaluation metrics at `450`: `ExecutorNativeFormatViolations=1.0`, `Pass@1=0.0`.

### Step 500: Terminal collapse persists
- Training step: `500`
- Item: `2043`
- Round: `1`
- Validation result: `invalid_format`
- Extracted action: `None`

Planner message:
```text
use pickup green box 1
```

Executor output:
```text
use pickup green box 1
```

Analysis:
- The failure at `500` is the same as `450`, indicating a stable bad attractor rather than a transient spike.
- Planner output is short imperative text; executor copies it directly as plain text.
- The run has effectively lost the executor schema by this point.


## Mechanistic Hypothesis
A defensible hypothesis from the combined metrics and traces is:
1. The prompt cleanup successfully removed the old header-copying failure mode.
2. Mid-training, the planner becomes highly fluent and verbose, often embedding action names directly into natural-language reasoning.
3. The executor benefits from this through the `150-300` window.
4. After extended training, the planner policy collapses toward short imperative suggestions or object labels.
5. The executor then overfits to copying planner text instead of re-rendering it into the BabyAI response schema.
6. The result is a two-stage late degradation:
   - first semantic invalidity (`unknown`, unavailable actions)
   - then full native-format failure (plain copied text)

This would explain why planner metrics stay at zero late in training while executor metrics collapse: the planner validator accepts these short imperative strings, but they are not sufficiently rich or well-structured for the executor policy to remain schema-aligned.

## What This Suggests About The Current Design
- Removing visible tags was necessary and appears to have worked.
- However, the planner and executor are now too close in surface form late in training.
- Because planner outputs are plain language and sometimes imperative, the executor can regress into treating the planner suggestion itself as the final answer.
- The current executor retry path is not preventing this late-stage direct copying.
- The planner validator may now be too permissive late in training: strings like `red ball 1` or `use pickup green box 1` pass planner validation even though they are poor planning signals.

## Concrete Next Experiments
1. **Strengthen executor schema anchoring**
   - Train or infer with a stronger executor-side instruction emphasizing that planner text is advisory only and cannot be emitted as the final answer.
   - Consider adding a small negative reward specifically for final outputs that match the planner message too closely.
2. **Re-tighten planner quality without reintroducing short-hint brittleness**
   - Keep planner free-form, but add a stronger planner-side requirement that the message must describe state and rationale, not just an imperative fragment.
   - Reject degenerate planner outputs like bare object names or `use ...` patterns.
3. **Add explicit executor repair on copied planner strings**
   - If executor output lacks `Thought:/Action:` but closely matches planner text, force a repair prompt before terminating.
4. **Checkpoint policy**
   - The evidence strongly supports treating `200` as the current best checkpoint and `300` as the best late-stage checkpoint.
   - Avoid deploying or analyzing `450+` as if they are merely weaker variants; they represent a qualitatively different bad regime.
5. **Need for fuller local trace archive**
   - The automation saved evaluation summaries locally, but not the full current-run trace directory.
   - For future deep analysis, sync `/home/mavinomichael/agentgym_runs/logs/babyai_2agent_scaling_100_8gpu_plain_split_retry_v2/trace_train/` into the project so the full run is reproducibly inspectable offline.

## Bottom Line
This experiment is a partial success.
- It appears to fix the original Mavino prompt-scaffolding collapse.
- It discovers a new, later collapse mode centered on executor schema loss rather than prompt-header leakage.
- The run is strong and useful through roughly `step 300`, peaks at `step 200`, then degrades, and fully collapses by `450`.

For external deep research, the right framing is:
- **Old failure mode fixed**
- **New failure mode identified**
- **Best-performing regime isolated (`150-300`, especially `200`)**
- **Late-run collapse characterized by executor copying of planner-style imperative text**
