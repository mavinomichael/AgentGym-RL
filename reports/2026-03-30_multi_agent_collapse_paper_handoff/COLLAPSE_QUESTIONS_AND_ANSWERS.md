# Collapse Questions and Answers with Evidence

This note updates the paper handoff with the question-led framing we discussed with the professor:

1. When does collapse begin?
2. Can the system recover from collapse?
3. How do we prevent collapse?

It also answers the follow-on questions that became important after the BabyAI 3-agent reviewer run and the WebArena clean 2-agent run:

- When collapse begins, is it sudden or gradual?
- Does adding more agents help, or just create new failure modes?
- Is collapse planner-driven, executor-driven, or interface-driven?
- Is collapse environment-specific, or does it generalize beyond BabyAI?
- Does removing tags / using cleaner planner-executor prompts help?
- Once full collapse happens, does the model recover later?

The regime-level summary table used by the figures below is stored in:

- `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/2026-03-30_multi_agent_collapse_paper_handoff/collapse_regime_summary.tsv`
- `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/2026-03-30_multi_agent_collapse_paper_handoff/collapse_regime_summary.json`

Updated question-focused figures:

![Collapse timeline](/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/2026-03-30_multi_agent_collapse_paper_handoff/figures/fig_collapse_timeline.png)

![Recovery summary](/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/2026-03-30_multi_agent_collapse_paper_handoff/figures/fig_recovery_summary.png)

## Evidence Base

Primary metric and trace sources:

- `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/babyai_multi_agent_diagnostics_2026-03-21/mavino_collapse_2agent_scaling_100/MAVINO_COLLAPSE_REPORT.md`
- `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/2026-03-23_plain_split_retry_v2_deep_research/DEEP_RESEARCH_ANALYSIS.md`
- `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/2026-03-23_dense500_eval_and_traces/SUMMARY.md`
- `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/20260402T123754Z_babyai_2agent_fixed20_8gpu_plain_split_notag_persist_v1/checkpoint_metrics.tsv`
- `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/20260403_130617_babyai_3agent_executor_reviewer_scaling_600_8gpu_no_tags_v1/COLLAPSE_ANALYSIS.md`
- `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/20260405T220020+0100_webarena_2agent_scaling_600_plain_split_notag_clean_v2/summary.md`
- `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/2026-03-30_multi_agent_collapse_paper_handoff/selected_trace_packets.md`

## Short Answers

### 1. When does collapse begin?

Collapse onset is **regime-dependent**, not universal. More importantly, the first sign is visible in the \emph{content of the messages}, not just in the step number.

#### What collapse looks like when it begins

The earliest trace-level signature across our runs is:

**the model stops speaking in grounded environment language and starts speaking in protocol artifacts, schema fragments, fallback text, or degenerate tokens.**

Concretely:

- **Tagged 2-agent ScalingRL**: the first suspicious planner message is not nonsense yet; it is a \emph{role token leaking into the semantic channel}.
  - First visible leak:
    - planner raw output: `[Planner] Check room for ball.`
  - Why this matters:
    - the planner is already copying prompt scaffolding into what should be environment-grounded guidance
    - later the collapse becomes explicit recursive wrapper copying:
      - planner raw output: `[Planner]\n[Executor Turn]\n[Environment Observation]\n[Executor Response]\n`
      - executor raw output begins repeating `[Executor Turn]`, `[Environment Observation]`, `[Reviewers Turn]`

- **Fixed-round tagged 2-agent**: the first sign is \emph{planner verbosity drift}.
  - Onset trace:
    - planner raw output: `Check if there is a red ball in the room or if you need to move to another part of the room.`
  - Why this matters:
    - this is not invalid by meaning, but it is already too long for the planner contract
    - it is the earliest sign that the planner is drifting away from a compact control signal toward uncontrolled free-form narration

- **Coarse no-tag 2-agent ScalingRL**: the first sign is \emph{executor action invention}.
  - Onset trace:
    - planner raw output: `red ball 1`
    - executor raw output: `Thought: red ball 1 is not available\n\nAction:\n\"unknown\"`
  - Why this matters:
    - once the executor emits `"unknown"` instead of one legal BabyAI action, collapse has started behaviorally even though there are still no role tags

- **Dense no-tag 2-agent ScalingRL**: the first sign is \emph{conversation-transcript leakage into the executor output}.
  - Onset trace:
    - planner raw output: `Let's check the available actions for any visible objects`
    - executor first pass raw output: `check available actions\n\nuser\nOutput:\nAvailable actions: []\n\nThere are no available actions visible in front of you...`
    - executor final raw output: `You are facing a wall\n\nAction:\nmove forward`
  - Why this matters:
    - the executor is no longer producing one clean environment-native action response
    - it is pulling in chat transcript fragments like `user\nOutput:`
    - this is the no-tag equivalent of scaffold contamination

- **3-agent reviewer run**: the first sign is \emph{schema disagreement between executor and reviewer channels}.
  - Earliest suspicious message:
    - executor first pass raw output: `Action:\nturn left\ncheck available actions`
    - executor final raw output: `Thought:\nFollow suggestion\n\nAction:\nturn left\ncheck available actions`
  - Why this matters:
    - one executor turn is now trying to emit two actions
    - later the reviewer schema leaks directly into executor output:
      - `Verdict:\nPASS\n\nReason:\nOK`
    - this is the clearest trace-level sign that the interface, not just one role, is destabilizing

- **WebArena clean 2-agent**: the first sign is \emph{instant tag-only / punctuation collapse}.
  - First bad planner message:
    - planner raw output: `!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`
  - First bad executor message:
    - executor raw output: `!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!...`
  - Why this matters:
    - this is not a wrong action; it is a complete abandonment of the task/action contract
    - the model goes from one grounded planner message at step 1 to pure degenerate tokens almost immediately

So the answer to "when does collapse begin?" in trace terms is:

**collapse begins the moment the model's messages stop being grounded control signals and start becoming protocol leakage, transcript leakage, schema contamination, invented actions, or pure punctuation spam.**

The step number only tells us \emph{how early}; the trace content tells us \emph{what collapse actually is}.

- **Tagged 2-agent ScalingRL**: first warning sign by `step 55` when planner headers start leaking into planner text; full recursive scaffold contamination by `step 98-100`.
  - Evidence:
    - `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/2026-03-30_multi_agent_collapse_paper_handoff/selected_trace_packets.md`
    - trace IDs: `tagged_step55_first_header_leak`, `tagged_step100_recursive_scaffold`
- **2-agent no-tag dense ScalingRL**: strong through `step 350`, transition at `step 400`, full executor-format collapse at `step 450-500`.
  - Evidence:
    - `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/2026-03-23_dense500_eval_and_traces/SUMMARY.md`
    - trace IDs: `dense500_step400_transition_example`, `dense500_step450_retry_exhaustion_example`, `dense500_step500_terminal_collapse_example`
- **3-agent reviewer**: first clear instability at `step 150`, recovery through `250-350`, terminal-collapse onset at `step 400`, full collapse at `step 450`.
  - Evidence:
    - `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/20260403_130617_babyai_3agent_executor_reviewer_scaling_600_8gpu_no_tags_v1/COLLAPSE_ANALYSIS.md`
- **WebArena clean 2-agent**: warning signs by `step 2`, first concrete tag-only failure at `step 3`, first fully collapsed sampled batch by `step 9`, and flat-zero eval at every saved checkpoint `50-600`.
  - Evidence:
    - `/Users/mavinomichael/PycharmProjects/AgentGym-RL/reports/20260405T220020+0100_webarena_2agent_scaling_600_plain_split_notag_clean_v2/summary.md`
    - trace IDs: `webarena_clean_step1_valid`, `webarena_clean_step3_tag_only_collapse`, `webarena_clean_step553_terminal`

### 2. Can the system recover from collapse?

**It can recover from early instability, but not from full collapse.**

- The clearest evidence of partial recovery is the **3-agent reviewer run**:
  - it falls sharply at `150-200`
  - then recovers to `Pass@1=0.6778` at `step 300` and `Pass@1=0.7111` at `step 350`
  - then enters terminal collapse by `450`
- There is **no evidence of spontaneous recovery after full collapse** in any BabyAI regime.
- There is also **no evidence of recovery in WebArena** once the early collapse begins; the entire eval frontier stays flat-zero.

### 3. How do we prevent collapse?

The evidence supports three practical prevention levers and three deeper hypotheses.

**Evidence-backed prevention levers from our runs**

- **Remove visible role tags**:
  - This removes the specific scaffold-copying collapse mode seen in tagged runs.
  - Evidence:
    - tagged collapse is explicit in `tagged_step55_first_header_leak`
    - no-tag runs do not show the same bracketed header recursion
- **Select checkpoints near the last stable band instead of training deep into collapse**:
  - Dense no-tag peaks at `step 300`.
  - 3-agent reviewer has a usable late recovery band at `300-350`.
  - Later checkpoints are consistently worse.
- **Treat rising format metrics as early-stop signals**:
  - `PlannerInvalidFormatRate`
  - `PlannerTagOnlyRate`
  - `ExecutorNativeFormatViolations`
  - reviewer-schema leakage into executor outputs

**Interpretive hypotheses consistent with our traces and deep-research notes**

- **Free-form planner text leaks scaffolding into executor outputs.**
  - This is directly consistent with:
    - tagged BabyAI header leakage
    - 3-agent reviewer schema leakage at `step 350`
    - WebArena planner punctuation/tag-only collapse
- **Shared reward and shared blame likely make learning noisy when executor format breaks.**
  - In the dense no-tag 2-agent run, the planner can remain valid while the executor collapses late.
  - That suggests the team reward does not cleanly tell the system which role failed.
- **Planner and executor have different action spaces and error distributions, so homogeneous normalization is likely unstable.**
  - The late no-tag BabyAI failure is executor-dominant.
  - The reviewer run adds another incompatible schema channel.
  - We did not isolate role-wise normalization experimentally in the baseline runs, so this remains an informed explanation rather than a claimed ablation result.

## Longer Answers with Evidence

### When collapse begins, is it sudden or gradual?

It is **both**, depending on the regime.

- **Gradual BabyAI collapse**:
  - dense no-tag 2-agent rises through `250-350`, degrades at `400`, then collapses at `450-500`
  - coarse no-tag 2-agent peaks at `200`, weakens through `350-400`, then collapses at `450`
- **Sudden BabyAI collapse**:
  - fixed-round no-tag stays healthy through `350`, then flips to complete failure at `400`
- **Near-immediate WebArena collapse**:
  - a valid trace exists at `step 1`
  - the first warning signs appear by `step 2`
  - the first clearly collapsed sampled batch appears by `step 9`

### Does adding more agents help, or just create new failure modes?

Adding a reviewer **changes the failure mode**, but does not solve collapse.

- 3-agent reviewer run:
  - best checkpoint `step 100`, `Pass@1=0.7222`
  - later recovery `step 350`, `Pass@1=0.7111`
  - full collapse at `450+`
- Dense 2-agent no-tag run:
  - best checkpoint `step 300`, `Pass@1=0.8222`
  - full collapse at `450+`

So the reviewer does **not** improve the best BabyAI checkpoint, but it does show that early instability can be survived and partially recovered from. The cost is a new failure surface: reviewer-schema contamination of executor outputs.

### Is collapse planner-driven, executor-driven, or interface-driven?

The answer depends on the regime.

- **Tagged 2-agent**: planner-driven scaffold leakage that becomes interface-wide.
- **Dense no-tag 2-agent**: late collapse is mainly **executor-side**; planner stays formally valid while executor format failure saturates.
- **3-agent reviewer**: interface-driven early, then full three-channel collapse later.
- **WebArena clean 2-agent**: planner collapse happens almost immediately and poisons the executor via fallback and invalid-format termination.

So the strongest cross-run conclusion is:

- collapse is usually **not just one role failing in isolation**
- it is often an **interface instability** that appears first in one channel and then contaminates the others

### Is collapse environment-specific, or does it generalize beyond BabyAI?

It clearly generalizes beyond BabyAI.

- BabyAI shows late-stage collapse after otherwise strong learning.
- WebArena shows near-immediate collapse under a simpler 2-agent topology.

That difference is important:

- BabyAI suggests long-horizon drift and late executor bottlenecks
- WebArena shows the protocol/interface problem can be so fragile that the run never develops a healthy frontier at all

### Does removing tags / using cleaner planner-executor prompts help?

**Yes, but only partially.**

- It removes the specific tagged scaffold-copying failure mode.
- It enables much stronger intermediate checkpoints:
  - coarse no-tag best `Pass@1=0.8111`
  - dense no-tag best `Pass@1=0.8222`
- But it does **not** prevent:
  - fixed-round no-tag total collapse at `400`
  - dense no-tag executor collapse at `450-500`
  - WebArena near-immediate failure

So prompt cleaning is necessary, but not sufficient.

### Once full collapse happens, does the model recover later?

Across the runs we have, **no**.

- Tagged 2-agent: no recovery after the step-100 recursive scaffold collapse.
- Dense/coarse no-tag 2-agent: no recovery after the `450+` executor-format collapse.
- 3-agent reviewer: no recovery after the `450+` three-channel collapse.
- WebArena clean 2-agent: no recovery at all after the early collapse onset; saved checkpoints `50-600` are all flat-zero.

## What the results support in the paper

These are the paper-safe claims that are directly supported by the current metrics and traces:

1. **Collapse onset is regime-dependent.**
2. **Full collapse is usually irreversible within the observed training horizon.**
3. **Removing prompt tags removes one collapse mode but does not stabilize training overall.**
4. **A reviewer can change the failure mode and allow partial recovery from early instability, but it does not eliminate terminal collapse.**
5. **The failure is not BabyAI-specific: WebArena shows the same instability problem in a harsher form.**

These are useful interpretation claims, but they should be labeled more carefully in the paper:

1. **Free-form planner text leakage is a likely cause of several observed failures.**
2. **Credit-assignment blur is a plausible explanation for planner/executor mismatch during collapse.**
3. **Treating planner and executor as one homogeneous PPO population is a plausible contributor to instability.**

Those three are well motivated by the traces and by the deep-research synthesis, but they are **not** isolated causal ablations in the current paper.
