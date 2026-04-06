# Improve Multi-Agent

This workspace is the clean research stack for the structured planner–executor
experiments.

Design goals:
- isolate the new protocol and optimization ideas from the legacy stack
- keep paper-facing artifacts and ablation manifests together
- support staged warm-start before joint RL

Suggested workflow:
1. Build executor and planner warm-start datasets from archived successful traces
2. Run executor warm-start
3. Run planner warm-start
4. Run planner-only RL with frozen executor
5. Run light joint RL

Key outputs live under:
- `datasets/`
- `notes/`
- `manifests/`
- `reports/`
