# VM Acquisition Scripts

This directory contains the direct on-demand GPU acquisition workflow for AgentGym training.

## Main entrypoint

Run the coordinator:

```bash
bash /Users/mavinomichael/PycharmProjects/AgentGym-RL/scripts/vm/acquire_agentgym_on_demand_vm.sh
```

The coordinator:

- launches four parallel workers
- tries `8 x H200` first and `8 x B200` second across disjoint zone groups
- waits for the first instance that reaches `RUNNING`
- verifies the winner with SSH and `nvidia-smi`
- deletes losing instances for the same acquisition batch
- syncs this repo to the winner
- prepares the training environment
- starts the 2-agent scaling run
- deletes the old `odion-agentgym-rl` instance only after the new training run writes its first `step:` line

## Dry run

Use a dry run to inspect the worker matrix and generated instance names without creating anything:

```bash
DRY_RUN=1 bash /Users/mavinomichael/PycharmProjects/AgentGym-RL/scripts/vm/acquire_agentgym_on_demand_vm.sh
```

Example with a simulated winner:

```bash
DRY_RUN=1 \
SIMULATED_WINNER_GROUP=h200-us \
SIMULATED_WINNER_ZONE=us-east5-a \
bash /Users/mavinomichael/PycharmProjects/AgentGym-RL/scripts/vm/acquire_agentgym_on_demand_vm.sh
```

## Key scripts

- `acquire_agentgym_on_demand_vm.sh`
  - coordinator and cleanup logic
- `create_agentgym_a100_vm.sh`
  - generic per-zone VM create helper
- `agentgym_a100_startup.sh`
  - startup metadata script installed onto the VM
- `remote_prepare_agentgym_training.sh`
  - remote environment/bootstrap script
- `remote_launch_agentgym_training.sh`
  - remote training launcher

## Defaults

- image: `ubuntu-accelerator-2204-amd64-with-nvidia-580-v20260313`
- boot disk: `2000 GB`
- boot disk type: `hyperdisk-balanced`
- tags: `agentgym-rl,visual-inspector`
- training launcher:
  - `/Users/mavinomichael/PycharmProjects/AgentGym-RL/scripts/multi_agent/run_babyai_2agent_scaling_200_8gpu.sh`

## Notes

- This workflow is direct on-demand only.
- It does not use Spot, reservations, or lower GPU-count fallbacks.
- The winner info is written to `/tmp/agentgym-acquire-<batch_id>/winner.env`.
