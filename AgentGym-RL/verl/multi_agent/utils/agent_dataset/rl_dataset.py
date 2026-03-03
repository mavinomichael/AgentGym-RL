# Multi-agent extension.
# Derived from: /Users/mavinomichael/PycharmProjects/AgentGym-RL/AgentGym-RL/verl/utils/agent_dataset/rl_dataset.py
# Original file left untouched for comparison.

from verl.utils.agent_dataset.rl_dataset import RLHFDataset as BaseRLHFDataset
from verl.utils.agent_dataset.rl_dataset import collate_fn
from verl.multi_agent.workers.rollout.agent_vllm_rollout.planner_executor import build_multi_agent_bootstrap


class RLHFDataset(BaseRLHFDataset):
    def __init__(self, *args, multi_agent_config=None, **kwargs):
        self.multi_agent_config = multi_agent_config
        super().__init__(*args, **kwargs)

    def _build_messages(self, example: dict):
        # ORIGINAL FLOW DIFFERENCE:
        # single-agent path used the environment prompt directly.
        # multi-agent path wraps it in planner-executor team instructions.
        example["data_source"] = example[self.prompt_key].split("_")[0]
        return build_multi_agent_bootstrap(self.env_client)


__all__ = ["RLHFDataset", "collate_fn", "build_multi_agent_bootstrap"]
