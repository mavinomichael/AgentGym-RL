from __future__ import annotations

from verl.utils.agent_dataset.rl_dataset import RLHFDataset as BaseRLHFDataset
from verl.utils.agent_dataset.rl_dataset import collate_fn

from .protocol import build_structured_bootstrap


class ImproveRLHFDataset(BaseRLHFDataset):
    def _build_messages(self, example: dict):
        example["data_source"] = example[self.prompt_key].split("_")[0]
        return build_structured_bootstrap()


__all__ = ["ImproveRLHFDataset", "collate_fn"]
