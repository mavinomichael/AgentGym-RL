# Multi-agent extension.
# Derived from: /Users/mavinomichael/PycharmProjects/AgentGym-RL/AgentGym-RL/verl/agent_trainer/main_generation.py
# Original file left untouched for comparison.

from collections import defaultdict
import json
import os

import hydra
import numpy as np
import pandas as pd
import ray
import verl.utils.torch_functional as verl_F

from verl import DataProto
from verl.multi_agent.envs import get_task_profile
from verl.multi_agent.utils.agent_dataset import build_multi_agent_bootstrap
from verl.multi_agent.workers.agent_fsdp_workers import ActorRolloutRefWorker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.agentgym.client import init_env_client
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    from omegaconf import OmegaConf
    from pprint import pprint
    from verl.utils import hf_tokenizer

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)
    local_path = copy_local_path_from_hdfs(config.model.path)
    tokenizer = hf_tokenizer(local_path)

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."

    dataset = pd.read_json(os.path.join(config.data.path, f"{config.agentgym.task_name}_test.json"))
    item_ids = dataset[config.data.prompt_key].tolist()
    category_files = os.listdir(config.data.path)
    category_files = [f for f in category_files if not f.startswith(f"{config.agentgym.task_name}_test")]
    category_map = {}
    for category_file in category_files:
        path = os.path.join(config.data.path, category_file)
        with open(path, "r") as file_obj:
            datas = json.load(file_obj)
            for data in datas:
                category_map[data["item_id"]] = category_file.split(".")[0]

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()

    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    dp_size = wg.world_size // config.rollout.tensor_model_parallel_size
    num_batch = (total_samples // config_batch_size) + 1
    output_lst = [[] for _ in range(config.data.n_samples)]
    native_format_valid_lst = []
    invalid_format_terminated_lst = []
    invalid_action_terminated_lst = []
    env_client = init_env_client(config.agentgym)
    task_profile = get_task_profile(config.agentgym.task_name)

    for batch_idx in range(num_batch):
        print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
        start_idx = batch_idx * config_batch_size
        end_idx = min(total_samples, start_idx + config_batch_size)
        batch_item_ids = item_ids[start_idx:end_idx]
        messages = []
        prompts = []
        for _ in range(len(batch_item_ids)):
            prompt_messages, prompt_with_chat_template = build_multi_agent_bootstrap(env_client, task_profile)
            messages.append(prompt_messages)
            prompts.append(prompt_with_chat_template)

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompts,
            tokenizer=tokenizer,
            max_length=config.data.max_prompt_length,
            pad_token_id=tokenizer.pad_token_id,
            left_pad=True,
        )
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

        data = DataProto.from_dict(tensors=batch_dict)
        data.meta_info["global_steps"] = "test_batch_" + str(batch_idx)
        data.meta_info["max_rounds"] = config.agentgym.max_rounds
        data.non_tensor_batch["item_id"] = np.array(batch_item_ids, dtype=object)
        data.non_tensor_batch["raw_prompt"] = np.array(messages, dtype=object)

        real_batch_size = data.batch["input_ids"].shape[0]
        if real_batch_size % dp_size != 0:
            dummy_data_size = dp_size - real_batch_size % dp_size
            data = DataProto.concat([data, data[:dummy_data_size]])
            print(
                f"dp_size {dp_size} is not divisible by real_batch_size {real_batch_size}, add {dummy_data_size} dummy data"
            )

        batch_size = data.batch["input_ids"].shape[0]
        assert batch_size % dp_size == 0, f"batch_size {batch_size} is not divisible by dp_size {dp_size}"
        print(f"[{batch_idx + 1}/{num_batch}] Start to generate.")

        for i in range(config.data.n_samples):
            output = wg.generate_sequences(data)
            output = output[:real_batch_size]
            output_lst[i].extend(output.batch["task_scores"].sum(dim=-1).tolist())
            if "executor_native_format_valid" in output.batch:
                native_format_valid_lst.extend(output.batch["executor_native_format_valid"].tolist())
            if "invalid_format_terminated" in output.batch:
                invalid_format_terminated_lst.extend(output.batch["invalid_format_terminated"].tolist())
            if "invalid_action_terminated" in output.batch:
                invalid_action_terminated_lst.extend(output.batch["invalid_action_terminated"].tolist())

    output_np = np.array(output_lst, dtype=object)
    output_np = np.transpose(output_np, axes=(1, 0))
    output_lst = output_np.tolist()

    print("============Total Task Evaluation============")
    print(f"Avg@{config.data.n_samples}: {np.mean(output_np)}")
    print(f"Pass@{config.data.n_samples}: {np.mean(np.max(output_np, axis=-1) > 0)}")
    if native_format_valid_lst:
        print(f"ExecutorNativeFormatViolations: {1.0 - np.mean(np.array(native_format_valid_lst, dtype=float))}")
    if invalid_format_terminated_lst:
        print(f"InvalidFormatTerminationRate: {np.mean(np.array(invalid_format_terminated_lst, dtype=float))}")
    if invalid_action_terminated_lst:
        print(f"InvalidActionTerminationRate: {np.mean(np.array(invalid_action_terminated_lst, dtype=float))}")
    print("============Sub Task Evaluation============")

    category_success_bucket = defaultdict(list)
    for item_id, score in zip(item_ids, output_lst):
        category = category_map[item_id]
        category_success_bucket[category].append(score)
    for category_file in category_files:
        category = category_file.split(".")[0]
        print(f"Category: {category}")
        print(f"Avg@{config.data.n_samples}: {np.mean(np.array(category_success_bucket[category]))}")
        print(f"Pass@{config.data.n_samples}: {np.mean(np.max(np.array(category_success_bucket[category]), axis=-1) > 0)}")


if __name__ == "__main__":
    main()
