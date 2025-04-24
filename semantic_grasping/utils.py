import time
import os
from typing import Any, Iterable

import numpy as np
from tqdm import tqdm as tqdm_
from omegaconf import DictConfig, OmegaConf

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


def backproject(cam_K: np.ndarray, depth: np.ndarray):
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    uvd = np.stack((u, v, np.ones_like(u)), axis=-1).astype(np.float32)
    uvd *= np.expand_dims(depth, axis=-1)
    xyz = uvd @ np.expand_dims(np.linalg.inv(cam_K).T, axis=0)
    return xyz

class tqdm(tqdm_):
    def __init__(self, *args, **kwargs):
        kwargs["bar_format"] = "{l_bar}{bar}{r_bar}\n"
        super().__init__(*args, **kwargs)

def move_to_device(batch: dict[str, Any], device_id: Any):
    ret = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            ret[k] = v.to(device_id, non_blocking=True)
        elif isinstance(v, dict):
            ret[k] = move_to_device(v, device_id)
        else:
            ret[k] = v
    return ret

def time_iter(iterable: Iterable[Any]):
    it = iter(iterable)
    while True:
        start = time.perf_counter()
        try:
            item = next(it)
        except StopIteration:
            break
        end = time.perf_counter()
        yield end - start, item

def build_wandb_config(config: DictConfig):
    wandb_config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    wandb_config["env"] = {
        **{k: v for k, v in os.environ.items() if k.startswith("GANTRY_")},
        **{k: v for k, v in os.environ.items() if k.startswith("BEAKER_")},
    }
    return wandb_config

def gather_info(info_to_gather: dict[Any, float], world_size: int):
    gathered_infos: list[dict[Any, float]] = [None] * world_size
    dist.all_gather_object(gathered_infos, info_to_gather)
    return {k: np.mean([d[k] for d in gathered_infos]) for k in info_to_gather}

def nested_dict_to_flat_dict(d: dict[str, Any], pfx: str = ""):
    flat_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flat_dict.update(nested_dict_to_flat_dict(v, pfx=f"{pfx}{k}/"))
        else:
            flat_dict[f"{pfx}{k}"] = v
    return flat_dict

def safe_div(a: int, b: int):
    if a % b != 0:
        raise ValueError(f"Cannot divide {a} by {b} evenly!")
    return a // b

def create_dist_dataloader(dataset: Dataset, rank: int, world_size: int, config: DictConfig):
    loader_kwargs = {
        "persistent_workers": True,
        "pin_memory": True,
        "batch_size": safe_div(config["train"]["dataloader"]["batch_size"], world_size),
        "num_workers": safe_div(config["train"]["dataloader"]["num_workers"], world_size),
    }
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    return DataLoader(dataset, sampler=sampler, **loader_kwargs)
