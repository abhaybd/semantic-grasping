import os
import time
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from model import GraspEncoder, Checkpointer, WarmupCosineLR
from data import GraspDescriptionRegressionDataset
from utils import tqdm

@torch.no_grad()
def test(model: nn.Module, test_loader: DataLoader, rank: int, world_size: int):
    model.eval()
    losses = []
    variances = []
    device_id = rank % torch.cuda.device_count()
    for batch in tqdm(test_loader, desc="Test", leave=False, disable=rank != 0):
        rgb, xyz, grasp_pose = batch["rgb"].to(device_id), batch["xyz"].to(device_id), batch["grasp_pose"].to(device_id)
        text_embedding = batch["text_embedding"].float().to(device_id)
        grasp_features = model(rgb, xyz, grasp_pose)
        batch_loss = 1.0 - F.cosine_similarity(grasp_features, text_embedding, dim=-1)
        losses.extend(batch_loss.tolist())
        variances.append(torch.var(grasp_features, dim=0).mean().item())
    model.train()
    info_to_gather = {
        "loss": np.mean(losses),
        "variance": np.mean(variances),
    }
    return gather_info(info_to_gather, world_size)

def gather_info(info_to_gather: dict[Any, float], world_size: int):
    gathered_infos: list[dict[Any, float]] = [None] * world_size
    dist.all_gather_object(gathered_infos, info_to_gather)
    return {k: np.mean([d[k] for d in gathered_infos]) for k in info_to_gather}

def safe_div(a: int, b: int):
    if a % b != 0:
        raise ValueError(f"Cannot divide {a} by {b} evenly!")
    return a // b

def nested_dict_to_flat_dict(d: dict[str, Any], pfx: str = ""):
    flat_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flat_dict.update(nested_dict_to_flat_dict(v, pfx=f"{pfx}{k}/"))
        else:
            flat_dict[f"{pfx}{k}"] = v
    return flat_dict

def build_wandb_config(config: DictConfig):
    wandb_config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    wandb_config["env"] = {
        **{k: v for k, v in os.environ.items() if k.startswith("GANTRY_")},
        **{k: v for k, v in os.environ.items() if k.startswith("BEAKER_")},
    }
    return wandb_config

def create_dataloader(dataset: GraspDescriptionRegressionDataset, rank: int, world_size: int, config: DictConfig):
    loader_kwargs = {
        "persistent_workers": True,
        "pin_memory": True,
        "batch_size": safe_div(config["train"]["dataloader"]["batch_size"], world_size),
        "num_workers": safe_div(config["train"]["dataloader"]["num_workers"], world_size),
    }
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    return DataLoader(dataset, sampler=sampler, collate_fn=dataset.collate_fn, **loader_kwargs)

@hydra.main(version_base=None, config_path="../config", config_name="regression.yaml")
def main(config: DictConfig):
    if missing_keys := OmegaConf.missing_keys(config):
        raise ValueError(f"Missing keys: {missing_keys}")

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_id = rank % torch.cuda.device_count()
    config["train"]["distributed"]["world_size"] = world_size

    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    if rank == 0:
        print(OmegaConf.to_yaml(config))
        os.makedirs(ckpt_dir, exist_ok=True)

        if "GANTRY_TASK_NAME" in os.environ:
            task_name = os.environ["GANTRY_TASK_NAME"]
        elif "name" in config:
            task_name = config["name"]
        else:
            task_name = None

        run_id = os.environ.get("BEAKER_EXPERIMENT_ID", None)
        run = wandb.init(
            entity="prior-ai2",
            project="semantic-grasping",
            config=build_wandb_config(config),
            name=task_name,
            dir=out_dir,
            job_type="train",
            id=run_id,
            resume="allow"
        )

    torch.manual_seed(config["train"]["seed"] + rank)

    grasp_encoder = GraspEncoder(config["model"]).to(device_id)
    model = DDP(grasp_encoder, device_ids=[device_id], find_unused_parameters=True)
    model.train()

    if rank == 0:
        print(f"# trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"# parameters: {sum(p.numel() for p in model.parameters()):,}")

    img_processor = grasp_encoder.create_rgb_processor()
    dataset = GraspDescriptionRegressionDataset(**config["train"]["dataset"], img_processor=img_processor)
    test_frac = config["train"]["test"]["frac"]
    if test_frac > 0:
        gen = torch.Generator().manual_seed(config["train"]["seed"])
        train_dataset, test_dataset = random_split(dataset, [1 - test_frac, test_frac], generator=gen)
        train_loader = create_dataloader(train_dataset, rank, world_size, config)
        test_loader = create_dataloader(test_dataset, rank, world_size, config)
    else:
        train_loader = create_dataloader(dataset, rank, world_size, config)
        test_loader = None

    optimizer = optim.AdamW(model.parameters(), **config["train"]["optimizer"])
    lr_scheduler = WarmupCosineLR(
        optimizer,
        config["train"]["lr_schedule"]["warmup_steps"],
        config["train"]["steps"],
        config["train"]["lr_schedule"]["final_factor"]
    )

    checkpointer = Checkpointer(ckpt_dir, model=model.module, optimizer=optimizer, lr_scheduler=lr_scheduler)
    start_step = checkpointer.load()
    dist.barrier()

    step = start_step
    with tqdm(total=config["train"]["steps"], initial=start_step, desc="Training", disable=rank != 0) as pbar:
        while step < config["train"]["steps"]:
            for batch in train_loader:
                optimizer.zero_grad()
                rgb, grasp_pose = batch["rgb"].to(device_id), batch["grasp_pose"].to(device_id)
                xyz_inputs = {k: v.to(device_id) for k, v in batch["xyz_inputs"].items()}
                text_embedding = batch["text_embedding"].to(device_id)

                infer_start = time.perf_counter()
                grasp_features = model(rgb, xyz_inputs, grasp_pose)
                infer_end = time.perf_counter()
                infer_time = infer_end - infer_start

                loss = 1.0 - F.cosine_similarity(grasp_features, text_embedding, dim=-1).mean()
                variance = torch.var(grasp_features, dim=0).mean().item()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                info = {
                    "step": step,
                    "epoch": step // len(train_loader),
                    "lr": np.mean(lr_scheduler.get_last_lr()),
                }
                info_to_gather = {
                    "loss": loss.item(),
                    "variance": variance,
                    "infer_time": infer_time,
                }
                info.update(gather_info(info_to_gather, world_size))

                if test_loader is not None and config["train"]["test"]["period"] and step % config["train"]["test"]["period"] == 0:
                    test_results = test(model, test_loader, rank, world_size)
                    info["test"] = test_results

                if rank == 0:
                    run.log(nested_dict_to_flat_dict(info))

                    if config["train"]["save_period"] and step % config["train"]["save_period"] == 0:
                        checkpointer.save(step)

                step += 1
                pbar.update(1)
                if step >= config["train"]["steps"]:
                    break

    if rank == 0:
        save_path = os.path.join(out_dir, "model.pt")
        torch.save(model.module.state_dict(), save_path)
        wandb.save(save_path, base_path=out_dir)

if __name__ == "__main__":
    main()
