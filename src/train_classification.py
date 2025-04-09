import os
import time
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision

from model import GraspClassifier, Checkpointer, WarmupCosineLR
from data import GraspDescriptionClassificationDataset
from utils import tqdm, nested_dict_to_flat_dict, build_wandb_config, create_dist_dataloader, time_iter, move_to_device, gather_info

@torch.inference_mode()
def test(model: nn.Module, test_loader: DataLoader, rank: int, world_size: int):
    model.eval()
    losses = []
    variances = []
    device_id = rank % torch.cuda.device_count()
    metrics = {
        "f1": BinaryF1Score(),
        "accuracy": BinaryAccuracy(),
        "precision": BinaryPrecision()
    }
    for metric in metrics.values():
        metric.cuda()
    for batch in tqdm(test_loader, desc="Test", leave=False):
        batch = move_to_device(batch, device_id)
        contrastive_batch, labels = construct_contrastive_batch(batch)

        rgb, xyz, grasp_pose = contrastive_batch["rgb"], contrastive_batch["xyz"], contrastive_batch["grasp_pose"]
        text_inputs = contrastive_batch["text_inputs"]

        pred_logits: torch.Tensor = model(rgb, xyz, grasp_pose, text_inputs)

        batch_loss = F.binary_cross_entropy_with_logits(pred_logits, labels).item()
        losses.append(batch_loss)
        variances.append(torch.var(pred_logits, dim=0).mean().item())

        for metric in metrics.values():
            metric(F.sigmoid(pred_logits), labels)
    model.train()
    info_to_gather = {
        "loss": np.mean(losses),
        "variance": np.mean(variances),
        **{k: metric.compute().item() for k, metric in metrics.items()},
    }
    return gather_info(info_to_gather, world_size)

def construct_contrastive_batch(batch: dict[str, Any]) -> tuple[dict[str, Any], torch.Tensor]:
    # find unique annotation IDs and find their first occurrence in the batch
    annot_to_id: dict[str, int] = {}
    id_to_first_idx: list[int] = []  # maps i-th observation to the first index of the observation in the batch
    obs_to_annot_id = np.empty(len(batch["annotation_id"]), dtype=np.int32)
    for i, annot_id in enumerate(batch["annotation_id"]):
        if annot_id not in annot_to_id:
            annot_to_id[annot_id] = len(annot_to_id)
            id_to_first_idx.append(i)
        obs_to_annot_id[i] = annot_to_id[annot_id]
    id_to_first_idx = np.array(id_to_first_idx, dtype=np.int32)

    batch_size = len(batch["annotation_id"])
    # randomly sample nonmatching annotation IDs
    nonmatching_annot_ids = np.random.randint(len(id_to_first_idx)-1, size=batch_size)
    nonmatching_annot_ids[nonmatching_annot_ids >= obs_to_annot_id] += 1
    # find the first occurrence of the nonmatching annotation IDs
    nonmatching_idxs = id_to_first_idx[nonmatching_annot_ids]
    # construct the nonmatching text inputs
    nonmatching_text_inputs = {k: v[nonmatching_idxs] for k, v in batch["text_inputs"].items()}

    contrastive_batch = {
        "rgb": torch.tile(batch["rgb"], (2, 1, 1, 1)),
        "xyz": torch.tile(batch["xyz"], (2, 1, 1, 1)),
        "grasp_pose": torch.tile(batch["grasp_pose"], (2, 1, 1)),
        "text_inputs": {k: torch.cat([batch["text_inputs"][k], nonmatching_text_inputs[k]]) for k in batch["text_inputs"]}
    }

    labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)]).reshape(-1, 1).to(batch["rgb"].device)

    return contrastive_batch, labels

@hydra.main(version_base=None, config_path="../config", config_name="classification.yaml")
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

    grasp_classifier = GraspClassifier(config["model"]).to(device_id)
    model = DDP(grasp_classifier, device_ids=[device_id], find_unused_parameters=True)
    model.train()

    if rank == 0:
        print(f"Num trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"Num parameters: {sum(p.numel() for p in model.parameters()):,}")

    img_processor = grasp_classifier.create_rgb_processor()
    text_processor = grasp_classifier.create_text_processor()
    dataset = GraspDescriptionClassificationDataset(**config["train"]["dataset"], img_processor=img_processor, text_processor=text_processor)
    test_frac = config["train"]["test"]["frac"]
    if test_frac > 0:
        gen = torch.Generator().manual_seed(config["train"]["seed"])
        train_dataset, test_dataset = random_split(dataset, [1 - test_frac, test_frac], generator=gen)
        train_loader = create_dist_dataloader(train_dataset, rank, world_size, config)
        test_loader = create_dist_dataloader(test_dataset, rank, world_size, config)
    else:
        train_loader = create_dist_dataloader(dataset, rank, world_size, config)
        test_loader = None

    optimizer = optim.AdamW(model.parameters(), **config["train"]["optimizer"])
    lr_scheduler = WarmupCosineLR(
        optimizer,
        config["train"]["lr_schedule"]["warmup_steps"],
        config["train"]["steps"],
        config["train"]["lr_schedule"]["final_factor"]
    )

    checkpointer = Checkpointer(
        ckpt_dir,
        model=model.module,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )
    start_step = checkpointer.load()
    dist.barrier()

    metrics = {
        "f1": BinaryF1Score(),
        "accuracy": BinaryAccuracy(),
        "precision": BinaryPrecision()
    }
    for metric in metrics.values():
        metric.cuda()

    step = start_step
    with tqdm(total=config["train"]["steps"], initial=start_step, desc="Training", disable=rank != 0) as pbar:
        while step < config["train"]["steps"]:
            for batch_load_time, batch in time_iter(train_loader):
                optimizer.zero_grad()
                batch = move_to_device(batch, device_id)

                contrastive_batch, labels = construct_contrastive_batch(batch)
                rgb, xyz, grasp_pose = contrastive_batch["rgb"], contrastive_batch["xyz"], contrastive_batch["grasp_pose"]
                text_inputs = contrastive_batch["text_inputs"]

                infer_start = time.perf_counter()
                pred_logits: torch.Tensor = model(rgb, xyz, grasp_pose, text_inputs)
                infer_end = time.perf_counter()
                infer_time = infer_end - infer_start

                loss = F.binary_cross_entropy_with_logits(pred_logits, labels)
                variance = torch.var(pred_logits).item()

                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                info = {
                    "step": step,
                    "epoch": step // len(train_loader),
                    "lr": np.mean(lr_scheduler.get_last_lr()),
                }

                metric_values = {k: metric(F.sigmoid(pred_logits), labels).item() for k, metric in metrics.items()}
                info_to_gather = {
                    "loss": loss.item(),
                    "variance": variance,
                    "infer_time": infer_time,
                    "batch_load_time": batch_load_time,
                    **metric_values,
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
