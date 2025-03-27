import os
import time
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from tqdm import tqdm
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision

from model import GraspClassifier, Checkpointer, WarmupCosineLR
from data import GraspDescriptionClassificationDataset, GraspDescriptionClassificationSampler

@torch.no_grad()
@torch.autocast("cuda", dtype=torch.bfloat16)
def test(model: nn.Module, test_loader: DataLoader):
    model.eval()
    losses = []
    variances = []
    metrics = {
        "f1": BinaryF1Score(),
        "accuracy": BinaryAccuracy(),
        "precision": BinaryPrecision()
    }
    for metric in metrics.values():
        metric.cuda()
    for batch in tqdm(test_loader, desc="Test", leave=False):
        rgb, xyz, grasp_pose = batch["rgb"].cuda(), batch["xyz"].cuda(), batch["grasp_pose"].cuda()
        labels = batch["label"].cuda()
        text_inputs = {k: v.cuda() for k, v in batch["text_inputs"].items()}
        pred_logits: torch.Tensor = model(rgb, xyz, grasp_pose, text_inputs)
        batch_loss = F.binary_cross_entropy_with_logits(pred_logits, labels)
        losses.append(batch_loss.item())
        variances.append(torch.var(pred_logits, dim=0).mean().item())

        for metric in metrics.values():
            metric(F.sigmoid(pred_logits), labels)
    model.train()
    return {
        "loss": np.mean(losses),
        "variance": np.mean(variances),
        **{k: metric.compute().item() for k, metric in metrics.items()},
    }

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

@hydra.main(version_base=None, config_path="../config", config_name="classification.yaml")
def main(config: DictConfig):
    torch.manual_seed(config["train"]["seed"])
    print(OmegaConf.to_yaml(config))
    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    ckpt_dir = os.path.join(out_dir, "checkpoints")
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

    model = torch.nn.DataParallel(GraspClassifier(config["model"]))
    model.cuda()
    model.train()
    # TODO: add torch compilation

    img_processor = model.module.create_rgb_processor()
    text_processor = model.module.create_text_processor()
    test_frac = config["train"]["test"]["frac"]
    if test_frac > 0:
        train_dataset, test_dataset = GraspDescriptionClassificationDataset.load_split(
            img_processor=img_processor,
            text_processor=text_processor,
            **config["train"]["dataset"],
            fracs=[1 - test_frac, test_frac],
            seed=config["train"]["seed"]
        )
        train_sampler = GraspDescriptionClassificationSampler(train_dataset)
        test_sampler = GraspDescriptionClassificationSampler(test_dataset)
        train_loader = DataLoader(train_dataset, sampler=train_sampler, persistent_workers=True, pin_memory=True, **config["train"]["dataloader"])
        test_loader = DataLoader(test_dataset, sampler=test_sampler, persistent_workers=True, pin_memory=True, **config["train"]["dataloader"])
    else:
        dataset = GraspDescriptionClassificationDataset.load(img_processor=img_processor, text_processor=text_processor, **config["train"]["dataset"])
        sampler = GraspDescriptionClassificationSampler(dataset)
        train_loader = DataLoader(dataset, sampler=sampler, persistent_workers=True, pin_memory=True, **config["train"]["dataloader"])
        test_loader = None

    optimizer = optim.AdamW(model.parameters(), **config["train"]["optimizer"])
    lr_scheduler = WarmupCosineLR(
        optimizer,
        config["train"]["lr_schedule"]["warmup_steps"],
        config["train"]["steps"],
        config["train"]["lr_schedule"]["final_factor"]
    )

    scaler = torch.GradScaler(**config["train"]["grad_scaler"])

    checkpointer = Checkpointer(
        ckpt_dir,
        model=model.module,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler
    )
    start_step = checkpointer.load()

    metrics = {
        "f1": BinaryF1Score(),
        "accuracy": BinaryAccuracy(),
        "precision": BinaryPrecision()
    }
    for metric in metrics.values():
        metric.cuda()

    step = start_step
    with tqdm(total=config["train"]["steps"], initial=start_step, desc="Training") as pbar:
        while step < config["train"]["steps"]:
            batch_load_start = time.perf_counter()
            for batch in train_loader:
                batch_load_end = time.perf_counter()
                batch_load_time = batch_load_end - batch_load_start
                optimizer.zero_grad()
                rgb, xyz, grasp_pose = batch["rgb"].cuda(), batch["xyz"].cuda(), batch["grasp_pose"].cuda()
                text_inputs = {k: v.cuda() for k, v in batch["text_inputs"].items()}
                labels = batch["label"].cuda()
                train_step_start = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16, **config["train"]["autocast"]):
                    infer_start = time.perf_counter()
                    pred_logits: torch.Tensor = model(rgb, xyz, grasp_pose, text_inputs)
                    infer_end = time.perf_counter()
                    infer_time = infer_end - infer_start
                    nanmask = torch.isnan(pred_logits)
                    if not nanmask.all():
                        pred_logits = pred_logits[~nanmask]
                        labels = labels[~nanmask]
                    loss = F.binary_cross_entropy_with_logits(pred_logits, labels)
                variance = torch.var(pred_logits).item()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                train_step_end = time.perf_counter()
                train_step_time = train_step_end - train_step_start

                metric_values = {k: metric(F.sigmoid(pred_logits), labels).item() for k, metric in metrics.items()}
                info = {
                    "epoch": step // len(train_loader),
                    "loss": loss.item(),
                    "lr": np.mean(lr_scheduler.get_last_lr()),
                    "variance": variance,
                    "infer_time": infer_time,
                    "train_step_time": train_step_time,
                    "batch_load_time": batch_load_time,
                    "nan_frac": nanmask.float().mean().item(),
                    "grad_scale": scaler.get_scale(),
                    **metric_values,
                }

                if test_loader is not None and config["train"]["test"]["period"] and step % config["train"]["test"]["period"] == 0:
                    test_results = test(model, test_loader)
                    info["test"] = test_results

                run.log(nested_dict_to_flat_dict(info))

                if config["train"]["save_period"] and step % config["train"]["save_period"] == 0:
                    checkpointer.save(step)

                step += 1
                pbar.update(1)
                if step >= config["train"]["steps"]:
                    break
                if torch.isnan(loss):
                    raise ValueError("Loss is NaN")
                batch_load_start = time.perf_counter()

    save_path = os.path.join(out_dir, "model.pt")
    torch.save(model.module.state_dict(), save_path)
    wandb.save(save_path, base_path=out_dir)

if __name__ == "__main__":
    main()
