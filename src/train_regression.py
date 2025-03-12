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

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from tqdm import tqdm

from model import GraspEncoder, Checkpointer, WarmupCosineLR
from data import GraspDescriptionRegressionDataset

@torch.no_grad()
@torch.autocast("cuda", dtype=torch.bfloat16)
def test(model: nn.Module, test_loader: DataLoader):
    model.eval()
    losses = []
    classification_losses = []
    variances = []
    for batch in tqdm(test_loader, desc="Test", leave=False):
        rgb, xyz, grasp_pose = batch["rgb"].cuda(), batch["xyz"].cuda(), batch["grasp_pose"].cuda()
        text_embedding = batch["text_embedding"].float().cuda()
        grasp_features = model(rgb, xyz, grasp_pose)
        batch_loss = 1.0 - F.cosine_similarity(grasp_features, text_embedding, dim=-1)
        losses.extend(batch_loss.tolist())
        variances.append(torch.var(grasp_features, dim=0).mean().item())

        pairwise_similarity = text_embedding @ grasp_features.T  # (i, j) => i-th text embedding, j-th grasp feature

        annotation_ids = batch["annotation_id"]
        annotation_id_idxs: dict[str, list[int]] = {}
        for i, annotation_id in enumerate(annotation_ids):
            if annotation_id not in annotation_id_idxs:
                annotation_id_idxs[annotation_id] = []
            annotation_id_idxs[annotation_id].append(i)
        gt_matrix = torch.zeros_like(pairwise_similarity)
        for i, annot_id in enumerate(annotation_ids):
            gt_matrix[i, annotation_id_idxs[annot_id]] = 1.0
        classification_loss = F.binary_cross_entropy_with_logits(pairwise_similarity * 100, gt_matrix)
        classification_losses.append(classification_loss.item())
    return {
        "loss": np.mean(losses),
        "classification_loss": np.mean(classification_losses),
        "variance": np.mean(variances),
    }

def nested_dict_to_flat_dict(d: dict[str, Any], pfx: str = ""):
    flat_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flat_dict.update(nested_dict_to_flat_dict(v, pfx=f"{pfx}{k}/"))
        else:
            flat_dict[f"{pfx}{k}"] = v
    return flat_dict

@hydra.main(version_base=None, config_path="../config", config_name="regression.yaml")
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
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        name=task_name,
        dir=out_dir,
        job_type="train",
        id=run_id,
        resume="allow"
    )

    model = torch.nn.DataParallel(GraspEncoder(config["grasp_encoder"]))
    model.cuda()
    print("Compiling model...")
    torch.compile(model)
    print("Done!")

    img_processor = model.module.create_rgb_processor()
    dataset = GraspDescriptionRegressionDataset(**config["train"]["dataset"], img_processor=img_processor)
    test_frac = config["train"]["test"]["frac"]
    if test_frac > 0:
        gen = torch.Generator().manual_seed(config["train"]["seed"])
        train_dataset, test_dataset = random_split(dataset, [1 - test_frac, test_frac], generator=gen)
        train_loader = DataLoader(train_dataset, shuffle=True, **config["train"]["dataloader"])
        test_loader = DataLoader(test_dataset, shuffle=False, **config["train"]["dataloader"])
    else:
        train_loader = DataLoader(dataset, shuffle=True, **config["train"]["dataloader"])
        test_loader = None

    optimizer = optim.AdamW(model.parameters(), **config["train"]["optimizer"])
    lr_scheduler = WarmupCosineLR(optimizer, config["train"]["warmup_epochs"], config["train"]["epochs"])

    checkpointer = Checkpointer(ckpt_dir, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    start_epoch = checkpointer.load()

    for epoch in tqdm(range(start_epoch, config["train"]["epochs"]), total=config["train"]["epochs"], initial=start_epoch):
        losses = []
        infer_times = []
        variances = []
        model.train()
        for batch in tqdm(train_loader, desc="Batch", leave=False):
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rgb, xyz, grasp_pose = batch["rgb"].cuda(), batch["xyz"].cuda(), batch["grasp_pose"].cuda()
                text_embedding = batch["text_embedding"].cuda()
                infer_start = time.perf_counter()
                grasp_features = model(rgb, xyz, grasp_pose)
                infer_end = time.perf_counter()
                infer_times.append(infer_end - infer_start)
                loss = 1.0 - F.cosine_similarity(grasp_features, text_embedding, dim=-1).mean()
            with torch.no_grad():
                variances.append(torch.var(grasp_features, dim=0).mean().item())
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        info = {
            "loss": np.mean(losses),
            "lr": np.mean(lr_scheduler.get_last_lr()),
            "variance": np.mean(variances),
            "infer_time": np.mean(infer_times),
        }

        if test_loader is not None and config["train"]["test"]["period"] and epoch % config["train"]["test"]["period"] == 0:
            test_results = test(model, test_loader)
            info["test"] = test_results

        run.log(nested_dict_to_flat_dict(info))

        if config["train"]["save_period"] and epoch % config["train"]["save_period"] == 0:
            checkpointer.save(epoch)

    save_path = os.path.join(out_dir, "grasp_encoder.pt")
    torch.save(model.state_dict(), save_path)
    wandb.save(save_path, base_path=out_dir)

if __name__ == "__main__":
    main()
