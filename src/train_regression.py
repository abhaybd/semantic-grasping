import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from tqdm import tqdm
from model import GraspEncoder, Checkpointer
from data import GraspDescriptionRegressionDataset

@hydra.main(version_base=None, config_path="../config", config_name="regression.yaml")
def main(config: DictConfig):
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
    model.train()
    print("Compiling model...")
    torch.compile(model)
    print("Done!")

    img_processor = model.module.create_rgb_processor()
    dataset = GraspDescriptionRegressionDataset(**config["train"]["dataset"], img_processor=img_processor)
    train_loader = DataLoader(dataset, shuffle=True, **config["train"]["dataloader"])

    optimizer = optim.Adam(model.parameters(), **config["train"]["optimizer"])

    checkpointer = Checkpointer(ckpt_dir, model, optimizer)
    start_epoch = checkpointer.load()

    for epoch in tqdm(range(start_epoch, config["train"]["epochs"]), total=config["train"]["epochs"], initial=start_epoch):
        losses = []
        infer_times = []
        for batch in tqdm(train_loader, desc="Batch", leave=False):
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rgb, xyz, grasp_pose = batch["rgb"].cuda(), batch["xyz"].cuda(), batch["grasp_pose"].cuda()
                text_embedding = batch["text_embedding"].cuda()
                infer_start = time.perf_counter()
                grasp_features = model(rgb, xyz, grasp_pose)
                infer_end = time.perf_counter()
                infer_times.append(infer_end - infer_start)
                batch_loss = -F.cosine_similarity(grasp_features, text_embedding, dim=-1)
                loss = batch_loss.mean()
            loss.backward()
            optimizer.step()
            losses.extend(batch_loss.tolist())

        run.log({"loss": np.mean(losses), "infer_time": np.mean(infer_times)})

        if config["train"]["save_period"] and epoch % config["train"]["save_period"] == 0:
            checkpointer.save(epoch)

    save_path = os.path.join(out_dir, "grasp_encoder.pt")
    torch.save(model.state_dict(), save_path)
    wandb.save(save_path, base_path=out_dir)

if __name__ == "__main__":
    main()
