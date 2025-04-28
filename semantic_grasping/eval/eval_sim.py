import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import random

from pydantic import BaseModel
from PIL import Image
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import h5py
import pandas as pd
import numpy as np
import numpy.typing as npt

from semantic_grasping.utils import build_wandb_config, tqdm
from semantic_grasping.eval.molmo_local_pred import LocalPredictor

class EvalModelConfig(BaseModel):
    name: str
    ckpt_dir: str
    prompt_pfx: str

class SimEvalConfig(BaseModel):
    obs_dir: str
    task_dir: str
    batch_size: int
    eval_model: EvalModelConfig
    far_clip: float
    dataloader_workers: int

class Batch(BaseModel, arbitrary_types_allowed=True):
    images: list[Image.Image]
    pcs: list[npt.NDArray[np.floating]]
    grasps: list[npt.NDArray[np.floating]]
    cam_Ks: list[npt.NDArray[np.floating]]
    tasks: list[str]
    labels: list[int]

def load_data(scene_path: str, view_id: str, far_clip: float):
    with h5py.File(scene_path, "r") as f:
        rgb = f[view_id]["rgb"][:]
        image = Image.fromarray(rgb)
        xyz: npt.NDArray[np.floating] = f[view_id]["xyz"][:]
        cam_K: npt.NDArray[np.floating] = f[view_id]["cam_params"][:]

        xyz_mask = (xyz[:, :, 2] > 0) & (xyz[:, :, 2] < far_clip)
        pc = xyz[xyz_mask]
        grasps_list: list[npt.NDArray[np.floating]] = []

        n_obs = sum(1 for obs_id in f[view_id].keys() if obs_id.startswith("obs_"))
        for i in range(n_obs):
            obs_id = f"obs_{i}"
            assert obs_id in f[view_id].keys()
            grasp = f[view_id][obs_id]["grasp_pose"][:]
            grasps_list.append(grasp)

    grasps = np.stack(grasps_list, axis=0)
    return image, pc, grasps, cam_K

def load_batch(config: SimEvalConfig, batch_df: pd.DataFrame):
    images = []
    pcs = []
    tasks = []
    grasps = []
    cam_Ks = []
    labels = []
    obs_dir = config.obs_dir
    for _, row in batch_df.iterrows():
        scene_path = os.path.join(obs_dir, row["scene_path"])
        image, pc, view_grasps, cam_K = load_data(scene_path, row["view_id"], config.far_clip)
        images.append(image)
        pcs.append(pc)
        grasps.append(view_grasps)
        cam_Ks.append(cam_K)
        tasks.append(row["task"])
        labels.append(int(row["obs_id"].split("_")[-1]))
    return Batch(
        images=images,
        pcs=pcs,
        grasps=grasps,
        cam_Ks=cam_Ks,
        tasks=tasks,
        labels=labels
    )

def eval_batch(predictor: LocalPredictor, batch: Batch):
    preds = predictor.pred_grasp(batch.images, batch.pcs, batch.tasks, batch.grasps, batch.cam_Ks, verbosity=3)
    succ_pred_viz = []
    fail_pred_viz = []
    n_succ = 0
    n_samples = 0
    for i in range(len(batch.images)):
        if preds[i] is not None and preds[i] == batch.labels[i]:
            succ_pred_viz.append((batch.images[i], batch.tasks[i]))
            n_succ += 1
        else:
            fail_pred_viz.append((batch.images[i], batch.tasks[i]))
        n_samples += 1
    results = {
        "n_samples": n_samples,
        "n_succ": n_succ,
    }
    return results, succ_pred_viz, fail_pred_viz


@hydra.main(version_base=None, config_path="../../config", config_name="eval_sim.yaml")
def main(cfg: DictConfig):
    if missing_keys := OmegaConf.missing_keys(cfg):
        raise ValueError(f"Missing keys: {missing_keys}")
    print(OmegaConf.to_yaml(cfg))
    config = SimEvalConfig(**OmegaConf.to_object(cfg))

    ckpt_dir = config.eval_model.ckpt_dir
    if "shard" in os.path.basename(ckpt_dir):
        ckpt_name = "-".join(os.path.normpath(ckpt_dir).split(os.sep)[-2:])
    else:
        ckpt_name = os.path.basename(ckpt_dir)

    df = pd.read_csv(os.path.join(config.task_dir, "matched_tasks.csv"))
    prompt_pfx = config.eval_model.prompt_pfx
    if config.eval_model.name == "graspmolmo" and "_evals" not in ckpt_name:
        print("WARN: Adding robot_control: instruction: prefix to prompts")
        prompt_pfx = "robot_control: instruction: " + prompt_pfx
    predictor = LocalPredictor(config.eval_model.ckpt_dir, prompt_pfx)

    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    task_name = os.getenv("GANTRY_TASK_NAME", None)
    run = wandb.init(
        entity="prior-ai2",
        project="semantic-grasping",
        config=build_wandb_config(cfg),
        name=task_name,
        dir=out_dir,
        job_type="eval_sim"
    )

    succ_viz: list[wandb.Image] = []
    fail_viz: list[wandb.Image] = []
    results = {
        "n_samples": 0,
        "n_succ": 0,
    }
    with ProcessPoolExecutor(config.dataloader_workers) as executor:
        # n_samples = len(df)
        n_samples = 2 * config.batch_size  # TODO: remove
        futures = [executor.submit(load_batch, config, df.iloc[i:i+config.batch_size]) for i in range(0, n_samples, config.batch_size)]
        for future in tqdm(as_completed(futures), total=len(futures)):
            batch: Batch = future.result()
            batch_results, succ_pred_viz, fail_pred_viz = eval_batch(predictor, batch)
            succ_viz.extend([wandb.Image(image, caption=task) for image, task in succ_pred_viz])
            fail_viz.extend([wandb.Image(image, caption=task) for image, task in fail_pred_viz])
            for k, v in batch_results.items():
                if k in results:
                    results[k] += v
    run.summary["results"] = results
    run.summary["accuracy"] = results["n_succ"] / results["n_samples"]
    print(f"Average top-1 accuracy: {results['n_succ'] / results['n_samples']:.1%}")

    if len(succ_viz) > 0:
        if len(succ_viz) > 100:
            random.shuffle(succ_viz)
            succ_viz = succ_viz[:100]
        run.log({"succ_predictions": succ_viz})
    if len(fail_viz) > 0:
        if len(fail_viz) > 100:
            random.shuffle(fail_viz)
            fail_viz = fail_viz[:100]
        run.log({"fail_predictions": fail_viz})

if __name__ == "__main__":
    main()
