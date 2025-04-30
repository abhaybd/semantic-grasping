from functools import cache
import os
import random
import json

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from PIL import Image
import numpy as np

from semantic_grasping_datagen.eval.utils import TaskGraspScanLibrary

from semantic_grasping.utils import tqdm, build_wandb_config
from semantic_grasping.eval.molmo_local_pred import LocalPredictor
from semantic_grasping.eval.utils import depth_to_pc, draw_grasp, draw_grasp_points

class TGEvalModelConfig(BaseModel):
    name: str
    ckpt_dir: str
    prompt_pfx: str

class TGEvalConfig(BaseModel):
    tg_dir: str
    split: str
    batch_size: int
    eval_model: TGEvalModelConfig
    fold: str | None = None

def parse_view_labels(tg_library: TaskGraspScanLibrary, path: str):
    view_labels: dict[tuple[str, int], dict[str, set[int]]] = {}  # (object_id, view_id) -> {task_verb -> set of positive grasp_ids}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            part1, label = line.split(":")
            obj_id, grasp_id, task_verb = part1.split("-")
            if label == "1":
                view_ids = tg_library.get_views(obj_id)
                for view_id in view_ids:
                    if (obj_id, view_id, "_registered_grasps.npy") in tg_library:
                        k = (obj_id, view_id)
                        if k not in view_labels:
                            view_labels[k] = {}
                        if task_verb not in view_labels[k]:
                            view_labels[k][task_verb] = set()
                        view_labels[k][task_verb].add(int(grasp_id))
    return view_labels

def filter_view_labels_for_fold(tg_library: TaskGraspScanLibrary, view_labels: dict[tuple[str, int], dict[str, set[int]]], split_dir: str, fold: str):
    trained_view_labels = parse_view_labels(tg_library, os.path.join(split_dir, fold, "train_split.txt"))
    new_view_labels: dict[tuple[str, int], dict[str, set[int]]] = {}
    for key, task_map in view_labels.items():
        if key not in trained_view_labels:
            new_view_labels[key] = task_map
            continue
        new_view_labels[key] = {}
        for task_verb, grasp_ids in task_map.items():
            if task_verb not in trained_view_labels[key]:
                new_view_labels[key][task_verb] = grasp_ids
            else:
                new_view_labels[key][task_verb] = grasp_ids - trained_view_labels[key][task_verb]
            if len(new_view_labels[key][task_verb]) == 0:
                del new_view_labels[key][task_verb]
        if len(new_view_labels[key]) == 0:
            del new_view_labels[key]
    for k, d in view_labels.items():
        for t, s in d.items():
            assert len(s) > 0, f"No grasp ids for {k} and {t}"
    return new_view_labels

@cache
def get_sample(tg_library: TaskGraspScanLibrary, object_id: str, view_id: int):
    return tg_library.get(object_id, view_id)

def eval_fold(tg_library: TaskGraspScanLibrary, predictor: LocalPredictor, split_dir: str, fold: str, all_view_labels: dict[tuple[str, int], dict[str, set[int]]], batch_size: int):
    view_labels = filter_view_labels_for_fold(tg_library, all_view_labels, split_dir, fold)

    eval_data = []
    for (object_id, view_id), task_grasps in view_labels.items():
        for task_verb, grasp_ids in task_grasps.items():
            eval_data.append((object_id, view_id, task_verb, grasp_ids))

    n_succ = 0
    n_samples = 0
    results_data = {}
    eval_results = []

    fail_pred_viz: list[tuple[Image.Image, str]] = []
    succ_pred_viz: list[tuple[Image.Image, str]] = []
    with tqdm(total=len(eval_data)) as pbar:
        for i in range(0, len(eval_data), batch_size):
            batch_eval_data = eval_data[i:i+batch_size]
            images = []
            pcs = []
            tasks = []
            grasps = []
            cam_Ks = []

            for object_id, view_id, task_verb, grasp_ids in batch_eval_data:
                sample = get_sample(tg_library, object_id, view_id)
                images.append(sample["rgb"].copy())
                pc = depth_to_pc(sample["depth"], sample["cam_params"])
                pcs.append(pc)
                grasps.append(sample["registered_grasps"])
                cam_Ks.append(sample["cam_params"])
                tasks.append(f"Grasp the {sample['object_name']} to {task_verb}")

            pred_grasp_ids = predictor.pred_grasp(images, pcs, tasks, grasps, cam_Ks, verbosity=3)
            for j in range(len(batch_eval_data)):
                pred_grasp_id = pred_grasp_ids[j]
                object_id, view_id, task_verb, grasp_ids = batch_eval_data[j]
                image = images[j]

                eval_results.append({
                    "object_id": object_id,
                    "view_id": view_id,
                    "task_verb": task_verb,
                    "grasp_ids": list(grasp_ids),
                    "pred_grasp_id": pred_grasp_id if pred_grasp_id is not None else -1,
                    "success": pred_grasp_id is not None and pred_grasp_id in grasp_ids,
                })

                grasp_mask = np.zeros(len(grasps[j]), dtype=bool)
                grasp_mask[list(grasp_ids)] = True
                draw_grasp_points(image, cam_Ks[j], pcs[j], grasps[j][grasp_mask], color="green")
                draw_grasp_points(image, cam_Ks[j], pcs[j], grasps[j][~grasp_mask], color="red")
                if pred_grasp_id is not None:
                    draw_grasp(image, cam_Ks[j], grasps[j][pred_grasp_id], color="blue")

                if pred_grasp_id is not None and pred_grasp_id in grasp_ids:
                    n_succ += 1
                    succ_pred_viz.append((image, tasks[j]))
                else:
                    fail_pred_viz.append((image, tasks[j]))
                n_samples += 1
            pbar.update(len(batch_eval_data))
            pbar.set_description(f"Evaluating fold {fold} (top-1 acc={n_succ}/{n_samples}={n_succ / n_samples:.1%})")
    results_data["n_samples"] = n_samples
    results_data["n_succ"] = n_succ
    results_data["accuracy"] = n_succ / n_samples

    print(f"Fold {fold} top-1 accuracy: {n_succ}/{len(eval_data)}={n_succ / len(eval_data):.1%}")
    return results_data, eval_results, succ_pred_viz, fail_pred_viz

@hydra.main(version_base=None, config_path="../../config", config_name="eval_tg.yaml")
def main(cfg: DictConfig):
    if missing_keys := OmegaConf.missing_keys(cfg):
        raise ValueError(f"Missing keys: {missing_keys}")
    print(OmegaConf.to_yaml(cfg))
    config = TGEvalConfig(**OmegaConf.to_object(cfg))

    ckpt_dir = config.eval_model.ckpt_dir
    if "shard" in os.path.basename(ckpt_dir):
        ckpt_name = "-".join(os.path.normpath(ckpt_dir).split(os.sep)[-2:])
    else:
        ckpt_name = os.path.basename(ckpt_dir)

    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    task_name = os.getenv("GANTRY_TASK_NAME", None)
    wandb_cfg = build_wandb_config(cfg)
    wandb_cfg["ckpt_name"] = ckpt_name
    run = wandb.init(
        entity="prior-ai2",
        project="semantic-grasping",
        config=wandb_cfg,
        name=task_name,
        dir=out_dir,
        job_type="eval_tg"
    )

    tg_library = TaskGraspScanLibrary(config.tg_dir)
    split_dir = os.path.join(config.tg_dir, "splits_final", config.split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory {split_dir} not found")

    # (object_id, view_id) -> {task_verb -> set of positive grasp_ids}
    view_labels = parse_view_labels(tg_library, os.path.join(config.tg_dir, "task2_results.txt"))

    prompt_pfx = config.eval_model.prompt_pfx
    if config.eval_model.name == "graspmolmo" and "_evals" not in ckpt_name:
        print("WARN: Adding robot_control: instruction: prefix to prompts")
        prompt_pfx = "robot_control: instruction: " + prompt_pfx
    predictor = LocalPredictor(config.eval_model.ckpt_dir, prompt_pfx)

    summary_results = {}
    eval_results = {}
    accs = []
    succ_viz: list[wandb.Image] = []
    fail_viz: list[wandb.Image] = []
    for fold in sorted(os.listdir(split_dir)):
        if config.fold is not None and fold != config.fold:
            continue
        summary_results[fold], eval_results[fold], succ_pred_viz, fail_pred_viz = eval_fold(tg_library, predictor, split_dir, fold, view_labels, config.batch_size)
        succ_viz.extend([wandb.Image(image, caption=task) for image, task in succ_pred_viz])
        fail_viz.extend([wandb.Image(image, caption=task) for image, task in fail_pred_viz])
        accs.append(summary_results[fold]["accuracy"])
    with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f)
    run.summary["fold_results"] = summary_results
    acc = sum(accs) / len(accs)
    run.summary["accuracy"] = acc
    print(f"Average top-1 accuracy: {acc:.1%}")

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
