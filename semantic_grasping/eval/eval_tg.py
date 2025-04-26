from functools import cache
import os

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from PIL import Image

from semantic_grasping_datagen.eval.utils import TaskGraspScanLibrary

from semantic_grasping.utils import tqdm, build_wandb_config
from semantic_grasping.eval.molmo_local_pred import LocalPredictor
from semantic_grasping.eval.molmo_pred import depth_to_pc

class TGEvalModelConfig(BaseModel):
    name: str
    ckpt_dir: str
    prompt_pfx: str

class TGEvalConfig(BaseModel):
    tg_dir: str
    out_dir: str
    split: str
    batch_size: int
    eval_model: TGEvalModelConfig

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

def random_eval_fold(tg_library: TaskGraspScanLibrary, split_dir: str, fold: str, all_view_labels: dict[tuple[str, int], dict[str, set[int]]]):
    view_labels = filter_view_labels_for_fold(tg_library, all_view_labels, split_dir, fold)

    eval_data = []
    for (object_id, view_id), task_grasps in view_labels.items():
        for task_verb, grasp_ids in task_grasps.items():
            eval_data.append((object_id, view_id, task_verb, grasp_ids))

    succ = []
    for object_id, view_id, task_verb, grasp_ids in eval_data:
        sample = get_sample(tg_library, object_id, view_id)
        grasps = sample["registered_grasps"]
        succ.append(len(grasp_ids) / len(grasps))
    return {
        "split": os.path.basename(split_dir),
        "fold": fold,
        "n_samples": len(eval_data),
        "n_succ": sum(succ),
    }

def eval_fold(tg_library: TaskGraspScanLibrary, predictor: LocalPredictor, split_dir: str, fold: str, all_view_labels: dict[tuple[str, int], dict[str, set[int]]], batch_size: int):
    view_labels = filter_view_labels_for_fold(tg_library, all_view_labels, split_dir, fold)

    eval_data = []
    for (object_id, view_id), task_grasps in view_labels.items():
        for task_verb, grasp_ids in task_grasps.items():
            eval_data.append((object_id, view_id, task_verb, grasp_ids))

    n_succ = 0
    n_samples = 0
    results_data = {}

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
                images.append(sample["rgb"])
                pc = depth_to_pc(sample["depth"], sample["cam_params"])
                pcs.append(pc)
                grasps.append(sample["registered_grasps"])
                cam_Ks.append(sample["cam_params"])
                tasks.append(f"Grasp the {sample['object_name']} to {task_verb}")

            pred_grasp_ids = predictor.pred_grasp(images, pcs, tasks, grasps, cam_Ks, verbosity=3)
            for j in range(len(batch_eval_data)):
                pred_grasp_id = pred_grasp_ids[j]
                _, _, _, grasp_ids = batch_eval_data[j]
                if pred_grasp_id is not None and pred_grasp_id in grasp_ids:
                    n_succ += 1
                    succ_pred_viz.append((images[j], tasks[j]))
                else:
                    fail_pred_viz.append((images[j], tasks[j]))
                n_samples += 1
            pbar.update(len(batch_eval_data))
            pbar.set_description(f"Evaluating fold {fold} (top-1 acc={n_succ}/{n_samples}={n_succ / n_samples:.1%})")
    results_data["n_samples"] = n_samples
    results_data["n_succ"] = n_succ
    results_data["accuracy"] = n_succ / n_samples

    print(f"Fold {fold} top-1 accuracy: {n_succ}/{len(eval_data)}={n_succ / len(eval_data):.1%}")
    return results_data, succ_pred_viz, fail_pred_viz

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

    out_dir = os.path.join(config.out_dir, config.eval_model.name, ckpt_name)
    os.makedirs(out_dir, exist_ok=True)

    task_name = os.getenv("GANTRY_TASK_NAME", None)
    run = wandb.init(
        entity="prior-ai2",
        project="semantic-grasping",
        config=build_wandb_config(cfg),
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

    fold_results = {}
    accs = []
    succ_viz: list[wandb.Image] = []
    fail_viz: list[wandb.Image] = []
    for fold in sorted(os.listdir(split_dir)):
        fold_results[fold], succ_pred_viz, fail_pred_viz = eval_fold(tg_library, predictor, split_dir, fold, view_labels, config.batch_size)
        succ_viz.extend([wandb.Image(image, caption=task) for image, task in succ_pred_viz])
        fail_viz.extend([wandb.Image(image, caption=task) for image, task in fail_pred_viz])
        accs.append(fold_results[fold]["accuracy"])
    run.summary["fold_results"] = fold_results
    acc = sum(accs) / len(accs)
    run.summary["accuracy"] = acc
    print(f"Average top-1 accuracy: {acc:.1%}")

    if len(succ_viz) > 0:
        run.log({"succ_predictions": succ_viz})
    if len(fail_viz) > 0:
        run.log({"fail_predictions": fail_viz})

if __name__ == "__main__":
    main()
