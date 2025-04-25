import argparse
from functools import cache
import json
import os

from semantic_grasping_datagen.eval.utils import TaskGraspScanLibrary

from semantic_grasping.utils import tqdm
from semantic_grasping.eval.molmo_local_pred import MolmoLocalPredictor
from semantic_grasping.eval.molmo_pred import depth_to_pc


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("tg_dir")
    parser.add_argument("ckpt_dir")
    parser.add_argument("out_dir")
    parser.add_argument("split")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tg_library = TaskGraspScanLibrary(args.tg_dir)
    predictor = MolmoLocalPredictor(args.ckpt_dir)

    @cache
    def get_sample(object_id: str, view_id: int):
        return tg_library.get(object_id, view_id)

    split_dir = os.path.join(args.tg_dir, "splits_final", args.split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory {split_dir} not found")

    view_labels: dict[tuple[str, int], dict[str, set[int]]] = {}  # (object_id, view_id) -> {task_verb -> set of positive grasp_ids}
    for fold in tqdm(os.listdir(split_dir), desc="Loading view labels from folds"):
        with open(os.path.join(split_dir, fold, "test_split.txt"), "r") as f:
            for line in f:
                part1, label = line.strip().split(":")
                obj_id, grasp_id, task_verb = part1.split("-")
                view_ids = tg_library.get_views(obj_id)
                if label == "1":
                    for view_id in view_ids:
                        if (obj_id, view_id, "_registered_grasps.npy") in tg_library:
                            k = (obj_id, view_id)
                            if k not in view_labels:
                                view_labels[k] = {}
                            if task_verb not in view_labels[k]:
                                view_labels[k][task_verb] = set()
                            view_labels[k][task_verb].add(int(grasp_id))

    eval_data = []
    for (object_id, view_id), task_grasps in view_labels.items():
        for task_verb, grasp_ids in task_grasps.items():
            eval_data.append((object_id, view_id, task_verb, grasp_ids))

    n_succ = 0
    n_samples = 0
    results_data = {
        "results": {},
        "ckpt_dir": args.ckpt_dir,
        "split": args.split,
    }
    with tqdm(total=len(eval_data)) as pbar:
        for i in range(0, len(eval_data), args.batch_size):
            batch_eval_data = eval_data[i:i+args.batch_size]
            images = []
            pcs = []
            tasks = []
            grasps = []
            cam_Ks = []

            for object_id, view_id, task_verb, grasp_ids in batch_eval_data:
                sample = get_sample(object_id, view_id)
                images.append(sample["rgb"])
                pc = depth_to_pc(sample["depth"], sample["cam_params"])
                pcs.append(pc)
                grasps.append(sample["registered_grasps"])
                cam_Ks.append(sample["cam_params"])
                tasks.append(f"Grasp the {sample['object_name']} to {task_verb}")

            pred_grasp_ids = predictor.pred_grasp(images, pcs, tasks, grasps, cam_Ks)
            for pred_grasp_id, (object_id, view_id, task_verb, grasp_ids) in zip(pred_grasp_ids, batch_eval_data):
                results_data["results"][f"{object_id}-{view_id}-{task_verb}"] = {
                    "pred_grasp_id": pred_grasp_id,
                    "gt_grasp_ids": list(grasp_ids),
                    "success": pred_grasp_id in grasp_ids
                }
                if pred_grasp_id in grasp_ids:
                    n_succ += 1
                n_samples += 1
            pbar.update(len(batch_eval_data))
            pbar.set_description(f"Evaluating (top-1 acc={n_succ}/{n_samples}={n_succ / n_samples:.1%})")

    print(f"Top-1 accuracy: {n_succ}/{len(eval_data)}={n_succ / len(eval_data):.1%}")

    with open(os.path.join(args.out_dir, f"results_{args.split}.json"), "w") as f:
        json.dump(results_data, f, indent=2)

if __name__ == "__main__":
    main()
