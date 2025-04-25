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
    parser.add_argument("--random", action="store_true")
    return parser.parse_args()

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

def eval_fold(tg_library: TaskGraspScanLibrary, predictor: MolmoLocalPredictor, split_dir: str, fold: str, all_view_labels: dict[tuple[str, int], dict[str, set[int]]], batch_size: int):
    view_labels = filter_view_labels_for_fold(tg_library, all_view_labels, split_dir, fold)

    eval_data = []
    for (object_id, view_id), task_grasps in view_labels.items():
        for task_verb, grasp_ids in task_grasps.items():
            eval_data.append((object_id, view_id, task_verb, grasp_ids))

    n_succ = 0
    n_samples = 0
    results_data = {
        "results": {},
        "ckpt_dir": predictor.ckpt_dir,
        "split": os.path.basename(split_dir),
        "n_samples": 0,
        "n_succ": 0,
        "fold": fold,
    }
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
            pbar.set_description(f"Evaluating fold {fold} (top-1 acc={n_succ}/{n_samples}={n_succ / n_samples:.1%})")
    results_data["n_samples"] = n_samples
    results_data["n_succ"] = n_succ

    print(f"Fold {fold} top-1 accuracy: {n_succ}/{len(eval_data)}={n_succ / len(eval_data):.1%}")
    return results_data

def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tg_library = TaskGraspScanLibrary(args.tg_dir)

    split_dir = os.path.join(args.tg_dir, "splits_final", args.split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory {split_dir} not found")

    # (object_id, view_id) -> {task_verb -> set of positive grasp_ids}
    view_labels = parse_view_labels(tg_library, os.path.join(args.tg_dir, "task2_results.txt"))

    if args.random:
        eval_results = {}
        accs = []
        for fold in sorted(os.listdir(split_dir)):
            eval_results[fold] = random_eval_fold(tg_library, split_dir, fold, view_labels)
            accs.append(eval_results[fold]["n_succ"] / eval_results[fold]["n_samples"])
        print(f"Average top-1 accuracy: {sum(accs) / len(accs):.1%}")
        with open(os.path.join(args.out_dir, f"results_random_{args.split}.json"), "w") as f:
            json.dump(eval_results, f, indent=2)
    else:
        predictor = MolmoLocalPredictor(args.ckpt_dir)
        eval_results = {}
        accs = []
        for fold in sorted(os.listdir(split_dir)):
            eval_results[fold] = eval_fold(tg_library, predictor, split_dir, fold, view_labels, args.batch_size)
            accs.append(eval_results[fold]["n_succ"] / eval_results[fold]["n_samples"])
        print(f"Average top-1 accuracy: {sum(accs) / len(accs):.1%}")

        with open(os.path.join(args.out_dir, f"results_{args.split}.json"), "w") as f:
            json.dump(eval_results, f, indent=2)

if __name__ == "__main__":
    main()
