import numpy as np
from PIL import Image
import json
from collections import defaultdict
import os
import argparse
import pickle
from tqdm import tqdm

from vlm_scorers import BaseGraspEvaluator, MolmoGraspEvaluator, MolmoPointingGraspEvaluator
from taskgrasp_utils import Scene, TaskGraspInfo

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--evaluator", nargs="+", default=["molmo_pointing"])
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("-c", "--grasp-conf-threshold", type=float, default=0.5)
    parser.add_argument("-d", "--data-dir", default="data/taskgrasp_scenes")
    parser.add_argument("-t", "--taskgrasp-dir", default="data/taskgrasp")
    parser.add_argument("-o", "--out-dir", default="eval/taskgrasp")
    return parser.parse_args()

def eval_scenes(evaluator: BaseGraspEvaluator, tg_info: TaskGraspInfo, scenes: list[Scene]):
    tasks, rgbs, depths, cam_infos, grasps = [], [], [], [], []
    for scene in scenes:
        obj_name = scene.name.split("-")[0]
        tasks.append(tg_info.get_task(obj_name))
        rgbs.append(scene.rgb)
        depths.append(scene.depth)
        cam_infos.append(scene.cam_info)
        grasps.append(scene.grasps)
    info_dicts = []
    grasp_idxs = evaluator.choose_grasp_batch(tasks, rgbs, depths, cam_infos, grasps, info_out=info_dicts)
    results = {}
    for scene, grasp_idx, info_dict in zip(scenes, grasp_idxs, info_dicts):
        obj_name = scene.name.split("-")[0]
        classification = tg_info.get_grasp_classification(obj_name, grasp_idx)
        results[scene.name] = {"grasp_idx": grasp_idx, "classification": classification, "info": info_dict}
    return results

def run_evaluator(data_dir: str, evaluator_name: str, tg_info: TaskGraspInfo, grasp_conf_threshold: float, batch_size: int, out_dir: str):
    os.makedirs(f"{out_dir}/{evaluator_name}", exist_ok=True)
    if evaluator_name == "molmo_pointing":
        evaluator = MolmoPointingGraspEvaluator()
    else:
        raise ValueError(f"Unknown evaluator: {evaluator_name}")
    print(f"Evaluating {evaluator_name}...")
    results = {}
    stats = {"correct": 0, "incorrect": 0, "unknown": 0}
    scene_names = sorted([sn for sn in os.listdir(data_dir) if tg_info.valid_obj(sn.split("-")[0])])
    for i in tqdm(range(0, len(scene_names), batch_size), desc=evaluator_name):
        scene_name_batch = scene_names[i:i+batch_size]
        scene_batch = [Scene(data_dir, scene, grasp_conf_threshold) for scene in scene_name_batch]
        batch_results = eval_scenes(evaluator, tg_info, scene_batch)
        results.update(batch_results)
        for scene, sample_result in batch_results.items():
            classification = sample_result["classification"]
            print(f"\t{scene}: {sample_result['grasp_idx']} -> {classification}")
            stats["correct"] += classification == 1
            stats["incorrect"] += classification == -1
            stats["unknown"] += classification == 0
    with open(f"{out_dir}/{evaluator_name}/results.pkl", "wb") as f:
        pickle.dump(results, f)
    with open(f"{out_dir}/{evaluator_name}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Done. Correct={stats['correct']/len(results)}, Incorrect={stats['incorrect']/len(results)}, Unknown={stats['unknown']/len(results)}")

def main():
    args = get_args()
    tg_info = TaskGraspInfo(args.taskgrasp_dir)
    for evaluator_name in args.evaluator:
        run_evaluator(args.data_dir, evaluator_name, tg_info, args.grasp_conf_threshold, args.batch_size, args.out_dir)

if __name__ == "__main__":
    main()
