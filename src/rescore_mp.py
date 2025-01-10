import argparse
import pickle
import os
from scipy.spatial.distance import cdist
import numpy as np
import json
from matplotlib import pyplot as plt

from taskgrasp_utils import TaskGraspInfo

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dir")
    return parser.parse_args()

def eval_scene(obj_name: str, result: dict, tg_info: TaskGraspInfo):
    info = result["info"]
    dists = cdist(info["grasp_px"], info["points"], "euclidean")
    pointed_grasps = np.argsort(dists, axis=0)[:2].flatten()

    for grasp_idx in pointed_grasps:
        if tg_info.get_grasp_classification(obj_name, grasp_idx) == 1:
            result["grasp_idx"] = grasp_idx
            result["classification"] = 1
            return True
    return False

def main():
    args = get_args()

    with open(os.path.join(args.eval_dir, "results.pkl"), "rb") as f:
        results: dict = pickle.load(f)

    tg_info = TaskGraspInfo("data/taskgrasp")

    n_correct = 0
    n_incorrect = 0
    for scene_name, result in results.items():
        obj_name = scene_name.split("-")[0]
        if tg_info.valid_obj(obj_name):
            if eval_scene(obj_name, result, tg_info):
                n_correct += 1
            else:
                n_incorrect += 1
        else:
            del results[scene_name]

    print(f"Top-k grasp success rate: {n_correct / (n_correct + n_incorrect)}")
    with open(os.path.join(args.eval_dir, f"stats_topk.json"), "w") as f:
        json.dump({"correct": n_correct, "incorrect": n_incorrect, "unknown": 0}, f, indent=2)
    with open(os.path.join(args.eval_dir, f"results_topk.pkl"), "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()
