import argparse
import io
import base64
import os
import json
import pickle
import multiprocessing as mp
import glob
import re

import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from PIL import Image
from pydantic import BaseModel
from openai import OpenAI
from openai.lib._pydantic import to_strict_json_schema

from grasp_renderer import SceneRenderer
from taskgrasp_utils import TaskGraspInfo, Scene
from vlm_scorers import OpenAIGraspEvaluator

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pointing_results_path")
    parser.add_argument("-k", "--top-k", type=int, default=2)
    parser.add_argument("-b", "--batch-size", type=int, default=32)

    parser.add_argument("-d", "--data-dir", default="data/taskgrasp_scenes")
    parser.add_argument("-t", "--taskgrasp-dir", default="data/taskgrasp")
    parser.add_argument("-o", "--out-dir", default="eval/taskgrasp/hybrid")

    return parser.parse_args()

class Response(BaseModel):
    object_description: str
    grasp_descriptions: list[str]
    best_grasp_id: int
    explanation: str

def init_eval_proc(tg_dir: str, pointing_results_path: str):
    globals()["tg_info"] = TaskGraspInfo(tg_dir)
    globals()["evaluator"] = OpenAIGraspEvaluator()
    with open(pointing_results_path, "rb") as f:
        globals()["pointing_results"] = pickle.load(f)

def eval_proc(data_dir: str, scene_name: str, k: int):
    scene = Scene(data_dir, scene_name, 0.0)
    tg_info: TaskGraspInfo = globals()["tg_info"]
    evaluator: OpenAIGraspEvaluator = globals()["evaluator"]
    if tg_info.valid_obj(scene.obj_name):
        return eval_scene(tg_info, scene, k, evaluator, globals()["pointing_results"][scene_name])
    else:
        return None

def eval_scene(tg_info: TaskGraspInfo, scene: Scene, k: int, evaluator: OpenAIGraspEvaluator, pointing_result: dict):
    task = tg_info.get_task(scene.obj_name)

    dists = cdist(pointing_result["info"]["grasp_px"], pointing_result["info"]["points"], "euclidean")
    pointed_grasp_ids = np.argsort(dists, axis=0)[:k].flatten()
    pointed_grasp_ids = sorted(set(pointed_grasp_ids.tolist()))  # dedupe

    info = {}
    # evaluator.scene = scene.name
    # evaluator.count = 0
    grasp_id = evaluator.choose_grasp(task, scene.rgb, scene.depth, scene.cam_info, scene.grasps[pointed_grasp_ids], info_out=info)
    grasp_id = pointed_grasp_ids[grasp_id]

    classification = tg_info.get_grasp_classification(scene.obj_name, grasp_id)
    result = {
        "grasp_idx": grasp_id,
        "classification": classification,
        "info": {
            **pointing_result["info"],
            "pointed_grasp_idxs": pointed_grasp_ids,
            **info
        }
    }
    return scene.name, result

def main():
    args = get_args()

    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    if args.batch_size > 1:
        with mp.Pool(args.batch_size, initializer=init_eval_proc, initargs=(args.taskgrasp_dir, args.pointing_results_path)) as pool:
            futures = []
            queue = mp.Queue()
            for scene_name in os.listdir(args.data_dir):
                futures.append(pool.apply_async(eval_proc, (args.data_dir, scene_name, args.top_k), callback=queue.put, error_callback=queue.put))
            for _ in tqdm(range(len(futures)), smoothing=0.0, dynamic_ncols=True):
                result = queue.get()
                if result is not None:
                    if isinstance(result, tuple):
                        scene_name, result = result
                        results[scene_name] = result
                    else:
                        print(result)
    else:
        init_eval_proc(args.taskgrasp_dir, args.pointing_results_path)
        for scene_name in ["153_spatula-view3"]:#os.listdir(args.data_dir):
            result = eval_proc(args.data_dir, scene_name, args.top_k)
            if result is not None:
                scene_name, result = result
                results[scene_name] = result
                print(scene_name, result)

    stats = {"correct": 0, "incorrect": 0, "unknown": 0}
    for result in results.values():
        classification = result["classification"]
        stats["correct"] += classification == 1
        stats["incorrect"] += classification == -1
        stats["unknown"] += classification == 0
    with open(f"{args.out_dir}/results.pkl", "wb") as f:
        pickle.dump(results, f)
    with open(f"{args.out_dir}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Correct={stats['correct']/len(results)}, Incorrect={stats['incorrect']/len(results)}, Unknown={stats['unknown']/len(results)}")

if __name__ == "__main__":
    main()
