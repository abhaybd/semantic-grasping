import json
from collections import defaultdict

import numpy as np
from PIL import Image

class TaskGraspInfo:
    def __init__(self, taskgrasp_path: str):
        with open(f"{taskgrasp_path}/task1_results.json") as f:
            self.obj_tasks = json.load(f)
        self.obj_task_grasps = defaultdict(lambda: (set(), set()))
        with open(f"{taskgrasp_path}/task2_results.txt") as f:
            for line in f.read().strip().splitlines():
                obj_grasp_task, classification = line.split(":")
                obj, grasp, task = obj_grasp_task.split("-")
                if classification == "1":
                    self.obj_task_grasps[(obj, task)][0].add(int(grasp))
                elif classification == "-1":
                    self.obj_task_grasps[(obj, task)][1].add(int(grasp))

    def valid_obj(self, obj: str):
        obj_name = obj.split("_", 1)[1]
        return obj_name in self.obj_tasks and (obj, self.get_task_verb(obj)) in self.obj_task_grasps

    def num_correct_grasps(self, obj: str):
        return len(self.obj_task_grasps[(obj, self.get_task_verb(obj))][0])

    def get_task_verb(self, obj_name: str):
        obj_class = obj_name.split("_", 1)[1]
        return self.obj_tasks[obj_class]

    def get_task(self, obj_name: str):
        obj_class = obj_name.split("_", 1)[1]
        return f"grasp a {obj_class} to {self.obj_tasks[obj_class]} something"

    def get_grasp_classification(self, obj: str, grasp: int):
        task = self.get_task_verb(obj)
        correct, incorrect = self.obj_task_grasps[(obj, task)]
        if grasp in correct:
            return 1
        elif grasp in incorrect:
            return -1
        else:
            return 0

class Scene:
    def __init__(self, data_dir: str, scene: str, grasp_conf_threshold: float):
        self.name = scene
        scene_dir = f"{data_dir}/{scene}"
        self.cam_info = np.load(f"{scene_dir}/cam_info.npy")
        self.rgb = np.asarray(Image.open(f"{scene_dir}/rgb.png"))
        self.depth = np.load(f"{scene_dir}/depth.npy")
        self.grasps = np.load(f"{scene_dir}/grasps.npy")
        self.grasp_confs = np.load(f"{scene_dir}/grasp_confs.npy")
        self.grasps = self.grasps[self.grasp_confs > grasp_conf_threshold]

    @property
    def obj_name(self):
        return self.name.split("-")[0]

    @property
    def obj_class(self):
        return self.obj_name.split("_", 1)[1]
