from typing import List, Literal
import numpy as np
import re
from abc import ABC, abstractmethod
from collections import defaultdict
import time

from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, GenerationConfig, AutoModelForCausalLM
import json5

import os
os.environ["HF_HOME"] = "/net/nfs2.prior/abhayd/huggingface_cache"

from grasp_renderer import render_grasps_pc, img_to_pc
from splitstream import splitfile
import io


def generate_prompt(task: str):
    # return f"Which grasp, red or green, is more suitable for the task of \"{task}\"? Answer the question through identifying the grasp that is the most suitable, finding grasps which are definitely not suitable, and providing a ranking from most to least preferable for the grasps. First, describe where the grasps are located, then give the question response, then provide the explanation. Finally, in a single word say the color of the best grasp."
    # return re.sub(r"\s+", " ", f"""
    return f"""
Which grasp, red or green, is more suitable for the task of "{task}"? If both are equally unsuitable, default to red.

Answer the following question in JSON format, with the following structure:
{{
    "image_description": "This should describe the image, and identify objects relevant to the task.",
    "grasp_description": "This should describe the red and green grasps, where they are placed relative to the relevant object, and how that affects the answer to this question.",
    "explanation": "This should describe why the most suitable grasp is the best, and why the other grasps are not suitable.",
    "best_grasp_color": "This should be a single word, red or green, indicating the color of the best grasp."
}}

Only provide a single JSON object as the answer, with no other text.
""".strip()

class BaseGraspEvaluator(ABC):
    @abstractmethod
    def choose_grasp(self, task: str, rgb: np.ndarray, depth: np.ndarray, cam_info: np.ndarray, grasps: np.ndarray) -> int:
        """
        rgb and depth are single images
        cam_info is 3x3 matrix of camera intrinsics
        grasps is a batch of grasp poses, in the camera body frame (i.e. +x=forward, +y=left, +z=up, NOT +z=forward)
        Returns index of optimal grasp
        """
        raise NotImplementedError

class ComparisonGraspEvaluator(BaseGraspEvaluator):
    def choose_grasp(self, task: str, rgb: np.ndarray, depth: np.ndarray, cam_info: np.ndarray, grasps: np.ndarray) -> int:
        grasp_idxs = np.arange(len(grasps))
        np.random.shuffle(grasp_idxs)
        red_green = np.array([[255, 0, 0], [0, 255, 0]])
        pc = img_to_pc(rgb, depth, cam_info)
        while len(grasp_idxs) > 1:
            inference_imgs = []
            start = time.perf_counter()
            for i in range(0, len(grasp_idxs)-1, 2):
                inference_imgs.append(render_grasps_pc(rgb, pc, cam_info, grasps[grasp_idxs[i:i+2]], red_green))
            print(f"Num samples: {len(inference_imgs)}, Render time: {time.perf_counter() - start:.3f}s")
            preds = self.infer(task, inference_imgs)
            new_grasp_idxs = []
            for i, pred in enumerate(preds):
                if pred == "green":
                    new_grasp_idxs.append(grasp_idxs[i*2+1])
                elif pred == "red":
                    new_grasp_idxs.append(grasp_idxs[i*2])
                else:
                    raise ValueError(f"Invalid prediction: {pred}")
            grasp_idxs = new_grasp_idxs
        return grasp_idxs[0]

    @abstractmethod
    def infer(self, task: str, image: np.ndarray) -> List[Literal["green", "red"]]:
        raise NotImplementedError

def parse_response(response: str):
    response = re.sub(r"\",?\n", "\",\n", response)
    f = io.BytesIO(response.encode("utf-8"))
    response_json = next(splitfile(f, format="json"))
    return json5.loads(response_json)

class MolmoGraspEvaluator(ComparisonGraspEvaluator):
    def __init__(self):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained("allenai/Molmo-7B-D-0924", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
        self.model = AutoModelForCausalLM.from_pretrained("allenai/Molmo-7B-D-0924", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")

    def infer(self, task: str, images: np.ndarray) -> List[Literal["green", "red"]]:
        images = np.asarray(images)
        if images.ndim == 3:
            images = np.expand_dims(images, 0)
        assert images.ndim == 4 and images.shape[-1] == 3

        task_prompt = generate_prompt(task)
        inputs = defaultdict(list)
        import time
        start = time.perf_counter()
        for img in images:
            sample_input = self.processor.process(images=[img], text=task_prompt)
            for k, v in sample_input.items():
                inputs[k].append(v)
        inputs = {k: torch.stack(v, dim=0).to(self.model.device) for k, v in inputs.items()}
        preprocess_time = time.perf_counter() - start

        start = time.perf_counter()
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=512, stop_strings="<|endoftext|>"), tokenizer=self.processor.tokenizer
            )
        infer_time = time.perf_counter() - start
        print(f"Num samples: {len(images)}, Preprocess time: {preprocess_time:.3f}s, Infer time: {infer_time:.3f}s")

        ret = []
        for o in output:
            text: str = self.processor.tokenizer.decode(o, skip_special_tokens=True)
            response = text.rsplit("Assistant: ")[-1].strip()
            try:
                pred_json = parse_response(response)
            except ValueError as e:
                print("Invalid JSON:\n", response)
                raise e
            pred = pred_json["best_grasp_color"].lower()
            assert pred in {"green", "red"}, f"Invalid prediction:\n{response}"
            ret.append(pred)
        return ret


def test_infer(evaluator: ComparisonGraspEvaluator):
    task = "grasp a pan to cook something"
    image = np.asarray(Image.open(f"pan.png"))
    print("Eval for pan.png:", evaluator.infer(task, image))

def test_choose_grasp(evaluator: BaseGraspEvaluator):
    from viz_scans import SCANS_DIR
    from grasp_renderer import render_grasps
    obj_name = "pan"
    cam_info = np.load(f"{SCANS_DIR}/{obj_name}/cam_info.npy")
    depth = np.load(f"{SCANS_DIR}/{obj_name}/depth.npy")
    rgb = np.array(Image.open(f"{SCANS_DIR}/{obj_name}/rgb.png"))
    grasps = np.load(f"{SCANS_DIR}/{obj_name}/grasps.npy")
    grasp_confs = np.load(f"{SCANS_DIR}/{obj_name}/grasp_confs.npy")
    grasps = grasps[grasp_confs > 0.4]

    grasps = grasps[np.random.choice(len(grasps), 16, replace=False)]

    grasp_idx = evaluator.choose_grasp("grasp a pan to cook something", rgb, depth, cam_info, grasps)
    print("Chose grasp:", grasp_idx)
    img = render_grasps(rgb, depth, cam_info, [grasps[grasp_idx]], [[0, 255, 0]])
    Image.fromarray(img).save("chosen_grasp.png")

    img = render_grasps(rgb, depth, cam_info, grasps, np.array([[255, 0, 0]]*len(grasps)))
    Image.fromarray(img).save("all_grasps.png")


def main():
    evaluator = MolmoGraspEvaluator()
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    test_infer(evaluator)
    test_choose_grasp(evaluator)

if __name__ == "__main__":
    main()
