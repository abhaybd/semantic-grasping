from typing import List, Literal
import numpy as np
import re
from abc import ABC, abstractmethod
from collections import defaultdict
import time
import base64
import argparse
from xml.etree import ElementTree as ET

from scipy.spatial.distance import cdist
from PIL import Image, ImageOps
import torch
from transformers import AutoProcessor, GenerationConfig, AutoModelForCausalLM
import json5
from openai import OpenAI
from pydantic import BaseModel

import os
os.environ["HF_HOME"] = "/net/nfs2.prior/abhayd/huggingface_cache"

from grasp_renderer import GraspRenderer
from splitstream import splitfile
import io


class BaseGraspEvaluator(ABC):
    @abstractmethod
    def choose_grasp(self, task: str, rgb: np.ndarray, depth: np.ndarray, cam_info: np.ndarray, grasps: np.ndarray, info_out: dict=None) -> int:
        """
        rgb and depth are single images
        cam_info is 3x3 matrix of camera intrinsics
        grasps is a batch of grasp poses, in the camera body frame (i.e. +x=forward, +y=left, +z=up, NOT +z=forward)
        Returns index of optimal grasp
        """
        raise NotImplementedError

    def choose_grasp_batch(self, task: list[str], rgb: np.ndarray, depth: np.ndarray, cam_info: np.ndarray, grasps: np.ndarray, info_out: list[dict]=None) -> list[int]:
        """
        Batch evaluation, every argument is a list of the same length
        """
        assert len(task) == len(rgb) == len(depth) == len(cam_info) == len(grasps)
        ret = []
        for args in zip(task, rgb, depth, cam_info, grasps):
            if info_out is not None:
                info_out.append({})
            d = info_out[-1] if info_out is not None else None
            ret.append(self.choose_grasp(*args, info_out=d))
        return ret

class ComparisonGraspEvaluator(BaseGraspEvaluator):
    def choose_grasp(self, task: str, rgb: np.ndarray, depth: np.ndarray, cam_info: np.ndarray, grasps: np.ndarray) -> int:
        grasp_renderer = GraspRenderer(rgb, depth, cam_info)
        grasp_idxs = np.arange(len(grasps))
        np.random.shuffle(grasp_idxs)
        red_green = np.array([[255, 0, 0], [0, 255, 0]])
        while len(grasp_idxs) > 1:
            inference_imgs = []
            start = time.perf_counter()
            for i in range(0, len(grasp_idxs)-1, 2):
                inference_imgs.append(grasp_renderer.render(grasps[grasp_idxs[i:i+2]], red_green))
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

    def generate_prompt(self, task: str):
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

    def infer(self, task: str, images: np.ndarray) -> List[Literal["green", "red"]]:
        images = np.asarray(images)
        if images.ndim == 3:
            images = np.expand_dims(images, 0)
        assert images.ndim == 4 and images.shape[-1] == 3

        task_prompt = self.generate_prompt(task)
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

class MolmoPointingGraspEvaluator(BaseGraspEvaluator):
    def __init__(self):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained("allenai/Molmo-7B-D-0924", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
        self.model = AutoModelForCausalLM.from_pretrained("allenai/Molmo-7B-D-0924", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
        self.processor.tokenizer.padding_side = "left"

    def generate_prompt(self, task: str):
        return f"Point to where I should grasp to complete the task '{task}'."

    def grasp_img_points(self, grasps: np.ndarray, cam_info: np.ndarray):
        grasp_pos = grasps[:, :3, 3] + grasps[:, :3, 2] * 0.1
        grasp_px = grasp_pos @ cam_info.T
        grasp_px = grasp_px[:, :2] / grasp_px[:, 2:]
        grasp_px = grasp_px.astype(int)
        return grasp_px

    def parse_vlm_output(self, xml_str: str, img_shape: tuple) -> np.ndarray:
        """Returns (N,2) array of points"""
        try:
            f = io.BytesIO(xml_str.encode("utf-8"))
            xml_str = next(splitfile(f, format="xml"))
            root = ET.fromstring(xml_str)
            if root.tag == "point":
                point = np.array([round(float(root.get("x")) / 100 * img_shape[1]), round(float(root.get("y")) / 100 * img_shape[0])])
                points = np.expand_dims(point, 0)
            elif root.tag == "points":
                points = []
                i = 1
                while f"x{i}" in root.attrib:
                    points.append([round(float(root.get(f"x{i}")) / 100 * img_shape[1]), round(float(root.get(f"y{i}")) / 100 * img_shape[0])])
                    i += 1
                points = np.array(points)
            return points
        except ET.ParseError as e:
            print("Invalid XML:\n", xml_str)
            raise e

    def choose_grasp(self, task: str, rgb: np.ndarray, depth: np.ndarray, cam_info: np.ndarray, grasps: np.ndarray, info_out: dict=None) -> int:
        info_out_batch = None if info_out is None else []
        ret = self.choose_grasp_batch([task], [rgb], [depth], [cam_info], [grasps], info_out=info_out_batch)[0]
        if info_out is not None:
            info_out.update(info_out_batch[0])
        return ret

    def choose_grasp_batch(self, task_batch: list[str], rgb_batch, depth_batch, cam_info_batch, grasps_batch, info_out: list[dict]=None) -> list[int]:
        rgb_batch = np.asarray(rgb_batch)
        assert rgb_batch.ndim == 4 and rgb_batch.shape[-1] == 3

        task_prompts = [self.generate_prompt(t) for t in task_batch]
        inputs = defaultdict(list)
        for t, rgb in zip(task_prompts, rgb_batch):
            sample_input = self.processor.process(images=[rgb], text=t)
            for k, v in sample_input.items():
                inputs[k].append(v)
        inputs = self.processor.tokenizer.pad(inputs, return_tensors="pt")
        del inputs["attention_mask"] # TODO: this doesn't work for padding!
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(
                    max_new_tokens=2048,
                    stop_strings="<|endoftext|>",
                ),
                tokenizer=self.processor.tokenizer
            )
        output_tokens = output[:, inputs["input_ids"].size(1):]
        ret = []
        for i in range(len(output_tokens)):
            xml_str = self.processor.tokenizer.decode(output_tokens[i], skip_special_tokens=True)
            points = self.parse_vlm_output(xml_str, rgb_batch.shape[1:3])
            grasp_px = self.grasp_img_points(grasps_batch[i], cam_info_batch[i])
            dists = cdist(grasp_px, points, "euclidean")
            best_idx = np.argmin(np.min(dists, axis=1))
            ret.append(best_idx.item())
            if info_out is not None:
                info_out.append({"points": points, "grasp_px": grasp_px, "prompt": task_prompts[i]})
        return ret


class OpenAIGraspEvaluator(ComparisonGraspEvaluator):
    def __init__(self):
        self.client = OpenAI()

    def choose_from_grasps(self, task: str, rgb: np.ndarray, depth: np.ndarray, cam_info: np.ndarray, grasps: np.ndarray) -> int:
        print(task)
        grasp_renderer = GraspRenderer(rgb, depth, cam_info)
        grasp_idxs = np.arange(len(grasps))

        class Response(BaseModel):
            object_description: str
            grasp_descriptions: list[str]
            best_grasp_id: int
            explanation: str

        messages = [
            {
                "role": "developer",
                "content": f"You are a robot task with choosing the best grasp for the task \"{task}\", which are represented as grasps drawn in red on the image. These grasps will be provided to you as a list of images, and you must choose the best one by replying with the ID of the best grasp. If all grasps are equally unsuitable, default to grasp 0."
            }
        ]
        for i, idx in enumerate(grasp_idxs):
            img = grasp_renderer.render([grasps[idx]], [[255, 0, 0]])
            buf = io.BytesIO()
            Image.fromarray(img).save(f"tmp/grasp_{idx}.jpg")
            Image.fromarray(img).save(buf, format="JPEG")
            img_encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
            messages.extend([
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"This is grasp {i}."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_encoded}"
                            }
                        }
                    ]
                },
                {"role": "assistant", "content": "Got it."}
            ])
        messages.append({
            "role": "user",
            "content": "Which grasp should you perform?"
        })
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=messages,
            response_format=Response
        )
        print(completion.choices[0].message.parsed)
        return grasp_idxs[completion.choices[0].message.parsed.best_grasp_id]


    def infer(self, task: str, images: np.ndarray) -> List[Literal["green", "red"]]:
        images = np.asarray(images)
        if images.ndim == 3:
            images = np.expand_dims(images, 0)
        assert images.ndim == 4 and images.shape[-1] == 3

        class Response(BaseModel):
            image_description: str
            grasp_description: str
            explanation: str
            best_grasp_color: Literal["red" , "green" , "blue" , "yellow" , "cyan" , "magenta" , "brown" , "orange"]# , "pink" , "purple" , "white" , "black" , "gray" , "olive" , "teal" , "lavender"]

        ret = []
        for image in images:
            buf = io.BytesIO()
            Image.fromarray(image).save(buf, format="JPEG")
            encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    # {"role": "system", "content": f"You are a robot tasked with choosing the best grasp for the task \"{task}\", which are represented as colored grasps drawn on the image. The options are: red, green, blue, yellow, cyan, magenta, brown, orange, pink, purple, white, black, gray, olive, teal, and lavender. If all are equally unsuitable, default to red."},
                    {"role": "system", "content": f"You are a robot tasked with choosing the best grasp for the task \"{task}\", which are represented as colored grasps drawn on the image. The options are: red, green, blue, yellow, cyan, magenta, brown, and orange. If all are equally unsuitable, default to red."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Which grasp should you perform?"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded}"
                                }
                            }
                        ]
                    }
                ],
                response_format=Response
            )
            print(completion.choices[0].message.parsed)
            ret.append(completion.choices[0].message.parsed.best_grasp_color)
        return ret

def test_infer(evaluator: BaseGraspEvaluator):
    if isinstance(evaluator, ComparisonGraspEvaluator):
        task = "grasp a pan to cook something"
        image = np.asarray(Image.open(f"pan.png"))
        print("Eval for pan.png:", evaluator.infer(task, image))

def test_choose_grasp(evaluator: BaseGraspEvaluator):
    SCANS_DIR = "data/taskgrasp_scenes"
    obj_name = "023_pan-view2"
    cam_info = np.load(f"{SCANS_DIR}/{obj_name}/cam_info.npy")
    depth = np.load(f"{SCANS_DIR}/{obj_name}/depth.npy")
    rgb = np.array(Image.open(f"{SCANS_DIR}/{obj_name}/rgb.png"))
    grasps = np.load(f"{SCANS_DIR}/{obj_name}/grasps.npy")
    grasp_confs = np.load(f"{SCANS_DIR}/{obj_name}/grasp_confs.npy")
    grasps = grasps[grasp_confs > 0.4]

    if not isinstance(evaluator, MolmoPointingGraspEvaluator):
        grasps = grasps[np.random.choice(len(grasps), 16, replace=False)]

    grasp_idx = evaluator.choose_grasp("grasp a pan to saute something", rgb, depth, cam_info, grasps)
    print("Chose grasp:", grasp_idx)
    gr = GraspRenderer(rgb, depth, cam_info)
    img = gr.render([grasps[grasp_idx]], [[0, 255, 0]])
    Image.fromarray(img).save("chosen_grasp.png")

    render_grasps = grasps[np.random.choice(len(grasps), min(32, len(grasps)), replace=False)]
    img = gr.render(render_grasps, np.array([[255, 0, 0]]*len(render_grasps)))
    Image.fromarray(img).save("all_grasps.png")

def test_openai(evaluator: OpenAIGraspEvaluator):
    from taskgrasp_utils import TaskGraspInfo
    SCANS_DIR = "data/taskgrasp_scenes"
    tg_info = TaskGraspInfo("data/taskgrasp")
    # obj_name = "001_squeezer-view1"
    obj_name = "009_pan-view1"
    cam_info = np.load(f"{SCANS_DIR}/{obj_name}/cam_info.npy")
    depth = np.load(f"{SCANS_DIR}/{obj_name}/depth.npy")
    rgb = np.array(Image.open(f"{SCANS_DIR}/{obj_name}/rgb.png"))
    grasps = np.load(f"{SCANS_DIR}/{obj_name}/grasps.npy")
    grasp_confs = np.load(f"{SCANS_DIR}/{obj_name}/grasp_confs.npy")
    grasps = grasps[grasp_confs > 0.4]

    # grasps = grasps[np.random.choice(len(grasps), 16, replace=False)]

    grasp_idx = evaluator.choose_from_grasps(f"grasp a {obj_name.split('-')[0].split('_',1)[1]} to {tg_info.get_task_verb(obj_name.split('-')[0])} something", rgb, depth, cam_info, grasps)
    print("Chose grasp:", grasp_idx)
    gr = GraspRenderer(rgb, depth, cam_info)
    img = gr.render([grasps[grasp_idx]], [[0, 255, 0]])
    Image.fromarray(img).save("chosen_grasp.png")

    render_grasps = grasps[np.random.choice(len(grasps), min(32, len(grasps)), replace=False)]
    img = gr.render(render_grasps, np.array([[255, 0, 0]]*len(render_grasps)))
    Image.fromarray(img).save("all_grasps.png")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluator", choices=["molmo", "openai", "molmo_pointing"])
    return parser.parse_args()

def main():
    args = get_args()
    if args.evaluator == "molmo":
        evaluator = MolmoGraspEvaluator()
    elif args.evaluator == "openai":
        evaluator = OpenAIGraspEvaluator()
        test_openai(evaluator)
    elif args.evaluator == "molmo_pointing":
        evaluator = MolmoPointingGraspEvaluator()
    else:
        raise ValueError(f"Invalid evaluator: {args.evaluator}")
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    # test_infer(evaluator)
    # test_choose_grasp(evaluator)

if __name__ == "__main__":
    main()
