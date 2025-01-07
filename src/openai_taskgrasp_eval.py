import argparse
import io
import base64
import os
import json
import pickle
import multiprocessing as mp

from tqdm import tqdm
from PIL import Image
from pydantic import BaseModel
from openai import OpenAI
from openai.lib._pydantic import to_strict_json_schema

from grasp_renderer import SceneRenderer
from taskgrasp_utils import TaskGraspInfo, Scene

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir", default="data/taskgrasp_scenes")
    parser.add_argument("-t", "--taskgrasp-dir", default="data/taskgrasp")
    parser.add_argument("-o", "--out-dir", default="eval/taskgrasp/openai")

    subparser = parser.add_subparsers(required=True, dest="subcommand")
    submit_parser = subparser.add_parser("submit")
    submit_parser.add_argument("description", nargs="?")
    submit_parser.add_argument("-b", "--batch-size", type=int, default=64)
    submit_parser.set_defaults(func=submit_job)

    get_parser = subparser.add_parser("get")
    get_parser.add_argument("job_id")
    get_parser.set_defaults(func=get_result)

    return parser.parse_args()

class Response(BaseModel):
    object_description: str
    grasp_descriptions: list[str]
    best_grasp_id: int
    explanation: str

def init_submit_proc(width: int, height: int, tg_dir: str):
    globals()["renderer"] = SceneRenderer(width, height)
    globals()["tg_info"] = TaskGraspInfo(tg_dir)

def submit_proc(data_dir: str, scene_name: str):
    scene = Scene(data_dir, scene_name, 0.0)
    tg_info = globals()["tg_info"]
    if tg_info.valid_obj(scene.obj_name):
        return generate_query(tg_info, scene, globals()["renderer"])
    else:
        return None

def submit_job(args, tg_info: TaskGraspInfo, client: OpenAI):
    batchinput_path = os.path.join(args.out_dir, "batchinput.jsonl")
    if not os.path.isfile(batchinput_path):
        queries = []
        height, width = Scene(args.data_dir, os.listdir(args.data_dir)[0], 0.0).rgb.shape[:2]
        with mp.Pool(args.batch_size, initializer=init_submit_proc, initargs=(width, height, args.taskgrasp_dir)) as pool:
            futures = []
            queue = mp.Queue()
            for scene_name in os.listdir(args.data_dir):
                futures.append(pool.apply_async(submit_proc, (args.data_dir, scene_name), callback=queue.put, error_callback=queue.put))
            for _ in tqdm(range(len(futures))):
                query = queue.get()
                if query is not None:
                    if isinstance(query, dict):
                        queries.append(query)
                    else:
                        print(query)
        with open(batchinput_path, "w") as f:
            for query in queries:
                f.write(json.dumps(query) + "\n")

    batchinput_file = client.files.create(file=open(batchinput_path, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=batchinput_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"Submitted batch job with id: {batch.id}")

def get_result(args, tg_info: TaskGraspInfo, client: OpenAI):
    results_file_path = f"{args.out_dir}/response.jsonl"
    if not os.path.isfile(results_file_path):
        batch = client.batches.retrieve(args.job_id)
        if batch.status != "completed":
            print(f"Batch job {args.job_id} is not completed. Status: {batch.status}")
            return
        results_file = client.files.content(batch.output_file_id)
        results_file.write_to_file(results_file_path)

    with open(results_file_path) as f:
        results_lines = f.read().splitlines()
    results = {}
    stats = {"correct": 0, "incorrect": 0, "unknown": 0}
    for line in results_lines:
        result = json.loads(line)
        scene: str = result["custom_id"]
        response = json.loads(result["response"]["body"]["choices"][0]["message"]["content"])
        grasp_idx = response["best_grasp_id"]
        classification = tg_info.get_grasp_classification(scene.split("-")[0], grasp_idx)
        results[scene] = {
            "grasp_idx": grasp_idx,
            "classification": classification,
            "info": response
        }
        stats["correct"] += classification == 1
        stats["incorrect"] += classification == -1
        stats["unknown"] += classification == 0
    with open(f"{args.out_dir}/results.pkl", "wb") as f:
        pickle.dump(results, f)
    with open(f"{args.out_dir}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Correct={stats['correct']/len(results)}, Incorrect={stats['incorrect']/len(results)}, Unknown={stats['unknown']/len(results)}")

def generate_query(tg_info: TaskGraspInfo, scene: Scene, renderer: SceneRenderer):
    task = tg_info.get_task(scene.obj_name)

    messages = [
        {
            "role": "developer",
            "content": f"You are a robot task with choosing the best grasp for the task \"{task}\", which are represented as grasps drawn in red on the image. These grasps will be provided to you as a list of images, and you must choose the best one by replying with the ID of the best grasp. If all grasps are equally unsuitable, default to grasp 0."
        }
    ]
    renderer.set_scene(scene.rgb, scene.depth, scene.cam_info)
    for i, grasp in enumerate(scene.grasps):
        img = renderer.render([grasp], [[255, 0, 0]])
        buf = io.BytesIO()
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

    schema = to_strict_json_schema(Response)
    request = {
        "custom_id": scene.name,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o",
            "messages": messages,
            "max_tokens": 8192,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "grasp_response",
                    "schema": schema
                }
            }
        }
    }
    return request

def main():
    args = get_args()

    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    tg_info = TaskGraspInfo(args.taskgrasp_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    client = OpenAI()

    args.func(args, tg_info, client)


if __name__ == "__main__":
    main()
