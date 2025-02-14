import argparse
import numpy as np
import pickle
import cv2

from PIL import Image

from taskgrasp_utils import TaskGraspInfo, Scene

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluator")
    parser.add_argument("-e", "--eval", default="taskgrasp")
    parser.add_argument("-o", "--obj")
    return parser.parse_args()

def main():
    args = get_args()

    with open(f"eval/{args.eval}/{args.evaluator}/results.pkl", "rb") as f:
        results = pickle.load(f)
    failures = {k:v for k,v in results.items() if v["classification"] != 1}
    assert args.obj is None or args.obj in failures
    obj = args.obj or np.random.choice(list(failures.keys()))
    print(f"Object: {obj}")
    print("Prompt:", results[obj]["info"]["prompt"])

    tg_info = TaskGraspInfo("data/taskgrasp")
    scene = Scene("data/taskgrasp_scenes", obj, 0)

    img = scene.rgb.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
    for i, grasp_px in enumerate(results[obj]["info"]["grasp_px"]):
        grasp_px = grasp_px.astype(int)
        classification = tg_info.get_grasp_classification(obj.split("-")[0], i)
        if classification == 1:
            color = (0, 255, 0)
        elif classification == -1:
            color = (0, 0, 255)
        else:
            color = (128, 128, 128)
        cv2.circle(img, tuple(grasp_px), 5, color, -1)
    for point_px in results[obj]["info"]["points"]:
        point_px = point_px.astype(int)
        cv2.circle(img, tuple(point_px), 5, (255, 0, 0), -1)
    cv2.imshow("grasps", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
