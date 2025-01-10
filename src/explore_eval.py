import argparse
import pickle

from PIL import Image
import cv2

from taskgrasp_utils import TaskGraspInfo, Scene
from grasp_renderer import GraspRenderer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dir")
    parser.add_argument("--tg-dir", default="data/taskgrasp")
    parser.add_argument("--data-dir", default="data/taskgrasp_scenes")

    subparser = parser.add_subparsers()

    view_parser = subparser.add_parser("view")
    choice_parser = view_parser.add_mutually_exclusive_group(required=True)
    choice_parser.add_argument("-i", "--idx", type=int)
    choice_parser.add_argument("-n", "--name")
    view_parser.add_argument("-g", "--draw-grasp", action="store_true")
    view_parser.set_defaults(func=view)

    list_parser = subparser.add_parser("list")
    list_parser.set_defaults(func=list_results)

    return parser.parse_args()

def view(args, results, tg_info):
    if args.idx is not None:
        misses = sorted([(k, v) for k, v in results.items() if v["classification"] == -1])
        name, results = misses[args.idx]
    else:
        name = args.name
        results = results[name]
    obj_name = name.split("-")[0]

    scene = Scene(args.data_dir, name, 0.0)
    if "grasp_px" in results["info"] and "points" in results["info"] and not args.draw_grasp:
        rgb = scene.rgb.copy()
        for i, grasp_px in enumerate(results["info"]["grasp_px"]):
            classification = tg_info.get_grasp_classification(obj_name, i)
            if classification == -1:
                cv2.circle(rgb, tuple(grasp_px), 3, (255, 0, 0), -1)
            elif classification == 1:
                cv2.circle(rgb, tuple(grasp_px), 3, (0, 255, 0), -1)
            else:
                cv2.circle(rgb, tuple(grasp_px), 3, (255, 255, 255), -1)
        for point_px in results["info"]["points"]:
            cv2.circle(rgb, tuple(point_px), 5, (0, 0, 255), -1)
    else:
        renderer = GraspRenderer(scene.rgb, scene.depth, scene.cam_info, mesh=False)
        idx = results["grasp_idx"]
        rgb = renderer.render([scene.grasps[idx]], [[255, 0, 0]])
    Image.fromarray(rgb).save(f"{args.eval_dir}/miss_{name}.png")

def list_results(args, results, tg_info):
    results = sorted(results.items())
    correct = [(k, v) for k, v in results if v["classification"] == 1]
    incorrect = [(k, v) for k, v in results if v["classification"] == -1]
    unknown = [(k, v) for k, v in results if v["classification"] == 0]

    print(f"Correct: {len(correct)}")
    for correct_name, _ in correct:
        print(f"\t{correct_name}")

    print(f"Incorrect: {len(incorrect)}")
    for incorrect_name, _ in incorrect:
        print(f"\t{incorrect_name}")
    
    print(f"Unknown: {len(unknown)}")
    for unknown_name, _ in unknown:
        print(f"\t{unknown_name}")


def main():
    args = get_args()

    with open(f"{args.eval_dir}/results.pkl", "rb") as f:
        results = pickle.load(f)

    tg_info = TaskGraspInfo(args.tg_dir)

    args.func(args, results, tg_info)

if __name__ == "__main__":
    main()
