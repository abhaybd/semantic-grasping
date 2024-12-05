import numpy as np
from PIL import Image
import argparse
from grasp_renderer import render_grasps

SCANS_DIR = "data/real_scans"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("obj_name")
    parser.add_argument("--conf_thresh", type=float, default=0.0)
    return parser.parse_args()

def main():
    args = get_args()
    obj_name = args.obj_name

    cam_info = np.load(f"{SCANS_DIR}/{obj_name}/cam_info.npy")
    depth = np.load(f"{SCANS_DIR}/{obj_name}/depth.npy")
    rgb = np.array(Image.open(f"{SCANS_DIR}/{obj_name}/rgb.png"))
    grasps = np.load(f"{SCANS_DIR}/{obj_name}/grasps.npy")
    grasp_confs = np.load(f"{SCANS_DIR}/{obj_name}/grasp_confs.npy")
    grasps = grasps[grasp_confs > args.conf_thresh]

    # TODO: filter for grasps on target object

    grasp_ids = np.random.choice(len(grasps), 2, replace=False)
    image = render_grasps(rgb, depth, cam_info, grasps[grasp_ids], np.array([[255, 0, 0], [0, 255, 0]]))

    Image.fromarray(image).save(f"{obj_name}.png")

if __name__ == "__main__":
    main()
