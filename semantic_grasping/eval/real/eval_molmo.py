import argparse
import os

from PIL import Image
import numpy as np

from semantic_grasping_datagen.eval.utils import TaskGraspScanLibrary
from semantic_grasping.eval.molmo_remote_pred import ZeroShotMolmoModal, GraspMolmoModal, GraspMolmoBeaker


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-shot", action="store_true")
    return parser.parse_args()

def depth_to_pc(depth: np.ndarray, cam_K: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    depth_mask = (depth > 0)
    uvd = np.stack((u, v, np.ones_like(u)), axis=-1).astype(np.float32)
    uvd *= np.expand_dims(depth, axis=-1)
    uvd = uvd[depth_mask]
    xyz = np.linalg.solve(cam_K, uvd.T).T
    return xyz

def get_obs():
    # library = TaskGraspScanLibrary("../semantic-grasping-datagen/data/taskgrasp/taskgrasp/scans")
    library = TaskGraspScanLibrary("../semantic-grasping-datasets/taskgrasp_image")
    data = library.get("003_pan", 0)
    image: Image.Image = data["rgb"]
    depth: np.ndarray = data["depth"]
    cam_K: np.ndarray = data["cam_params"]
    grasps: np.ndarray = data["registered_grasps"]
    return image, depth, cam_K, grasps

def main():
    args = get_args()

    if args.zero_shot:
        token = os.getenv("MOLMO_TOKEN")
        molmo = ZeroShotMolmoModal(token)
    else:
        molmo = GraspMolmoBeaker("http://neptune-cs-aus-256.reviz.ai2.in:8080/api/predict_point")

    image, depth, cam_K, grasps = get_obs()
    pc: np.ndarray = depth_to_pc(depth, cam_K)

    task = "Put the pan on the stove."

    grasp_idx = molmo.pred_grasp([image], [pc], [task], [grasps], [cam_K], verbosity=3)[0]

    print(f"Predicted grasp index: {grasp_idx}")
    image.save("pred.png")

if __name__ == "__main__":
    main()
