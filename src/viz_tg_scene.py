import open3d as o3d
import numpy as np
from PIL import Image
import argparse

from grasp_renderer import render_grasps, img_to_pc, create_grasp

SCANS_DIR = "data/taskgrasp/scans"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("obj_name")
    parser.add_argument("scan_id", type=int)
    parser.add_argument("-f", "--save-to-file", action="store_true")
    return parser.parse_args()

def pc_to_img(xyz: np.ndarray, cam_info: np.ndarray, height, width):
    uvd = xyz @ cam_info.T
    uvd /= uvd[:, 2:]
    uv = uvd[:, :2].astype(np.int32)

    img = np.zeros((height, width), dtype=np.float32)
    mask = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    uv = uv[mask]
    xyz = xyz[mask]
    img[uv[:, 1], uv[:, 0]] = xyz[:, 2]
    return img

def main():
    args = get_args()

    rgb = np.asarray(Image.open(f"{SCANS_DIR}/{args.obj_name}/{args.scan_id}_color.png"))
    depth = np.load(f"{SCANS_DIR}/{args.obj_name}/{args.scan_id}_depth.npy") / 1000.0
    depth[depth>2] = 0
    cam_info = np.load(f"{SCANS_DIR}/{args.obj_name}/{args.scan_id}_camerainfo.npy")
    grasps = np.load(f"{SCANS_DIR}/{args.obj_name}/{args.scan_id}_grasp.npy")

    pc = img_to_pc(rgb, depth, cam_info)
    pc[:, 0] += 0.021
    depth = pc_to_img(pc[:,:3], cam_info, rgb.shape[0], rgb.shape[1])

    if args.save_to_file:
        grasp_ids = np.random.choice(len(grasps), 2, replace=False)
        image = render_grasps(rgb, depth, cam_info, grasps[grasp_ids], np.array([[255, 0, 0], [0, 255, 0]]))
        Image.fromarray(image).save(f"{args.obj_name}_{args.scan_id}.png")
    else:
        pc = img_to_pc(rgb, depth, cam_info)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc[..., :3])
        pcd.colors = o3d.utility.Vector3dVector(pc[..., 3:6] / 255)

        from itertools import chain
        idx = np.random.choice(len(grasps), 20)
        grasps = grasps[idx]
        grasp_geoms = list(chain.from_iterable(create_grasp(g, [0, 1, 0]) for g in grasps))

        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, axes] + grasp_geoms)

if __name__ == "__main__":
    main()
