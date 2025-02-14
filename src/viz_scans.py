import numpy as np
from PIL import Image
import argparse
from grasp_renderer import render_grasps, create_grasp, img_to_pc, create_grasp_mesh

import open3d as o3d

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("obj_name")
    parser.add_argument("-d", "--data-dir", default="data/real_scans")
    parser.add_argument("--conf_thresh", type=float, default=0.0)
    parser.add_argument("-i", "--interactive", action="store_true")
    return parser.parse_args()

def main():
    args = get_args()
    obj_name = args.obj_name
    data_dir = args.data_dir

    cam_info = np.load(f"{data_dir}/{obj_name}/cam_info.npy")
    depth = np.load(f"{data_dir}/{obj_name}/depth.npy")
    rgb = np.array(Image.open(f"{data_dir}/{obj_name}/rgb.png"))
    grasps = np.load(f"{data_dir}/{obj_name}/grasps.npy")
    grasp_confs = np.load(f"{data_dir}/{obj_name}/grasp_confs.npy")
    grasps = grasps[grasp_confs > args.conf_thresh]

    # TODO: filter for grasps on target object

    if args.interactive:
        pc = img_to_pc(rgb, depth, cam_info)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc[..., :3])
        pcd.colors = o3d.utility.Vector3dVector(pc[..., 3:6] / 255)
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        geoms = [pcd, axes]

        colors = [[0, 1, 0]] * len(grasps)
        if "taskgrasp" in data_dir:
            from taskgrasp_utils import TaskGraspInfo
            tg_info = TaskGraspInfo("data/taskgrasp")
            colors = []
            for i in range(len(grasps)):
                colors.append([0, 1, 0] if tg_info.get_grasp_classification(obj_name.split("-")[0], i) == 1 else [1, 0, 0])
        else:
            grasps = [grasps[np.random.choice(len(grasps))]]
            colors = [[0, 1, 0]]

        for grasp, color in zip(grasps, colors):
            geoms.extend(create_grasp(grasp, color))
        # geoms.extend(create_grasp(grasps[0], [0, 1, 0]))
        # geoms.extend(create_grasp_mesh(grasps[21]))
        
        o3d.visualization.draw_geometries(geoms, front=[0, 0, -1], lookat=[0, 0, 1], up=[0, -1, 0], zoom=0.25)
    else:
        grasp_ids = np.random.choice(len(grasps), 2, replace=False)
        image = render_grasps(rgb, depth, cam_info, grasps[grasp_ids], np.array([[255, 0, 0], [0, 255, 0]]))
        Image.fromarray(image).save(f"{obj_name}.png")

if __name__ == "__main__":
    main()
