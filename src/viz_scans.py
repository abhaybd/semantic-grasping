import open3d as o3d
import numpy as np
import os
from PIL import Image
import argparse

SCANS_DIR = "data/real_scans"

GRIPPER_POINTS = np.array([
        [-0.10, 0, 0, 1],
        [-0.03, 0, 0, 1],
        [-0.03, 0.07, 0, 1],
        [0.03, 0.07, 0, 1],
        [-0.03, 0.07, 0, 1],
        [-0.03, -0.07, 0, 1],
        [0.03, -0.07, 0, 1]])

GRIPPER_POINTS[:, 0] += 0.08

CAM_POSE = np.eye(4)
CAM_POSE[:3, :3] = np.array([
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0]
])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("obj_name")
    parser.add_argument("--conf_thresh", type=float, default=0.0)
    return parser.parse_args()

def cam_to_pc(rgb: np.ndarray, depth: np.ndarray, cam_info: np.ndarray):
    h, w = rgb.shape[:2]
    u, v = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    mask = (depth > 0)# & (depth < 1.25)
    uvd = np.stack((u, v, np.ones_like(u)), axis=-1).astype(np.float32)
    uvd *= np.expand_dims(depth, axis=-1)
    uvd = uvd[mask]
    xyz = np.linalg.solve(cam_info, uvd.T).T
    return np.concatenate([xyz, rgb[mask]], axis=-1)

def look_at(p1: np.ndarray, p2: np.ndarray):
    z = p2 - p1
    z /= np.linalg.norm(z)
    x = np.array([1, 0, 0])
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    rot = np.eye(4)
    rot[:3, 0] = x
    rot[:3, 1] = y
    rot[:3, 2] = z
    rot[:3, 3] = p1
    return rot

def create_grasp(grasp_pose: np.ndarray, color=None):
    gripper_points = GRIPPER_POINTS @ np.linalg.inv(CAM_POSE).T @ grasp_pose.T
    geoms = []
    for i in range(len(gripper_points) - 1):
        p1 = gripper_points[i,:3]
        p2 = gripper_points[i + 1,:3]
        height = np.linalg.norm(p2 - p1)
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.003, height=height)
        cylinder.translate([0, 0, height / 2])
        cylinder.transform(look_at(p1, p2))
        if color is not None:
            cylinder.paint_uniform_color(color)
        geoms.append(cylinder)
    return geoms

def render_offline(geometries, width, height, cam_info):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    for geom in geometries:
        vis.add_geometry(geom)
    ctr = vis.get_view_control()
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])
    params = o3d.camera.PinholeCameraParameters()
    params.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, cam_info)
    params.extrinsic = np.eye(4)
    ctr.convert_from_pinhole_camera_parameters(params, True)
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()
    return (np.asarray(image) * 255).astype(np.uint8)

def main():
    args = get_args()
    obj_name = args.obj_name

    cam_info = np.load(f"{SCANS_DIR}/{obj_name}/cam_info.npy")
    depth = np.load(f"{SCANS_DIR}/{obj_name}/depth.npy")
    rgb = np.array(Image.open(f"{SCANS_DIR}/{obj_name}/rgb.png"))
    grasps = np.load(f"{SCANS_DIR}/{obj_name}/grasps.npy")
    grasp_confs = np.load(f"{SCANS_DIR}/{obj_name}/grasp_confs.npy")
    pc = cam_to_pc(rgb, depth, cam_info)

    grasps = grasps[grasp_confs > args.conf_thresh]

    # TODO: filter for grasps on target object

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[..., :3])
    pcd.colors = o3d.utility.Vector3dVector(pc[..., 3:6] / 255)


    good_id = np.random.choice(len(grasps))
    bad_id = np.random.choice(len(grasps))

    grasp_geoms = [
        *create_grasp(grasps[good_id], [1, 0, 0]),
        *create_grasp(grasps[bad_id], [0, 1, 0])
    ]

    # from itertools import chain
    # grasp_geoms = list(chain.from_iterable(create_grasp(g, [0, 1, 0]) for g in grasps))

    height, width = rgb.shape[:2]
    image = render_offline([pcd] + grasp_geoms, width, height, cam_info)

    mask = np.all(image == 0, axis=-1)
    image[mask] = rgb[mask]

    # Save the rendered image
    Image.fromarray(image).save(f"{obj_name}.png")

if __name__ == "__main__":
    main()
