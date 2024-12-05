import open3d as o3d
import numpy as np
import os
from PIL import Image

SCANS_DIR = "data/taskgrasp/scans"
OBJ_NAME = "151_pitcher"
SCAN_ID = "1"

CAM_POSE = np.eye(4)
CAM_POSE[:3, :3] = np.array([
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0]
])

def img_to_pc(rgb: np.ndarray, depth: np.ndarray, cam_info: np.ndarray):
    h, w = rgb.shape[:2]
    u, v = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    mask = (depth > 0)# & (depth < 1.25)
    uvd = np.stack((u, v, np.ones_like(u)), axis=-1).astype(np.float32)
    uvd *= np.expand_dims(depth, axis=-1)
    uvd = uvd[mask]
    xyz = np.linalg.solve(cam_info, uvd.T).T
    return np.concatenate([xyz, rgb[mask]], axis=-1)

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


rgb = np.asarray(Image.open(f"{SCANS_DIR}/{OBJ_NAME}/{SCAN_ID}_color.png"))
depth = np.load(f"{SCANS_DIR}/{OBJ_NAME}/{SCAN_ID}_depth.npy") / 1000.0
depth[depth>2] = 0
cam_info = np.load(f"{SCANS_DIR}/{OBJ_NAME}/{SCAN_ID}_camerainfo.npy")
grasps = np.load(f"{SCANS_DIR}/{OBJ_NAME}/{SCAN_ID}_grasp.npy")

pc = img_to_pc(rgb, depth, cam_info)
pc[:, 0] += 0.021
depth = pc_to_img(pc[:,:3], cam_info, rgb.shape[0], rgb.shape[1])

# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(121)
# ax.imshow(rgb)
# ax.imshow(depth, cmap="viridis", alpha=0.5)
# ax.set_title('RGB1')
# ax = fig.add_subplot(122)
# ax.imshow(depth, cmap="viridis")
# ax.set_title('Depth')
# fig.show()
# plt.show()

pc = img_to_pc(rgb, depth, cam_info)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc[..., :3])# @ CAM_POSE[:3,:3].T)
pcd.colors = o3d.utility.Vector3dVector(pc[..., 3:6] / 255)

GRIPPER_POINTS = np.array([
        [-0.10, 0, 0, 1],
        [-0.03, 0, 0, 1],
        [-0.03, 0.07, 0, 1],
        [0.03, 0.07, 0, 1],
        [-0.03, 0.07, 0, 1],
        [-0.03, -0.07, 0, 1],
        [0.03, -0.07, 0, 1]])

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
    gripper_points = GRIPPER_POINTS @ grasp_pose.T
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

# good_id = np.random.choice(good_grasp_ids)
# bad_id = np.random.choice(bad_grasp_ids)

# print("Good grasp id:", good_id)
# print("Bad grasp id:", bad_id)

# grasp_geoms = [
#     *create_grasp(grasps[good_id], [1, 0, 0]),
#     *create_grasp(grasps[bad_id], [0, 1, 0])
# ]

from itertools import chain
idx = np.random.choice(len(grasps), 20)
grasps = grasps[idx]
grasp_geoms = list(chain.from_iterable(create_grasp(g, [0, 1, 0]) for g in grasps))

axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd, axes] + grasp_geoms)

