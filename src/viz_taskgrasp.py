import open3d as o3d
import numpy as np
import os

SCANS_DIR = "data/taskgrasp/scans"
OBJ_NAME = "112_spatula"
SCAN_ID = "0"
TASK_NAME = "stir"

pc = np.load(f"{SCANS_DIR}/{OBJ_NAME}/fused_pc_clean.npy")
pc[:,:3] -= pc[..., :3].mean(axis=0, keepdims=True)

grasps_map = {}
for grasp_id in os.listdir(os.path.join(SCANS_DIR, OBJ_NAME, "grasps")):
    grasp = np.load(os.path.join(SCANS_DIR, OBJ_NAME, "grasps", grasp_id, "grasp.npy"))
    grasps_map[int(grasp_id)] = grasp
grasps = np.zeros((max(grasps_map.keys()) + 1, 4, 4))
for k, v in grasps_map.items():
    grasps[k] = v

good_grasp_ids = []
bad_grasp_ids = []
with open("data/taskgrasp/task2_results.txt", "r") as f:
    for line in f.read().split("\n"):
        if line.startswith(OBJ_NAME):
            k, v = line.rsplit(":", 1)
            _, grasp_id, task = k.split("-")
            if task == TASK_NAME:
                if v == "1":
                    good_grasp_ids.append(int(grasp_id))
                elif v == "-1":
                    bad_grasp_ids.append(int(grasp_id))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc[..., :3])
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

good_id = np.random.choice(good_grasp_ids)
bad_id = np.random.choice(bad_grasp_ids)

print("Good grasp id:", good_id)
print("Bad grasp id:", bad_id)

grasp_geoms = [
    *create_grasp(grasps[good_id], [1, 0, 0]),
    *create_grasp(grasps[bad_id], [0, 1, 0])
]

# from itertools import chain
# grasp_geoms = list(chain.from_iterable(create_grasp(g, [0, 1, 0]) for g in grasps[good_grasp_ids]))
# grasp_geoms += list(chain.from_iterable(create_grasp(g, [1, 0, 0]) for g in grasps[bad_grasp_ids]))

o3d.visualization.draw_geometries([pcd] + grasp_geoms)
