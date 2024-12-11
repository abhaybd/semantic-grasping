import argparse
import os
import multiprocessing as mp

import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm

from grasp_renderer import GRIPPER_OFFSET, GeomRenderer

# change-of-basis trf from TaskGrasp to M2T2 grasp frame
TG_TO_M2T2_TRF = np.eye(4)
TG_TO_M2T2_TRF[:3, :3] = np.array([
    [0, -1, 0],
    [0, 0, -1],
    [1, 0, 0]
])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir", type=str, default="data/taskgrasp/scans")
    parser.add_argument("-n", "--n-views", type=int, default=5)
    parser.add_argument("-o", "--output-dir", type=str, default="data/taskgrasp_scenes")
    parser.add_argument("-s", "--synchronous", action="store_true", help="Disable parallel processing")
    return parser.parse_args()

def random_camera_pose():
    r = 1.0
    theta = np.random.rand() * 2 * np.pi
    psi = np.random.rand() * (np.pi/3 - np.pi/12) + np.pi/12
    pos = np.array([r * np.cos(theta) * np.sin(psi), r * np.sin(theta) * np.sin(psi), r * np.cos(psi)])

    z = -pos / np.linalg.norm(pos)
    y = np.array([0, 0, -1])
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    pose = np.eye(4)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = z
    pose[:3, 3] = pos
    return np.linalg.inv(pose)

def create_floor(z_pos, size=10.0):
    plane = o3d.geometry.TriangleMesh.create_box(size, size, 0.01, create_uv_map=True)
    plane.translate([-size/2, -size/2, z_pos - 0.01])
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.albedo_img = o3d.io.read_image("img/wood_texture_4k.jpg")
    return {"name": "floor", "geometry": plane, "material": material}

def render_scenes(renderer: GeomRenderer, data_dir: str, obj_name: str, n_views: int, output_dir: str):
    obj_dir = os.path.join(data_dir, obj_name)
    pc: np.ndarray = np.load(f"{obj_dir}/fused_pc_clean.npy")
    pc[:,:3] -= pc[..., :3].mean(axis=0, keepdims=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(pc[:,3:6] / 255)
    geoms = [pcd, create_floor(np.min(pc[..., 2]) - 0.04)]

    img_h, img_w = renderer.height, renderer.width
    fov_x = np.pi/3
    f_x = img_w / (2 * np.tan(fov_x / 2))
    cam_info = np.array([
        [f_x, 0, img_w / 2],
        [0, f_x, img_h / 2],
        [0, 0, 1]
    ])

    grasps = []
    for grasp_id in sorted(os.listdir(f'{obj_dir}/grasps'), key=int):
        grasp = np.load(f'{obj_dir}/grasps/{grasp_id}/grasp.npy')
        grasps.append(grasp)
    grasps = np.array(grasps)
    grasp_confs = np.ones(len(grasps), dtype=np.float32)

    for i in range(n_views):
        cam_pose = random_camera_pose() # world-to-cam transform
        rgb = renderer.render(geoms, cam_info, cam_pose, depth=False)
        depth = renderer.render(geoms, cam_info, cam_pose, depth=True)
        grasps_trf = np.expand_dims(cam_pose, axis=0) @ grasps @ np.linalg.inv(TG_TO_M2T2_TRF)
        grasps_trf[:, :3, 3] -= GRIPPER_OFFSET * grasps_trf[:, :3, 2]

        obj_save_dir = f"{output_dir}/{obj_name}-view{i}"
        if not os.path.isdir(obj_save_dir):
            os.mkdir(obj_save_dir)
        np.save(f"{obj_save_dir}/cam_info.npy", cam_info)
        Image.fromarray(rgb).save(f"{obj_save_dir}/rgb.png")
        np.save(f"{obj_save_dir}/depth.npy", depth)
        np.save(f"{obj_save_dir}/grasps.npy", grasps_trf)
        np.save(f"{obj_save_dir}/grasp_confs.npy", grasp_confs)

def init_process():
    globals()["renderer"] = GeomRenderer(1280, 720)

def render_async(*args):
    return render_scenes(globals()["renderer"], *args)

def main():
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.synchronous:
        renderer = GeomRenderer(1280, 720)
        for obj_name in tqdm(os.listdir(args.data_dir)):
            render_scenes(renderer, args.data_dir, obj_name, args.n_views, args.output_dir)
    else:
        with mp.Pool(processes=64, initializer=init_process) as p:
            futures = []
            queue = mp.Queue()
            for obj_name in os.listdir(args.data_dir):
                futures.append(
                    p.apply_async(render_async,
                                (args.data_dir, obj_name, args.n_views, args.output_dir),
                                callback=queue.put,
                                error_callback=queue.put))
            for _ in tqdm(range(len(futures))):
                queue.get()
            assert all(future.ready() and future.successful() for future in futures)

if __name__ == "__main__":
    main()
