import open3d as o3d
import numpy as np

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

def img_to_pc(rgb: np.ndarray, depth: np.ndarray, cam_info: np.ndarray):
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

def render_offscreen(geometries, width, height, cam_info):
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background(np.array([0, 0, 0, 1]))
    renderer.scene.view.set_post_processing(False)
    for i, geom in enumerate(geometries):
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        renderer.scene.add_geometry(f"geom{i}", geom, material)
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, cam_info)
    renderer.setup_camera(intrinsics, np.eye(4))
    img = renderer.render_to_image()
    return np.asarray(img).astype(np.uint8)

def render_grasps(rgb: np.ndarray, depth: np.ndarray, cam_info: np.ndarray, grasps: np.ndarray, colors: np.ndarray):
    pc = img_to_pc(rgb, depth, cam_info)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[..., :3])
    pcd.colors = o3d.utility.Vector3dVector(pc[..., 3:6] / 255)

    grasps = np.asarray(grasps)
    colors = np.asarray(colors)
    if np.issubdtype(colors.dtype, np.integer):
        colors = colors / 255

    geoms = [pcd]
    for grasp, color in zip(grasps, colors):
        geoms.extend(create_grasp(grasp, color))

    rendered = render_offscreen(geoms, rgb.shape[1], rgb.shape[0], cam_info)
    mask = np.all(rendered == 0, axis=-1)
    rendered[mask] = rgb[mask]
    return rendered
