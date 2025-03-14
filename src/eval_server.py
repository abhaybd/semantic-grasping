import io
from itertools import compress
import uuid
import math
import os
import re
from scipy.spatial.transform import Rotation as R

if os.environ.get("PYOPENGL_PLATFORM") is None:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
# pyrender spawns a lot of OMP threads, limiting to 1 significantly reduces overhead
if os.environ.get("OMP_NUM_THREADS") is None:
    os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from fastapi import FastAPI, Response
import pyrender
import pyrender.light
import torch
import pickle
from torch.nn import functional as F
from PIL import Image
import scene_synthesizer as ss
from acronym_tools import create_gripper_marker
import matplotlib.cm as cm

from semantic_grasping_datagen.annotation import Annotation, Object, GraspLabel
from semantic_grasping_datagen.datagen.datagen_utils import MeshLibrary, rejection_sample, not_none
from semantic_grasping_datagen.datagen.datagen import DatagenConfig, noncolliding_annotations, sample_scene, generate_lighting
from semantic_grasping_datagen.grasp_desc_encoder import GraspDescriptionEncoder

from model import GraspEncoder

SUPPORT_LIBRARY = MeshLibrary.from_categories("../acronym/data", ["Table"], {"scale": 0.025})
OBJECT_LIBRARY = MeshLibrary.from_categories("../acronym/data", ["Mug", "Pan", "WineGlass"])
DATAGEN_CFG = DatagenConfig(n_views=0, n_objects_range=(5, 8), n_background_range=(5, 8), min_annots_per_view=20)
N_GRASPS = 100

scenes: dict[str, ss.Scene] = {}
lightings: dict[str, list[dict]] = {}
scene_grasps: dict[str, np.ndarray] = {}
scene_preds: dict[str, np.ndarray] = {}

grasp_encoder = GraspEncoder.from_wandb("01JP75M35K9P61ESAQT9YVG3Y6", map_location="cuda").cuda()
grasp_rgb_processor = grasp_encoder.create_rgb_processor()
text_encoder = GraspDescriptionEncoder("cuda:1", full_precision=False)
# grasp_encoder = None
# grasp_rgb_processor = None
# text_encoder = None

print("Done loading models")

renderer = pyrender.OffscreenRenderer(640, 480)


def get_grasps_in_scene(scene: ss.Scene, object_library: MeshLibrary):
    grasps_dict: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}  # maps object in scene to its grasps in centroid frame
    for name in scene.get_object_names():
        assert isinstance(name, str)
        if not name.startswith("object_"):
            continue
        _, cat, obj_id = name.split("_", 2)
        grasps_dict[(cat, obj_id)] = object_library.grasps(cat, obj_id)

    in_scene_annotations: list[Annotation] = []
    annotation_grasps = []  # grasps in scene frame

    for (category, obj_id), (grasps, succs) in grasps_dict.items():
        succ_idxs = np.nonzero(succs)[0]
        for idx in succ_idxs:
            grasp = grasps[idx].copy()
            obj_name = f"object_{category}_{obj_id}"
            geom_names = scene.get_geometry_names(obj_name)
            assert len(geom_names) == 1
            grasp[:3, 3] += scene.get_centroid(geom_names[0], obj_name)
            obj_trf = scene.get_transform(obj_name)
            grasp = obj_trf @ grasp
            annotation_grasps.append(grasp)
            in_scene_annotations.append(Annotation(
                obj=Object(object_category=category, object_id=obj_id),
                grasp_id=idx,
                obj_description="",
                grasp_description="",
                grasp_label=GraspLabel.GOOD,
                is_mesh_malformed=False
            ))
    annotation_grasps = np.array(annotation_grasps)

    noncolliding = noncolliding_annotations(scene, in_scene_annotations, annotation_grasps, {})
    in_scene_annotations = list(compress(in_scene_annotations, noncolliding))
    annotation_grasps = annotation_grasps[noncolliding]

    return in_scene_annotations, annotation_grasps

def set_camera(scene: pyrender.Scene, cam_K: np.ndarray, cam_pose: np.ndarray):
    cam = pyrender.camera.IntrinsicsCamera(fx=cam_K[0, 0], fy=cam_K[1, 1], cx=cam_K[0, 2], cy=cam_K[1, 2], name="camera")
    cam_node = pyrender.Node(name="camera", camera=cam, matrix=cam_pose)
    for n in (scene.get_nodes(name=cam_node.name) or []):
        scene.remove_node(n)
    scene.add_node(cam_node)

    cam_light = pyrender.light.PointLight(intensity=2.0, name="camera_light")
    camera_light_node = pyrender.Node(name="camera_light", matrix=cam_pose, light=cam_light)
    for n in (scene.get_nodes(name=camera_light_node.name) or []):
        scene.remove_node(n)
    scene.add_node(camera_light_node)

def backproject(cam_K: np.ndarray, depth: np.ndarray):
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    uvd = np.stack((u, v, np.ones_like(u)), axis=-1).astype(np.float32)
    uvd *= np.expand_dims(depth, axis=-1)
    xyz = uvd @ np.expand_dims(np.linalg.inv(cam_K).T, axis=0)
    return xyz

def render(scene: pyrender.Scene, cam_pose: np.ndarray, cam_params: np.ndarray):
    vfov, cx, cy = cam_params
    fx = fy = cy / np.tan(np.radians(vfov/2))
    cam_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    set_camera(scene, cam_K, cam_pose)
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
    xyz = backproject(cam_K, depth).astype(np.float32)
    return Image.fromarray(color), xyz

def build_scene(ss_scene: ss.Scene, lighting: list[dict]):
    scene = pyrender.Scene.from_trimesh_scene(ss_scene.scene)
    for light in lighting:
        light_type = getattr(pyrender.light, light["type"])
        light_args = light["args"]
        light_args["color"] = np.array(light["args"]["color"]) / 255.0
        light_node = pyrender.Node(light["args"]["name"], matrix=light["transform"], light=light_type(**light["args"]))
        scene.add_node(light_node)
    return scene

app = FastAPI()

@app.post("/api/generate-scene")
async def generate_scene():
    scene, _, _, _ = rejection_sample(
        lambda: sample_scene(DATAGEN_CFG, [], OBJECT_LIBRARY, OBJECT_LIBRARY, SUPPORT_LIBRARY),
        not_none,
        -1
    )
    lighting = generate_lighting(scene, DATAGEN_CFG)
    scene_id = uuid.uuid4().hex
    _, grasps = get_grasps_in_scene(scene, OBJECT_LIBRARY)
    scenes[scene_id] = scene
    lightings[scene_id] = lighting
    if len(grasps) > N_GRASPS:
        grasp_idxs = np.random.choice(len(grasps), N_GRASPS, replace=False)
        grasps = grasps[grasp_idxs]
    scene_grasps[scene_id] = grasps
    np.save("grasps.npy", grasps)
    scene.export("scene.json")
    with open("lighting.pkl", "wb") as f:
        pickle.dump(lighting, f)
    return scene_id

@app.post("/api/clear-pred/{scene_id}")
async def clear_pred(scene_id: str):
    if scene_id in scene_preds:
        del scene_preds[scene_id]

@app.post("/api/clear-scene/{scene_id}")
async def clear_scene(scene_id: str):
    if scene_id in scenes:
        del scenes[scene_id]
    if scene_id in scene_grasps:
        del scene_grasps[scene_id]
    if scene_id in scene_preds:
        del scene_preds[scene_id]

@app.get("/api/get-scene/{scene_id}/{show_pred}", responses={200: {"content": {"model/gltf-binary": {}}}}, response_class=Response)
async def get_scene(scene_id: str, show_pred: bool):
    scene_id = re.sub(r"[^a-zA-Z0-9]", "", scene_id)
    try:
        scene = scenes[scene_id].copy()
    except KeyError:
        print(f"Scene {scene_id} not found, curr scenes: {scenes.keys()}")
        return Response(content="Scene not found", media_type="text/plain", status_code=404)

    grasps = scene_grasps[scene_id]
    if scene_id in scene_preds and show_pred:
        similarities = scene_preds[scene_id]
        print("Similarity range: ", similarities.min(), similarities.max())
        normalized = (similarities - similarities.min()) / (similarities.max() - similarities.min())
        colors = (cm.viridis(normalized) * 255).astype(np.uint8)[:, :3]
    else:
        colors = [(0, 255, 0)] * len(grasps)
    for grasp, color in zip(grasps, colors):
        gripper_marker = create_gripper_marker(color=color)
        scene.add_object(ss.TrimeshAsset(gripper_marker), transform=grasp)

    # glb_bytes = io.BytesIO()
    # scene.export(glb_bytes, file_type="glb")
    scene.export("scene.glb")
    with open("scene.glb", "rb") as f:
        b = f.read()
    return Response(content=b, media_type="model/gltf-binary")

@app.post("/api/predict/{scene_id}")
async def predict(scene_id: str, body: dict):
    query = body["query"]

    cam_pos = np.array(body["cam_pos"])
    cam_quat = np.array(body["cam_quat"])
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = R.from_quat(cam_quat).as_matrix()
    cam_pose[:3, 3] = cam_pos
    cam_params = np.array(body["cam_params"])

    scene = build_scene(scenes[scene_id], lightings[scene_id])
    grasps = torch.from_numpy(scene_grasps[scene_id]).float().cuda()
    batch_size = 128

    print("Rendering")
    image, xyz = render(scene, cam_pose, cam_params)
    print("Done rendering")
    image.save("image.png")
    xyz = torch.from_numpy(xyz).float().permute(2, 0, 1).cuda()
    rgb: torch.Tensor = grasp_rgb_processor(image).cuda()
    if rgb.shape[-2:] != xyz.shape[-2:]:
        xyz = F.interpolate(xyz.unsqueeze(0), size=rgb.shape[-2:], mode='bilinear').squeeze(0)

    print("Encoding query")
    query_embedding = text_encoder.encode([query])[0]
    print("Done encoding query")
    grasp_embeddings = []
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(grasps), batch_size):
                print(f"Processing batch {i//batch_size + 1} of {math.ceil(len(grasps)/batch_size)}")
                grasps_batch = grasps[i:i+batch_size]
                rgb_batch = rgb.unsqueeze(0).repeat(len(grasps_batch), 1, 1, 1)
                xyz_batch = xyz.unsqueeze(0).repeat(len(grasps_batch), 1, 1, 1)
                embedding = grasp_encoder(rgb_batch, xyz_batch, grasps_batch)
                grasp_embeddings.append(embedding.cpu().numpy())
    grasp_embeddings = np.concatenate(grasp_embeddings, axis=0)
    similarities = query_embedding @ grasp_embeddings.T
    scene_preds[scene_id] = similarities
