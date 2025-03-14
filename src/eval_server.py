import io
from itertools import compress
import uuid

import numpy as np
from fastapi import FastAPI, Response
import pyrender
from torch import nn
from PIL import Image
from semantic_grasping_datagen.annotation import Annotation, Object, GraspLabel
from semantic_grasping_datagen.datagen.datagen_utils import MeshLibrary, rejection_sample, not_none
from semantic_grasping_datagen.datagen.datagen import DatagenConfig, noncolliding_annotations, sample_scene
from semantic_grasping_datagen.grasp_desc_encoder import GraspDescriptionEncoder

import scene_synthesizer as ss
from scene_synthesizer import Scene
import torch

from model import GraspEncoder

SUPPORT_LIBRARY = MeshLibrary.from_categories("data", ["Table"], {"scale": 0.025})
OBJECT_LIBRARY = MeshLibrary.from_categories("data", ["Mug", "Pan", "WineGlass"])
DATAGEN_CFG = DatagenConfig(n_views=0, n_objects_range=(5, 8), n_background_range=(5, 8), min_annots_per_view=20)
N_GRASPS = 100

scenes: dict[str, Scene] = {}
scene_grasps: dict[str, np.ndarray] = {}
scene_preds: dict[str, np.ndarray] = {}

grasp_encoder = nn.DataParallel(GraspEncoder.from_wandb("01JP75M35K9P61ESAQT9YVG3Y6", map_location="cuda"))
grasp_rgb_processor = grasp_encoder.module.create_rgb_processor()
text_encoder = GraspDescriptionEncoder("cuda", full_precision=False)

renderer = pyrender.OffscreenRenderer(640, 480)


def get_grasps_in_scene(scene: Scene, object_library: MeshLibrary):
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

def set_camera(scene: pyrender.Scene, cam_params: np.ndarray, cam_pose: np.ndarray):
    fx, fy, cx, cy = cam_params
    cam = pyrender.camera.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, name="camera")
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

def render(scene: Scene, cam_pose: np.ndarray, cam_params: np.ndarray):
    fx, fy, cx, cy = cam_params
    cam_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    set_camera(scene, cam_K, cam_pose)
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
    xyz = backproject(cam_K, depth).astype(np.float32)
    return Image.fromarray(color), xyz

app = FastAPI()

@app.post("/api/generate-scene")
async def generate_scene():
    scene, _, _, _ = rejection_sample(
            lambda: sample_scene(DATAGEN_CFG, [], OBJECT_LIBRARY, OBJECT_LIBRARY, SUPPORT_LIBRARY),
            not_none,
            -1
        )
    scene_id = uuid.uuid4().hex
    _, grasps = get_grasps_in_scene(scene, OBJECT_LIBRARY)
    scenes[scene_id] = scene
    if len(grasps) > N_GRASPS:
        grasp_idxs = np.random.choice(len(grasps), N_GRASPS, replace=False)
        grasps = grasps[grasp_idxs]
    scene_grasps[scene_id] = grasps

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

@app.get("/api/get-scene/{scene_id}/{viz_grasps}", responses={200: {"content": {"model/gltf-binary": {}}}}, response_class=Response)
async def get_scene(scene_id: str, viz_grasps: bool):
    scene = scenes[scene_id].copy()

    scene.remove_object("floor")
    scene.remove_object(".+_wall")

    # if viz_grasps, add grasps to scene, colored by matching if pred is present
    # if not, just return the scene
    glb_bytes = io.BytesIO()
    scene.export(glb_bytes, file_type="glb")
    return Response(content=glb_bytes.getvalue(), media_type="model/gltf-binary")

@app.post("/api/predict/{scene_id}")
async def predict(scene_id: str, body: dict):
    query = body["query"]
    query_embedding = text_encoder.encode([query])[0]

    cam_pose = np.array(body["cam_pose"]).reshape(4, 4)
    cam_params = np.array(body["cam_params"])

    scene = scenes[scene_id]
    grasps = torch.from_numpy(scene_grasps[scene_id]).float()
    batch_size = 128

    image, xyz = render(scene, cam_pose, cam_params)
    xyz = torch.from_numpy(xyz).float().permute(2, 0, 1)
    rgb: torch.Tensor = grasp_rgb_processor(image)

    grasp_embeddings = []
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(grasps), batch_size):
                grasps_batch = grasps[i:i+batch_size]
                rgb_batch = rgb.unsqueeze(0).repeat(len(grasps_batch), 1, 1, 1)
                xyz_batch = xyz.unsqueeze(0).repeat(len(grasps_batch), 1, 1, 1)
                embedding = grasp_encoder(rgb_batch, xyz_batch, grasps_batch)
                grasp_embeddings.append(embedding.cpu().numpy())
    grasp_embeddings = np.concatenate(grasp_embeddings, axis=0)
    similarities = query_embedding @ grasp_embeddings.T
    scene_preds[scene_id] = similarities
