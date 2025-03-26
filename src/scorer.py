from abc import ABC, abstractmethod

from PIL import Image
import numpy as np
import torch
import math
from torchvision.transforms.v2.functional import resize

from semantic_grasping_datagen.grasp_desc_encoder import GraspDescriptionEncoder
from model import GraspEncoder, GraspClassifier

class GraspScorer(ABC):
    @abstractmethod
    def score_grasps(self, cam_pose: np.ndarray, rgb: Image.Image, xyz: np.ndarray, grasps: np.ndarray, query: str) -> np.ndarray:
        """
        cam_pose: numpy.ndarray, shape (4, 4)
        rgb: PIL.Image
        xyz: numpy.ndarray, shape (H, W, 3)
        grasps: numpy.ndarray, shape (N, 4, 4) in camera frame
        query: str
        returns: numpy.ndarray, shape (N,) of scores in [-1, 1]
        """
        raise NotImplementedError

class GraspRegressionScorer(GraspScorer):
    def __init__(self, grasp_encoder: GraspEncoder, query_encoder: GraspDescriptionEncoder, device: torch.device, query_device: torch.device):
        self.grasp_encoder = grasp_encoder
        self.grasp_encoder.to(device)
        self.query_encoder = query_encoder
        self.query_encoder.to(query_device)
        self.device = device
        self.query_device = query_device

        self.rgb_processor = self.grasp_encoder.create_rgb_processor()

    def to(self, device: torch.device):
        self.device = device
        self.grasp_encoder.to(device)

    def score_grasps(self, cam_pose: np.ndarray, rgb: Image.Image, xyz: np.ndarray, grasps: np.ndarray, query: str) -> np.ndarray:
        rgb = self.rgb_processor(rgb).to(self.device)
        xyz = torch.from_numpy(xyz).float().permute(2, 0, 1).to(self.device)

        if rgb.shape[-2:] != xyz.shape[-2:]:
            xyz = resize(xyz, size=rgb.shape[-2:])

        rgb = rgb.unsqueeze(0)
        xyz = xyz.unsqueeze(0)

        trf = np.eye(4)
        trf[[1,2]] = -trf[[1,2]]
        grasps = trf[None] @ grasps
        grasps = torch.from_numpy(grasps).float().to(self.device)

        query_embedding = self.query_encoder.encode([query], is_query=False)[0]
        grasp_embeddings = []
        batch_size = 128
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(0, len(grasps), batch_size):
                    print(f"Processing batch {i//batch_size + 1} of {math.ceil(len(grasps)/batch_size)}")
                    grasps_batch = grasps[i:i+batch_size]
                    embedding = self.grasp_encoder(rgb, xyz, grasps_batch)
                    grasp_embeddings.append(embedding.cpu().numpy())
        grasp_embeddings = np.concatenate(grasp_embeddings, axis=0)
        similarities = query_embedding @ grasp_embeddings.T
        return similarities

class GraspClassificationScorer(GraspScorer):
    def __init__(self, classifier: GraspClassifier, device: torch.device):
        self.classifier = classifier
        self.classifier.to(device)
        self.rgb_processor = self.classifier.create_rgb_processor()
        self.text_processor = self.classifier.create_text_processor()
        self.device = device

    def to(self, device: torch.device):
        self.device = device
        self.classifier.to(device)

    def score_grasps(self, cam_pose: np.ndarray, rgb: Image.Image, xyz: np.ndarray, grasps: np.ndarray, query: str) -> np.ndarray:
        rgb = self.rgb_processor(rgb).to(self.device)
        xyz = torch.from_numpy(xyz).float().permute(2, 0, 1).to(self.device)

        if rgb.shape[-2:] != xyz.shape[-2:]:
            xyz = resize(xyz, size=rgb.shape[-2:])

        rgb = rgb.unsqueeze(0)
        xyz = xyz.unsqueeze(0)

        trf = np.eye(4)
        trf[[1,2]] = -trf[[1,2]]
        grasps = trf[None] @ grasps
        grasps = torch.from_numpy(grasps).float().to(self.device)

        query_input_ids, query_attention_mask = self.text_processor([query])
        query_input_ids = query_input_ids.to(self.device)
        query_attention_mask = query_attention_mask.to(self.device)

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                batch_size = 128
                scores = []
                for i in range(0, len(grasps), batch_size):
                    print(f"Processing batch {i//batch_size + 1} of {math.ceil(len(grasps)/batch_size)}")
                    grasps_batch = grasps[i:i+batch_size]
                    logits = self.classifier(rgb, xyz, grasps_batch, query_input_ids, query_attention_mask)
                    scores_batch = torch.nn.functional.sigmoid(logits).flatten() * 2 - 1
                    scores.append(scores_batch.float().cpu().numpy())

        return np.concatenate(scores, axis=0)

def load_scorer(run_id: str, ckpt: int | None = None, map_location: str = "cpu") -> GraspScorer:
    import wandb
    run_path = f"prior-ai2/semantic-grasping/{run_id}"
    api = wandb.Api()
    run = api.run(run_path)
    config = run.config
    
    if "type" not in config:
        config["type"] = "classification" if "text_encoder" in config["model"] else "regression"
        print("Inferring scorer type")

    print("Loading scorer type:", config["type"])
    if config["type"] == "regression":
        grasp_encoder = GraspEncoder.from_wandb(run_id, ckpt, map_location=map_location)
        query_encoder = GraspDescriptionEncoder("cuda:1")
        return GraspRegressionScorer(grasp_encoder, query_encoder, map_location, "cuda:1")
    elif config["type"] == "classification":
        classifier = GraspClassifier.from_wandb(run_id, ckpt, map_location=map_location)
        return GraspClassificationScorer(classifier, map_location)
