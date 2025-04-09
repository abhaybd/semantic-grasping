from typing import Callable, Optional, Any
import os

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms.v2 as T
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2 import functional as trfF

import pandas as pd
import h5py
from PIL import Image
from scipy.spatial.transform import Rotation as R


class ImageAugmentation:
    def __init__(
        self,
        color_jitter_prob=0.8,
        gray_scale_prob=0.2,
        horizontal_flip_prob=0.5,
        flip_grasp_prob=0.5,
        depth_mask_prob=0.5,
        depth_mask_scale_range=(0.02, 0.2),
        depth_mask_ratio_range=(0.5, 2.0),
    ):
        self.color_jitter_prob = color_jitter_prob
        self.gray_scale_prob = gray_scale_prob
        self.horizontal_flip_prob = horizontal_flip_prob
        self.flip_grasp_prob = flip_grasp_prob

        self.flip_grasp_trf = torch.eye(4)
        self.flip_grasp_trf[[0, 1], [0, 1]] = -1

        # Color augmentation
        self.color_jitter = T.ColorJitter(0.4, 0.4, 0.4, 0.1)
        self.gray_scale = T.Grayscale(num_output_channels=3)

        self.depth_erasing = T.RandomErasing(p=depth_mask_prob, scale=depth_mask_scale_range, ratio=depth_mask_ratio_range)

    def __call__(self, rgb: Image.Image, xyz: torch.Tensor, grasp_pose: torch.Tensor):
        """
        rgb: PIL Image
        xyz: (3, H, W) torch.Tensor
        grasp_pose: (4, 4) torch.Tensor
        """

        # Random horizontal flip
        if torch.rand(1) < self.horizontal_flip_prob:
            rgb = trfF.horizontal_flip(rgb)
            xyz = torch.flip(xyz, dims=[-1])
            xyz[0] = -xyz[0]  # Flip x-coordinates
            # flip x position of grasp and reflect approach vector
            grasp_pose[0, 3] = -grasp_pose[0, 3]
            new_z = grasp_pose[:3, 2]
            new_z[0] = -new_z[0]
            rot = torch.from_numpy(R.align_vectors(new_z, grasp_pose[:3, 2])[0].as_matrix()).float()
            grasp_pose[:3, :3] = rot @ grasp_pose[:3, :3] @ self.flip_grasp_trf[:3, :3]

        # Color augmentation (only for RGB)
        if torch.rand(1) < self.color_jitter_prob:
            rgb = self.color_jitter(rgb)
        if torch.rand(1) < self.gray_scale_prob:
            rgb = self.gray_scale(rgb)

        xyz = self.depth_erasing(xyz)

        # Random flip grasp pose
        if torch.rand(1) < self.flip_grasp_prob:
            grasp_pose = grasp_pose @ self.flip_grasp_trf

        return rgb, xyz, grasp_pose

def load_obs(data_dir: str, row) -> dict[str, Any]:
    with h5py.File(os.path.join(data_dir, row["scene_path"]), "r") as f:
        return {
            'rgb': Image.fromarray(f[row["rgb_key"]][:]).convert("RGB"),
            'xyz': f[row["xyz_key"]][:],
            'normals': f[row["normals_key"]][:],
            'grasp_pose': f[row["grasp_pose_key"]][:],
        }

class GraspDescriptionRegressionDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        data_dir: str,
        text_embedding_path: str,
        img_processor: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        augment: bool = True,
        augmentation_params: Optional[dict] = None,
        xyz_far_clip: float = 1.0,
    ):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path)
        self.text_embeddings = np.load(text_embedding_path)
        self.xyz_far_clip = xyz_far_clip
        aug_params = augmentation_params or {}
        self.transform = ImageAugmentation(**aug_params) if augment else None
        if img_processor is not None:
            self.img_processor = img_processor
        else:
            self.img_processor = T.Compose([
                T.PILToTensor(),
                T.ToDtype(torch.float32, scale=True),
                T.Resize((512, 512)),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]

        obs = load_obs(self.data_dir, row)
        rgb: Image.Image = obs['rgb']
        xyz: np.ndarray = obs['xyz']
        grasp_pose: np.ndarray = obs['grasp_pose']

        xyz: torch.Tensor = torch.from_numpy(xyz).float()  # (H, W, 3)
        xyz = xyz.permute(2, 0, 1)  # (3, H, W)
        grasp_pose: torch.Tensor = torch.from_numpy(grasp_pose).float()  # 4x4 transform matrix

        far_clip_mask = xyz[2] >= self.xyz_far_clip
        xyz[:, far_clip_mask] = 0.0

        if self.transform is not None:
            rgb, xyz, grasp_pose = self.transform(rgb, xyz, grasp_pose)

        rgb = self.img_processor(rgb)

        if rgb.shape[-2:] != xyz.shape[-2:]:
            xyz = trfF.resize(xyz, rgb.shape[-2:], interpolation=InterpolationMode.NEAREST_EXACT)

        return {
            'annotation_id': row['annotation_id'],
            'text_embedding': self.text_embeddings[idx],
            'rgb': rgb,
            'xyz': xyz,
            'grasp_pose': grasp_pose
        }

class GraspDescriptionClassificationDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        data_dir: str,
        img_processor: Callable[[Image.Image], torch.Tensor],
        text_processor: Callable[[str], tuple[torch.Tensor, torch.Tensor]],
        text_embedding_path: Optional[str] = None,
        use_frozen_text_embeddings: bool = False,
        augment: bool = True,
        augmentation_params: Optional[dict] = None,
        xyz_far_clip: float = 1.0,
    ):
        self.data_df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.xyz_far_clip = xyz_far_clip
        self.text_embeddings = np.load(text_embedding_path) if text_embedding_path and use_frozen_text_embeddings else None
        aug_params = augmentation_params or {}
        self.transform = ImageAugmentation(**aug_params) if augment else None
        self.img_processor = img_processor
        self.text_processor = text_processor

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]

        obs = load_obs(self.data_dir, row)
        rgb: Image.Image = obs['rgb']
        xyz: np.ndarray = obs['xyz']
        grasp_pose: np.ndarray = obs['grasp_pose']

        xyz: torch.Tensor = torch.from_numpy(xyz).float()  # (H, W, 3)
        xyz = xyz.permute(2, 0, 1)  # (3, H, W)
        grasp_pose: torch.Tensor = torch.from_numpy(grasp_pose).float()  # 4x4 transform matrix

        far_clip_mask = xyz[2] >= self.xyz_far_clip
        xyz[:, far_clip_mask] = 0.0

        if self.text_embeddings is not None:
            text_embedding = self.text_embeddings[idx]
            text_inputs = {
                "text_embedding": text_embedding,
            }
        else:
            annotation = row["annot"]
            text_input_ids, text_attention_mask = self.text_processor(annotation)
            text_inputs = {
                "input_ids": text_input_ids,
                "attention_mask": text_attention_mask,
            }

        if self.transform is not None:
            rgb, xyz, grasp_pose = self.transform(rgb, xyz, grasp_pose)

        rgb = self.img_processor(rgb)

        if rgb.shape[-2:] != xyz.shape[-2:]:
            xyz = trfF.resize(xyz, rgb.shape[-2:], interpolation=InterpolationMode.NEAREST_EXACT)

        return {
            'annotation_id': row['annotation_id'],
            'rgb': rgb,
            'xyz': xyz,
            'grasp_pose': grasp_pose,
            'text_inputs': text_inputs
        }
