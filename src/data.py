from typing import Callable, Optional
import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from torchvision.transforms.v2 import functional as F

import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation as R

DATASET_PATH = "/net/nfs2.prior/abhayd/acronym/data/dataset.csv"
DATA_DIR = "/net/nfs2.prior/abhayd/acronym/data/procgen/observations"


class ImageAugmentation:
    def __init__(self,
                 color_jitter_prob=0.8,
                 gray_scale_prob=0.2,
                 horizontal_flip_prob=0.5,
                 flip_grasp_prob=0.5):

        self.color_jitter_prob = color_jitter_prob
        self.gray_scale_prob = gray_scale_prob
        self.horizontal_flip_prob = horizontal_flip_prob
        self.flip_grasp_prob = flip_grasp_prob

        self.flip_grasp_trf = torch.eye(4)
        self.flip_grasp_trf[[0, 1], [0, 1]] = -1

        # Color augmentation
        self.color_jitter = T.ColorJitter(0.4, 0.4, 0.4, 0.1)
        self.gray_scale = T.Grayscale(num_output_channels=3)

    def __call__(self, rgb, xyz, grasp_pose):
        """
        rgb: PIL Image
        xyz: (3, H, W) torch.Tensor
        grasp_pose: (4, 4) torch.Tensor
        """

        # Random horizontal flip
        if torch.rand(1) < self.horizontal_flip_prob:
            rgb = F.horizontal_flip(rgb)
            xyz = torch.flip(xyz, dims=[-1])
            xyz[0] = -xyz[0]  # Flip x-coordinates
            # flip x position of grasp and reflect approach vector
            # TODO: verify this is correct
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

        # Random flip grasp pose
        if torch.rand(1) < self.flip_grasp_prob:
            grasp_pose = grasp_pose @ self.flip_grasp_trf

        return rgb, xyz, grasp_pose


class GraspDescriptionDataset(Dataset):
    def __init__(self, csv_path, data_dir, img_processor: Optional[Callable[[Image.Image], torch.Tensor]] = None, augment=True):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path)
        self.transform = ImageAugmentation() if augment else None
        if img_processor is not None:
            self.img_processor = img_processor
        else:
            self.img_processor = T.Compose([
                T.PILToTensor(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.data_df)

    def _load_obs(self, pkl_path: str):
        with open(pkl_path, 'rb') as f:
            observation_data = pickle.load(f)

        rgb = observation_data['rgb']  # (H, W, 3)
        xyz = observation_data['xyz']  # (H, W, 3)
        grasp_pose = observation_data['grasp_pose']  # 4x4 transform matrix
        return rgb, xyz, grasp_pose

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]

        pkl_path = os.path.join(self.data_dir, row['observation_path'])
        rgb, xyz, grasp_pose = self._load_obs(pkl_path)

        rgb = Image.fromarray(rgb)

        xyz: torch.Tensor = torch.from_numpy(xyz).float()  # (H, W, 3)
        xyz = xyz.permute(2, 0, 1)  # (3, H, W)
        grasp_pose: torch.Tensor = torch.from_numpy(grasp_pose).float()  # 4x4 transform matrix

        if self.transform is not None:
            rgb, xyz, grasp_pose = self.transform(rgb, xyz, grasp_pose)

        rgb = self.img_processor(rgb)

        return {
            'annotation_id': row['annotation_id'],
            'text': row['text'],
            'rgb': rgb,
            'xyz': xyz,
            'grasp_pose': grasp_pose
        }


def get_dataloader(batch_size=32, num_workers=4, shuffle=True, augment=True):
    dataset = GraspDescriptionDataset(DATASET_PATH, DATA_DIR, augment=augment)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )
    return dataloader


if __name__ == "__main__":
    dataset = GraspDescriptionDataset(DATASET_PATH, DATA_DIR, augment=True)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True)
    batch = next(iter(dataloader))
    print(batch)
