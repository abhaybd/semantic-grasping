import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from pathlib import Path
import pandas as pd
import torchvision.transforms as T
import numpy as np
from scipy.spatial.transform import Rotation as R

DATASET_PATH = "/net/nfs2.prior/abhayd/acronym/data/dataset.csv"
DATA_DIR = "/net/nfs2.prior/abhayd/acronym/data/procgen/observations"

class ImageAugmentation:
    def __init__(self, 
                 img_size=224,
                 color_jitter_prob=0.8,
                 gray_scale_prob=0.2,
                 horizontal_flip_prob=0.5,
                 flip_grasp_prob=0.5):
        
        self.img_size = img_size
        self.color_jitter_prob = color_jitter_prob
        self.gray_scale_prob = gray_scale_prob
        self.horizontal_flip_prob = horizontal_flip_prob
        self.flip_grasp_prob = flip_grasp_prob

        self.flip_grasp_trf = torch.eye(4)
        self.flip_grasp_trf[[0, 1], [0, 1]] = -1

        # Color augmentation
        self.color_jitter = T.ColorJitter(0.4, 0.4, 0.4, 0.1)
        self.gray_scale = T.Grayscale(num_output_channels=3)
        
        # For normalizing RGB images
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __call__(self, rgb, xyz, grasp_pose):
        # Convert to torch tensor and normalize to [0, 1]
        rgb = torch.from_numpy(rgb).float() / 255.0  # (H, W, 3)
        xyz = torch.from_numpy(xyz).float()  # (H, W, 3)
        grasp_pose = torch.from_numpy(grasp_pose).float()  # 4x4 transform matrix
        
        # Random crop (maintain correspondence between rgb and xyz)
        if self.img_size is not None:
            # TODO: ensure that the grasp is in the crop
            raise NotImplementedError
            i, j, h, w = T.RandomCrop.get_params(
                rgb, output_size=(self.img_size, self.img_size))
            rgb = rgb[i:i+h, j:j+w]
            xyz = xyz[i:i+h, j:j+w]
        
        # Random horizontal flip
        if torch.rand(1) < self.horizontal_flip_prob:
            rgb = torch.flip(rgb, dims=[-2])  # Flip along width
            xyz = torch.flip(xyz, dims=[-2])
            xyz[..., 0] = -xyz[..., 0]  # Flip x-coordinates
            # flip x position of grasp and reflect approach vector
            # TODO: verify this is correct
            grasp_pose[0, 3] = -grasp_pose[0, 3]
            new_z = grasp_pose[:3, 2]
            new_z[0] = -new_z[0]
            rot = R.align_vectors(new_z, grasp_pose[:3, 2])[0].as_matrix()
            grasp_pose[:3, :3] = rot @ grasp_pose[:3, :3] @ self.flip_grasp_trf[:3, :3]

        # Color augmentation (only for RGB)
        if torch.rand(1) < self.color_jitter_prob:
            rgb = self.color_jitter(rgb)
        if torch.rand(1) < self.gray_scale_prob:
            rgb = self.gray_scale(rgb)

        # Random flip grasp pose
        if torch.rand(1) < self.flip_grasp_prob:
            grasp_pose = grasp_pose @ self.flip_grasp_trf
        
        # Normalize RGB
        rgb = rgb.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
        rgb = self.normalize(rgb)
        
        # Keep xyz in (H, W, 3) format
        return rgb, xyz, grasp_pose

class ObservationDataset(Dataset):
    def __init__(self, csv_path, data_dir, augment=True):
        self.data_dir = Path(data_dir)
        self.data_df = pd.read_csv(csv_path)
        self.augment = augment
        self.transform = ImageAugmentation() if augment else None
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        
        # Load the pickle file from the observation path
        pkl_path = self.data_dir / row['observation_path']
        with open(pkl_path, 'rb') as f:
            observation_data = pickle.load(f)
        
        rgb = observation_data['rgb']  # (H, W, 3)
        xyz = observation_data['xyz']  # (H, W, 3)
        grasp_pose = observation_data['grasp_pose']  # 4x4 transform matrix
        
        # Apply augmentation if enabled
        if self.augment and self.transform is not None:
            rgb, xyz, grasp_pose = self.transform(rgb, xyz, grasp_pose)
            
        return {
            'annotation_id': row['annotation_id'],
            'text': row['text'],
            'rgb': rgb,
            'xyz': xyz,
            'grasp_pose': grasp_pose
        }

def get_dataloader(batch_size=32, num_workers=4, shuffle=True, augment=True):
    dataset = ObservationDataset(DATASET_PATH, DATA_DIR, augment=augment)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )
    return dataloader



