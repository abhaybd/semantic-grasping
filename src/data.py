from typing import Callable, Optional
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms.v2 as T
from torchvision.transforms.v2 import functional as trfF

import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation as R


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
            rgb = trfF.horizontal_flip(rgb)
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


class GraspDescriptionRegressionDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        data_dir: str,
        text_embedding_path: str,
        img_processor: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        augment: bool = True,
        augmentation_params: Optional[dict] = None
    ):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path)
        self.text_embeddings = np.load(text_embedding_path)
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

        rgb = Image.open(os.path.join(self.data_dir, row["rgb_path"])).convert("RGB")
        xyz = np.load(os.path.join(self.data_dir, row["xyz_path"]))
        grasp_pose = np.load(os.path.join(self.data_dir, row["grasp_pose_path"]))

        xyz: torch.Tensor = torch.from_numpy(xyz).float()  # (H, W, 3)
        xyz = xyz.permute(2, 0, 1)  # (3, H, W)
        grasp_pose: torch.Tensor = torch.from_numpy(grasp_pose).float()  # 4x4 transform matrix

        if self.transform is not None:
            rgb, xyz, grasp_pose = self.transform(rgb, xyz, grasp_pose)

        rgb = self.img_processor(rgb)

        if rgb.shape[-2:] != xyz.shape[-2:]:
            xyz = trfF.resize(xyz, rgb.shape[-2:])

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
        augment: bool = True,
        augmentation_params: Optional[dict] = None
    ):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path)
        aug_params = augmentation_params or {}
        self.transform = ImageAugmentation(**aug_params) if augment else None
        self.img_processor = img_processor
        self.text_processor = text_processor

        self.unique_annots = self.data_df["text"].unique().tolist()
        annot_to_idx = {annot: i for i, annot in enumerate(self.unique_annots)}
        self.n_unique_annots = len(self.unique_annots)
        self.obs_to_annot_id = np.empty(len(self.data_df), dtype=np.uint32)
        for i, row in self.data_df.iterrows():
            annot = row["text"]
            self.obs_to_annot_id[i] = annot_to_idx[annot]

    def __len__(self):
        return len(self.data_df) * self.n_unique_annots

    def __getitem__(self, idx):
        annot_idx, obs_idx = divmod(idx, len(self.data_df))
        row = self.data_df.iloc[obs_idx]

        rgb = Image.open(os.path.join(self.data_dir, row["rgb_path"])).convert("RGB")
        xyz = np.load(os.path.join(self.data_dir, row["xyz_path"]))
        grasp_pose = np.load(os.path.join(self.data_dir, row["grasp_pose_path"]))

        xyz = torch.from_numpy(xyz).float()  # (H, W, 3)
        xyz = xyz.permute(2, 0, 1)  # (3, H, W)
        grasp_pose = torch.from_numpy(grasp_pose).float()  # 4x4 transform matrix

        annotation = self.unique_annots[annot_idx]
        text_input_ids, text_attention_mask = self.text_processor(annotation)

        if self.transform is not None:
            rgb, xyz, grasp_pose = self.transform(rgb, xyz, grasp_pose)

        rgb = self.img_processor(rgb)

        if rgb.shape[-2:] != xyz.shape[-2:]:
            xyz = trfF.resize(xyz, rgb.shape[-2:])

        label = torch.tensor([1 if self.obs_to_annot_id[obs_idx] == annot_idx else 0]).float()

        return {
            "rgb": rgb,
            "xyz": xyz,
            "grasp_pose": grasp_pose,
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "label": label,
        }

class GraspDescriptionClassificationSampler(Sampler):
    def __init__(self, dataset: GraspDescriptionClassificationDataset):
        self.dataset = dataset

    def __len__(self):
        return 2 * len(self.dataset.data_df)

    def __iter__(self):
        match_idxs = []
        nonmatch_idxs = []

        obs_ids = np.arange(len(self.dataset.data_df))
        matching_annot_ids = self.dataset.obs_to_annot_id[obs_ids]
        match_idxs = matching_annot_ids * len(self.dataset.data_df) + obs_ids

        nonmatching_annot_ids = np.random.randint(self.dataset.n_unique_annots-1, size=len(self.dataset.data_df))
        nonmatching_annot_ids[nonmatching_annot_ids >= matching_annot_ids] += 1
        nonmatch_idxs = nonmatching_annot_ids * len(self.dataset.data_df) + obs_ids

        idxs = np.concatenate([match_idxs, nonmatch_idxs])
        np.random.shuffle(idxs)

        return iter(idxs)


if __name__ == "__main__":
    dataset = GraspDescriptionRegressionDataset(
        csv_path="/net/nfs2.prior/abhayd/datasets/dataset.csv",
        data_dir="/net/nfs2.prior/abhayd/datasets",
        text_embedding_path="/net/nfs2.prior/abhayd/datasets/text_embeddings.npy",
        augment=True
    )
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    batch = next(iter(dataloader))
    breakpoint()
    print(batch)
