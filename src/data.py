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

    def __call__(self, rgb: Image.Image, xyz: torch.Tensor, grasp_pose: torch.Tensor, normals: torch.Tensor):
        """
        rgb: PIL Image
        xyz: (3, H, W) torch.Tensor
        grasp_pose: (4, 4) torch.Tensor
        normals: (3, H, W) torch.Tensor
        """

        # Random horizontal flip
        if torch.rand(1) < self.horizontal_flip_prob:
            rgb = trfF.horizontal_flip(rgb)
            xyz = torch.flip(xyz, dims=[-1])
            xyz[0] = -xyz[0]  # Flip x-coordinates
            normals = torch.flip(normals, dims=[-1])
            normals[0] = -normals[0]  # Flip x-coordinates
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
        normals[:, xyz[2] == 0] = 0.0

        # Random flip grasp pose
        if torch.rand(1) < self.flip_grasp_prob:
            grasp_pose = grasp_pose @ self.flip_grasp_trf

        return rgb, xyz, grasp_pose, normals

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
        pc_processor: Optional[Callable[[dict], torch.Tensor]] = None,
        augment: bool = True,
        augmentation_params: Optional[dict] = None,
        xyz_far_clip: float = 1.0,
    ):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path)
        self.text_embeddings = np.load(text_embedding_path)
        self.xyz_far_clip = xyz_far_clip
        self.pc_processor = pc_processor
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
        normals: np.ndarray = obs['normals']
        grasp_pose: np.ndarray = obs['grasp_pose']
        rgb_array = torch.from_numpy(np.array(rgb)).float().permute(2, 0, 1)

        xyz: torch.Tensor = torch.from_numpy(xyz).float()  # (H, W, 3)
        xyz = xyz.permute(2, 0, 1)  # (3, H, W)
        normals: torch.Tensor = torch.from_numpy(normals).float().permute(2, 0, 1)  # (3, H, W)
        grasp_pose: torch.Tensor = torch.from_numpy(grasp_pose).float()  # 4x4 transform matrix

        far_clip_mask = xyz[2] >= self.xyz_far_clip
        xyz[:, far_clip_mask] = 0.0
        normals[:, far_clip_mask] = 0.0

        if self.transform is not None:
            rgb, xyz, grasp_pose, normals = self.transform(rgb, xyz, grasp_pose, normals)

        rgb = self.img_processor(rgb)

        if rgb.shape[-2:] != xyz.shape[-2:]:
            xyz = trfF.resize(xyz, rgb.shape[-2:], interpolation=InterpolationMode.NEAREST_EXACT)
            normals = trfF.resize(normals, rgb.shape[-2:], interpolation=InterpolationMode.NEAREST_EXACT)
        if rgb.shape[-2:] != rgb_array.shape[-2:]:
            rgb_array = trfF.resize(rgb_array, rgb.shape[-2:])

        valid_mask = xyz[2] > 0.0  # (H, W)

        pc_inputs = {
            "coord": xyz[:, valid_mask].T,  # (*, 3)
            "color": rgb_array[:, valid_mask].T,  # (*, 3)
            "normal": normals[:, valid_mask].T,  # (*, 3)
        }

        return {
            'annotation_id': row['annotation_id'],
            'text_embedding': self.text_embeddings[idx],
            'rgb': rgb,
            'grasp_pose': grasp_pose,
            'xyz_inputs': {
                'xyz': xyz,
                'valid_mask': valid_mask,
                **pc_inputs,
            }
        }

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        ret = {
            "annotation_id": [b["annotation_id"] for b in batch],
            "text_embedding": torch.stack([torch.as_tensor(b["text_embedding"]) for b in batch]),
            "rgb": torch.stack([torch.as_tensor(b["rgb"]) for b in batch]),
            "grasp_pose": torch.stack([torch.as_tensor(b["grasp_pose"]) for b in batch]),
            "xyz_inputs": {
                "xyz": torch.stack([torch.as_tensor(b["xyz_inputs"]["xyz"]) for b in batch]),
                "valid_mask": torch.stack([torch.as_tensor(b["xyz_inputs"]["valid_mask"]) for b in batch]),
            },
        }

        point = {
            "coord": torch.cat([torch.as_tensor(b["xyz_inputs"]["coord"]) for b in batch]).numpy(),
            "color": torch.cat([torch.as_tensor(b["xyz_inputs"]["color"]) for b in batch]).numpy(),
            "normal": torch.cat([torch.as_tensor(b["xyz_inputs"]["normal"]) for b in batch]).numpy(),
            "batch": np.concatenate([[i] * len(b["xyz_inputs"]["coord"]) for i, b in enumerate(batch)]),
        }
        if self.pc_processor is not None:
            point = self.pc_processor(point)
        ret["xyz_inputs"].update(point)

        return ret

class GraspDescriptionClassificationDataset(Dataset):
    def __init__(
        self,
        data_df: pd.DataFrame,
        data_dir: str,
        img_processor: Callable[[Image.Image], torch.Tensor],
        text_processor: Callable[[str], tuple[torch.Tensor, torch.Tensor]],
        text_embeddings: Optional[np.ndarray] = None,
        augment: bool = True,
        augmentation_params: Optional[dict] = None
    ):
        self.data_dir = data_dir
        self.data_df = data_df
        self.text_embeddings = text_embeddings
        aug_params = augmentation_params or {}
        self.transform = ImageAugmentation(**aug_params) if augment else None
        self.img_processor = img_processor
        self.text_processor = text_processor

        self.unique_annots = self.data_df["annot"].unique().tolist()
        annot_to_idx = {annot: i for i, annot in enumerate(self.unique_annots)}
        self.n_unique_annots = len(self.unique_annots)
        self.obs_to_annot_id = np.empty(len(self.data_df), dtype=np.uint32)
        for i, row in self.data_df.iterrows():
            annot = row["annot"]
            self.obs_to_annot_id[i] = annot_to_idx[annot]

    def __len__(self):
        return len(self.data_df) * self.n_unique_annots

    @classmethod
    def load(
        cls,
        csv_path: str,
        data_dir: str,
        img_processor: Callable[[Image.Image], torch.Tensor],
        text_processor: Callable[[str], tuple[torch.Tensor, torch.Tensor]],
        text_embedding_path: Optional[str] = None,
        use_frozen_text_embeddings: bool = False,
        augment: bool = True,
        augmentation_params: Optional[dict] = None
    ):
        df = pd.read_csv(csv_path)
        text_embeddings = np.load(text_embedding_path) if text_embedding_path is not None and use_frozen_text_embeddings else None
        return cls(
            df,
            data_dir,
            img_processor,
            text_processor,
            text_embeddings,
            augment,
            augmentation_params
        )

    @classmethod
    def load_split(
        cls,
        csv_path: str,
        data_dir: str,
        img_processor: Callable[[Image.Image], torch.Tensor],
        text_processor: Callable[[str], tuple[torch.Tensor, torch.Tensor]],
        fracs: list[float],
        augment: bool = True,
        text_embedding_path: str = None,
        use_frozen_text_embeddings: bool = False,
        augmentation_params: Optional[dict] = None,
        seed: Optional[int] = None
    ) -> list["GraspDescriptionClassificationDataset"]:
        assert np.isclose(sum(fracs), 1.0)
        if len(fracs) == 1:
            return [cls.load(csv_path, data_dir, img_processor, text_processor, augment, augmentation_params)]
        df = pd.read_csv(csv_path)
        text_embeddings = np.load(text_embedding_path) if text_embedding_path is not None and use_frozen_text_embeddings else None
        n_rows = len(df)
        shuffled_indices = np.random.RandomState(seed=seed).permutation(n_rows)

        # Compute partition sizes and split indices
        sizes = (np.array(fracs) * n_rows).astype(int)
        sizes[-1] += n_rows - sizes.sum()  # Ensure the sum of sizes equals n_rows
        indices = np.split(shuffled_indices, np.cumsum(sizes)[:-1])

        subsets = []
        for idxs in indices:
            sub_df = df.iloc[idxs].reset_index(drop=True, inplace=False)
            sub_embeddings = text_embeddings[idxs] if text_embeddings is not None else None
            dataset = cls(
                sub_df,
                data_dir,
                img_processor,
                text_processor,
                sub_embeddings,
                augment,
                augmentation_params
            )
            subsets.append(dataset)
        return subsets

    def __getitem__(self, idx):
        annot_idx, obs_idx = divmod(idx, len(self.data_df))
        row = self.data_df.iloc[obs_idx]

        obs = load_obs(self.data_dir, row)
        rgb: Image.Image = obs['rgb']
        xyz: np.ndarray = obs['xyz']
        grasp_pose: np.ndarray = obs['grasp_pose']
        normals: np.ndarray = obs['normals']

        xyz = torch.from_numpy(xyz).float()  # (H, W, 3)
        xyz = xyz.permute(2, 0, 1)  # (3, H, W)
        normals = torch.from_numpy(normals).float().permute(2, 0, 1)  # (3, H, W)
        grasp_pose = torch.from_numpy(grasp_pose).float()  # 4x4 transform matrix

        far_clip_mask = xyz[2] >= self.xyz_far_clip
        xyz[:, far_clip_mask] = 0.0
        normals[:, far_clip_mask] = 0.0

        if self.text_embeddings is not None:
            text_embedding = self.text_embeddings[obs_idx]
            text_inputs = {
                "text_embedding": text_embedding,
            }
        else:
            annotation = self.unique_annots[annot_idx]
            text_input_ids, text_attention_mask = self.text_processor(annotation)
            text_inputs = {
                "input_ids": text_input_ids,
                "attention_mask": text_attention_mask,
            }

        if self.transform is not None:
            rgb, xyz, grasp_pose, normals = self.transform(rgb, xyz, grasp_pose, normals)

        rgb = self.img_processor(rgb)

        if rgb.shape[-2:] != xyz.shape[-2:]:
            xyz = trfF.resize(xyz, rgb.shape[-2:])

        label = torch.tensor([1 if self.obs_to_annot_id[obs_idx] == annot_idx else 0]).float()

        # TODO: return normals
        return {
            "rgb": rgb,
            "xyz": xyz,
            "grasp_pose": grasp_pose,
            "text_inputs": text_inputs,
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
