from typing import Any, Optional
from contextlib import nullcontext

from transformers import AutoModel, AutoProcessor
import torch
from torch import nn
import torch.nn.functional as F


class SiglipPatchFeatureExtractor(nn.Module):
    def __init__(self, checkpoint="google/siglip2-large-patch16-512"):
        super().__init__()
        self.checkpoint = checkpoint
        siglip = AutoModel.from_pretrained(checkpoint)
        self.siglip = siglip.vision_model
        del siglip

    def create_processor(self):
        processor = AutoProcessor.from_pretrained(self.checkpoint)
        def fn(rgb):
            inputs = processor(images=rgb, return_tensors="pt")
            return inputs["pixel_values"][0]
        return fn

    @property
    def embed_dim(self):
        return self.siglip.config.hidden_size

    @property
    def patch_size(self):
        return self.siglip.config.patch_size

    def forward(self, rgbs):
        """
        Expects (B, 3, H, W) rgbs, returns (B, H/patch_size*W/patch_size, embed_dim) features
        """
        outputs = self.siglip(pixel_values=rgbs)
        patch_features = outputs.last_hidden_state
        return patch_features


class CustomBatchNorm(nn.Module):
    def __init__(self, num_features: int, **kwargs):
        super().__init__()
        self.num_features = num_features
        self.batch_norm = nn.BatchNorm1d(num_features, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # batch norm expects (batch_size, dim, n_patches), but we have (batch_size, n_patches, dim)
        need_swap = x.ndim > 2
        if need_swap:
            x = x.transpose(-2, -1)
        x = self.batch_norm(x)
        if need_swap:
            x = x.transpose(-2, -1)
        return x


def create_mlp(input_dim: int, output_dim: int, layers: list[int], batch_norm: bool = True):
    ret = nn.Sequential()
    if batch_norm:
        ret.append(CustomBatchNorm(input_dim))
    layers = [input_dim] + layers
    for i in range(len(layers) - 1):
        ret.append(nn.Linear(layers[i], layers[i + 1]))
        ret.append(nn.ReLU())
    ret.append(nn.Linear(layers[-1], output_dim))
    return ret


class GraspEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()

        self.config = config
        embed_dim: int = config["embed_dim"]
        xyz_layers: list[int] = config["xyz_layers"]
        grasp_layers: list[int] = config["grasp_layers"]

        self.feature_extractor = SiglipPatchFeatureExtractor()
        self.feature_encoder = nn.Linear(self.feature_extractor.embed_dim, embed_dim)

        patch_size = self.feature_extractor.patch_size
        # kernel for averaging xyz over patches
        self.register_buffer(
            "xyz_kernel",
            torch.ones(3, 3, patch_size, patch_size) / patch_size**2,
            persistent=False,
        )
        # zero out channels of the kernel so that averaging is only within a dimension
        self.xyz_kernel[0, [1, 2]] = 0
        self.xyz_kernel[1, [0, 2]] = 0
        self.xyz_kernel[2, [0, 1]] = 0
        self.xyz_encoder = create_mlp(3, embed_dim, xyz_layers, batch_norm=True)

        # encoder for grasp pose
        self.grasp_pose_encoder = create_mlp(12, embed_dim, grasp_layers, batch_norm=True)
        self.grasp_pos_encoding = nn.Parameter(torch.randn(1, embed_dim))

        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.transformer_layer = nn.Transformer(
            batch_first=True,
            d_model=embed_dim,
            **self.config["transformer"],
        )

    def create_rgb_processor(self):
        return self.feature_extractor.create_processor()

    def forward(self, rgbs: torch.Tensor, xyzs: torch.Tensor, grasp_poses: torch.Tensor):
        """
        Expects (B, 3, H, W) rgbs, (B, 3, H, W) xyzs, (B, 4, 4) grasp_poses
        Returns (B, embed_dim) grasp_features
        """
        with torch.no_grad() if not self.config["train_vision_model"] else nullcontext():
            patch_features = self.feature_extractor(rgbs)  # (B, n_patches, siglip_dim)
        patch_features = self.feature_encoder(patch_features)  # (B, n_patches, embed_dim)

        xyz_patch = F.conv2d(xyzs, self.xyz_kernel, stride=self.feature_extractor.patch_size, padding=0)
        xyz_patch = xyz_patch.reshape(len(xyz_patch), 3, -1).transpose(1, 2)  # (B, n_patches, 3)
        xyz_features = self.xyz_encoder(xyz_patch)  # (B, n_patches, embed_dim)

        patch_xyz_features = patch_features + xyz_features

        grasp_poses = torch.cat([grasp_poses[:, :3, 3], grasp_poses[:, :3, :3].reshape(-1, 9)], dim=1)  # (B, 12)
        grasp_features: torch.Tensor = self.grasp_pose_encoder(grasp_poses)  # (B, embed_dim)
        grasp_features = grasp_features + self.grasp_pos_encoding
        grasp_features = grasp_features.unsqueeze(1)  # (B, 1, embed_dim)

        input_sequence = torch.cat([patch_xyz_features, grasp_features], dim=1)  # (B, n_patches + 1, embed_dim)

        query_tokens = self.query_token.repeat(input_sequence.shape[0], 1, 1)
        output: torch.Tensor = self.transformer_layer(input_sequence, query_tokens)
        output = output[:, 0, :]
        return output / torch.linalg.norm(output, dim=-1, keepdim=True)

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    import yaml
    with open("config/params.yaml", "r") as f:
        config = yaml.safe_load(f)
    model = GraspEncoder(config["grasp_encoder"]).to(device)
    model.eval()
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Total number of params in transformer: {sum(p.numel() for p in model.transformer_layer.parameters()):,}")
    batch_size = 12
    with torch.autocast(device_type=device, dtype=torch.float16):
        rgbs = torch.rand(batch_size, 3, 512, 512).to(device)
        xyzs = torch.rand(batch_size, 3, 512, 512).to(device)
        grasp_poses = torch.randn(batch_size, 4, 4).to(device)
        with torch.no_grad():
            import time
            for _ in range(10):
                start = time.perf_counter()
                grasp_features = model(rgbs, xyzs, grasp_poses)
                print(time.perf_counter() - start)
    print(grasp_features.shape)
