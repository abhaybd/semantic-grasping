from transformers import AutoModel
import torch
from torch import nn
import torch.nn.functional as F

class SiglipPatchFeatureExtractor(nn.Module):
    def __init__(self, checkpoint="google/siglip2-large-patch16-512"):
        super().__init__()
        siglip = AutoModel.from_pretrained(checkpoint)
        self.siglip = siglip.vision_model
        del siglip

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


class GraspEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim

        self.feature_extractor = SiglipPatchFeatureExtractor()
        self.feature_encoder = nn.Linear(self.feature_extractor.embed_dim, embed_dim)

        patch_size = self.feature_extractor.patch_size
        # kernel for averaging xyz over patches
        self.register_buffer(
            "xyz_kernel",
            torch.ones(3, 3, patch_size, patch_size) / patch_size**2 / 3,
            persistent=False,
        )
        self.xyz_encoder = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim),
        )

        # encoder for grasp pose
        self.grasp_pose_encoder = nn.Sequential(
            nn.Linear(12, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim),
        )
        self.grasp_pos_encoding = nn.Parameter(torch.randn(1, embed_dim))

        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.transformer_layer = nn.Transformer(
            d_model=embed_dim,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            batch_first=True,
        )

    def forward(self, rgbs: torch.Tensor, xyzs: torch.Tensor, grasp_poses: torch.Tensor):
        """
        Expects (B, 3, H, W) rgbs, (B, 3, H, W) xyzs, (B, 4, 4) grasp_poses
        Returns (B, embed_dim) grasp_features
        """
        patch_features = self.feature_extractor(rgbs)  # (B, n_patches, siglip_dim)
        patch_features = self.feature_encoder(patch_features)  # (B, n_patches, embed_dim)

        xyz_patch = F.conv2d(xyzs, self.xyz_kernel, stride=self.feature_extractor.patch_size, padding=0)
        xyz_patch = xyz_patch.reshape(len(xyz_patch), 3, -1).transpose(1, 2)  # (B, n_patches, 3)
        xyz_features = self.xyz_encoder(xyz_patch)  # (B, n_patches, embed_dim)

        patch_xyz_features = patch_features + xyz_features

        grasp_poses = torch.cat([grasp_poses[:, :3, 3], grasp_poses[:, :3, :3].reshape(-1, 9)], dim=1)  # (B, 12)
        assert grasp_poses.shape == (len(rgbs), 12)
        grasp_features: torch.Tensor = self.grasp_pose_encoder(grasp_poses)  # (B, embed_dim)
        grasp_features = grasp_features + self.grasp_pos_encoding
        grasp_features = grasp_features.unsqueeze(1)  # (B, 1, embed_dim)

        input_sequence = torch.cat([patch_xyz_features, grasp_features], dim=1)  # (B, n_patches + 1, embed_dim)

        query_tokens = self.query_token.repeat(input_sequence.shape[0], 1, 1)
        output: torch.Tensor = self.transformer_layer(input_sequence, query_tokens)
        return output[:, 0, :]

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = GraspEncoder().to(device)
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
