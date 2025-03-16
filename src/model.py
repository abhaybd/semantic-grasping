import os
from typing import Any, Protocol
import math
from contextlib import nullcontext
import yaml

from transformers import AutoModel, AutoProcessor
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

class StateDictProtocol(Protocol):
    def state_dict(self) -> dict[str, Any]:
        ...

    def load_state_dict(self, state_dict: dict[str, Any]):
        ...


class Checkpointer:
    def __init__(self, ckpt_dir: str, **modules: StateDictProtocol):
        self.ckpt_dir = ckpt_dir
        self.modules = modules

    def save(self, step: int):
        ckpt = {
            "step": step,
            **{k: v.state_dict() for k, v in self.modules.items()}
        }
        torch.save(ckpt, os.path.join(self.ckpt_dir, f"ckpt_{step}.pth"))

    def load(self, step: int | None = None) -> int:
        if step is None:
            steps = []
            for f in os.listdir(self.ckpt_dir):
                if f.startswith("ckpt_") and f.endswith(".pth"):
                    e = int(f[:-len(".pth")].split("_")[-1])
                    steps.append(e)
            if len(steps) == 0:
                return 0
            step = max(steps)
        else:
            ckpt_path = os.path.join(self.ckpt_dir, f"ckpt_{step}.pth")
        ckpt = torch.load(ckpt_path)
        for k, v in self.modules.items():
            v.load_state_dict(ckpt[k])
        return step

class WarmupCosineLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int, total_steps: int, final_factor: float = 0.1):
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps  # Linear warmup
            else:
                decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
                return final_factor + 0.5 * (1 - final_factor) * (1 + math.cos(math.pi * decay_ratio))  # Cosine decay

        super().__init__(optimizer, lr_lambda)

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

    def embed(self, rgbs):
        outputs = self.siglip(pixel_values=rgbs)
        return outputs.pooler_output


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


def create_mlp(input_dim: int, output_dim: int, layers: list[int], batch_norm: bool = False):
    ret = nn.Sequential()
    if batch_norm:
        ret.append(CustomBatchNorm(input_dim))
    layers = [input_dim] + layers
    for i in range(len(layers) - 1):
        ret.append(nn.Linear(layers[i], layers[i + 1]))
        ret.append(nn.ReLU())
    ret.append(nn.Linear(layers[-1], output_dim))
    return ret


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[bach_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return x

class GraspEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()

        self.config = config
        hidden_dim: int = config["hidden_dim"]
        embed_dim: int = config["embed_dim"]
        feature_layers: list[int] = config["feature_layers"]
        xyz_layers: list[int] = config["xyz_layers"]
        grasp_layers: list[int] = config["grasp_layers"]
        final_fc_layers: list[int] = config["final_fc_layers"]
        self.feature_extractor = SiglipPatchFeatureExtractor()
        self.feature_encoder = create_mlp(self.feature_extractor.embed_dim, hidden_dim, feature_layers)

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
        self.xyz_encoder = create_mlp(3, hidden_dim, xyz_layers, batch_norm=False)

        # encoder for grasp pose
        self.grasp_pose_encoder = create_mlp(12, hidden_dim, grasp_layers, batch_norm=False)
        # self.grasp_pos_encoding = nn.Parameter(torch.randn(1, hidden_dim))

        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.trf_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=self.config["transformer"]["nhead"],
                dim_feedforward=int(hidden_dim * 4),
                dropout=0.0,
                batch_first=True
            ),
            num_layers=self.config["transformer"]["num_encoder_layers"]
        )
        self.final_fc = create_mlp(hidden_dim, embed_dim, final_fc_layers)

        self.pos_embeddings = PositionalEncoding(hidden_dim)

    @classmethod
    def from_wandb(cls, run_id: str, ckpt: int | None = None, map_location="cpu"):
        assert ckpt is None, "Checkpoint loading not supported yet"
        import wandb
        dl_path = f"/tmp/semantic-grasping/{run_id}"
        os.makedirs(dl_path, exist_ok=True)
        run_path = f"prior-ai2/semantic-grasping/{run_id}"

        cfg_file = wandb.restore("config.yaml", run_path, root=dl_path)
        with open(cfg_file.name, "r") as f:
            config = yaml.safe_load(f)

        weights_file = wandb.restore("grasp_encoder.pt", run_path, root=dl_path)
        with open(weights_file.name, "rb") as f:
            state_dict = torch.load(f, map_location=map_location, weights_only=True)
        new_state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items() if "siglip" not in k.lower()}

        model = cls(config["grasp_encoder"]["value"])
        model.load_state_dict(new_state_dict, strict=False)
        return model

    def create_rgb_processor(self):
        return self.feature_extractor.create_processor()

    def forward(self, rgbs: torch.Tensor, xyzs: torch.Tensor, grasp_poses: torch.Tensor):
        """
        Expects (B, 3, H, W) rgbs, (B, 3, H, W) xyzs, (B, 4, 4) grasp_poses
        Returns (B, embed_dim) grasp_features
        """
        with torch.no_grad() if not self.config["train_vision_model"] else nullcontext():
            patch_features = self.feature_extractor(rgbs)  # (B, n_patches, siglip_dim)
        patch_features = self.feature_encoder(patch_features)  # (B, n_patches, hidden_dim)

        xyz_patch = F.conv2d(xyzs, self.xyz_kernel, stride=self.feature_extractor.patch_size, padding=0)
        xyz_patch = xyz_patch.reshape(len(xyz_patch), 3, -1).transpose(1, 2)  # (B, n_patches, 3)
        xyz_features = self.xyz_encoder(xyz_patch)  # (B, n_patches, hidden_dim)

        patch_xyz_features = patch_features + xyz_features
        # If we're evaluating multiple grasps in a single image, evaluate patch features only once
        if len(patch_xyz_features) == 1 and len(grasp_poses) > 1:
            patch_xyz_features = patch_xyz_features.repeat(len(grasp_poses), 1, 1)

        grasp_poses = torch.cat([grasp_poses[:, :3, 3], grasp_poses[:, :3, :3].reshape(-1, 9)], dim=1)  # (B, 12)
        grasp_features: torch.Tensor = self.grasp_pose_encoder(grasp_poses)  # (B, hidden_dim)
        # grasp_features = grasp_features + self.grasp_pos_encoding
        grasp_features = grasp_features.unsqueeze(1)  # (B, 1, hidden_dim)

        input_sequence = torch.cat([patch_xyz_features, grasp_features], dim=1)  # (B, n_patches + 1, hidden_dim)

        query_tokens = self.query_token.repeat(input_sequence.shape[0], 1, 1)  # (B, 1, hidden_dim)
        sequence = torch.cat([query_tokens, input_sequence], dim=1)
        embedded_sequence = self.pos_embeddings(sequence)
        output = self.trf_encoder(embedded_sequence)
        output = output[:, 0, :]
        output = self.final_fc(output)
        return output / torch.linalg.norm(output, dim=-1, keepdim=True)

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    import yaml
    with open("config/regression.yaml", "r") as f:
        config = yaml.safe_load(f)
    model = GraspEncoder(config["grasp_encoder"]).to(device)
    model.eval()
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Total number of params in transformer: {sum(p.numel() for p in model.trf_encoder.parameters()):,}")
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
