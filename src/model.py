import os
import io
from typing import Any, Protocol, Optional
import math
from contextlib import nullcontext
import yaml

from transformers import AutoModel, AutoProcessor, T5Tokenizer, T5EncoderModel
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.models import VisionTransformer
from torchvision.transforms.v2.functional import resize

from beaker import Beaker

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
        ckpt_path = os.path.join(self.ckpt_dir, f"ckpt_{step}.pth")
        ckpt = torch.load(ckpt_path, weights_only=True)
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
    def __init__(self, checkpoint="google/siglip2-large-patch16-512", frozen=True):
        super().__init__()
        self.checkpoint = checkpoint
        siglip = AutoModel.from_pretrained(checkpoint)
        self.siglip = siglip.vision_model
        del siglip
        if frozen:
            for param in self.siglip.parameters():
                param.requires_grad = False

    def create_processor(self):
        processor = AutoProcessor.from_pretrained(self.checkpoint)
        def fn(rgb):
            rgb = resize(rgb, (self.image_size, self.image_size))
            inputs = processor(images=rgb, return_tensors="pt")
            return inputs["pixel_values"][0]
        return fn

    @property
    def embed_dim(self):
        return self.siglip.config.hidden_size

    @property
    def image_size(self):
        return self.siglip.config.image_size

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


class T5TextEncoder(nn.Module):
    def __init__(self, checkpoint="google-t5/t5-base", max_length=128, frozen=True):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(checkpoint, legacy=True)
        self.encoder = T5EncoderModel.from_pretrained(checkpoint)
        self.max_length = max_length
        if frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False

    @property
    def embed_dim(self):
        return self.encoder.config.d_model

    def create_processor(self):
        def fn(text: str | list[str]) -> tuple[torch.LongTensor, torch.FloatTensor]:
            inputs = self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=self.max_length)
            if isinstance(text, str):
                return inputs["input_ids"][0], inputs["attention_mask"][0]
            else:
                return inputs["input_ids"], inputs["attention_mask"]
        return fn

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor | None=None):
        """
        Expects (B, max_length) input_ids, (B, max_length) attention_mask
        Returns (B, max_length, embed_dim) features
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

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


def create_mlp(input_dim: int, output_dim: int, layers: list[int], layer_norm=True, final_activation=True):
    ret = nn.Sequential()
    layers = [input_dim] + layers
    for i in range(len(layers) - 1):
        ret.append(nn.Linear(layers[i], layers[i + 1]))
        if layer_norm:
            ret.append(nn.LayerNorm(layers[i + 1]))
        ret.append(nn.ReLU())
    ret.append(nn.Linear(layers[-1], output_dim))
    if final_activation:
        if layer_norm:
            ret.append(nn.LayerNorm(output_dim))
        ret.append(nn.ReLU())
    return ret

class ViTEncoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = VisionTransformer(**kwargs)

    def forward(self, x):
        """
        Expects (B, 3, H, W) input, returns (B, n_patches, hidden_dim) features
        """
        x = self.model._process_input(x)
        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(len(x), -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.model.encoder(x)
        x = x[:, 1:]  # Remove the class token
        return x

    @property
    def embed_dim(self):
        return self.model.hidden_dim

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
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
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[0, :x.size(1)]
        return x


class Model(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config

    @classmethod
    def from_beaker(cls, dataset: str, ckpt: int | None = None, map_location="cpu", config: dict[str, Any] | None = None):
        beaker = Beaker.from_env()

        if config is None:
            cfg_bytes = io.BytesIO(beaker.dataset.get_file(dataset, ".hydra/config.yaml"))
            config = yaml.safe_load(cfg_bytes)

        def download(remote_path: str):
            local_path = f"/tmp/semantic-grasping/{dataset}/{remote_path}"
            if not os.path.isfile(local_path):
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, "wb") as f:
                    for chunk in beaker.dataset.stream_file(dataset, remote_path):
                        f.write(chunk)
            return local_path

        if ckpt is not None:
            ckpt_fileinfos = beaker.dataset.ls(dataset, "checkpoints/")
            ckpt_files = [fi.path for fi in ckpt_fileinfos]
            ckpt_file = min(ckpt_files, key=lambda x: abs(int(x[:-len(".pth")].split("_")[-1]) - ckpt))
            ckpt_fn = download(ckpt_file)
            ckpt = torch.load(ckpt_fn, map_location=map_location, weights_only=True)
            state_dict = ckpt["model"]
        else:
            ckpt_fn = download("model.pt")
            state_dict = torch.load(ckpt_fn, map_location=map_location, weights_only=True)
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items() if "siglip" not in k.lower()}
        model = cls(config["model"])
        model.load_state_dict(state_dict, strict=False)
        return model

    @classmethod
    def from_wandb(cls, run_id: str, ckpt: int | None = None, map_location="cpu"):
        import wandb
        run_path = f"prior-ai2/semantic-grasping/{run_id}"
        api = wandb.Api()
        run = api.run(run_path)
        config = run.config

        if ckpt is not None:
            dataset_id = config["env"]["BEAKER_RESULT_DATASET_ID"]
            return cls.from_beaker(dataset_id, ckpt, map_location=map_location, config=config)

        dl_path = f"/tmp/semantic-grasping/{run_id}"
        os.makedirs(dl_path, exist_ok=True)
        weights_file = wandb.restore("model.pt", run_path, root=dl_path)
        with open(weights_file.name, "rb") as f:
            state_dict = torch.load(f, map_location=map_location, weights_only=True)
        new_state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items() if "siglip" not in k.lower()}

        model = cls(config["model"])
        model.load_state_dict(new_state_dict, strict=False)
        return model

class GraspEncoder(Model):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

        hidden_dim: int = config["hidden_dim"]
        embed_dim: int = config["embed_dim"]
        feature_layers: list[int] = config["feature_layers"]
        xyz_feature_layers: list[int] = config["xyz_feature_layers"]
        grasp_layers: list[int] = config["grasp_layers"]
        final_fc_layers: list[int] = config["final_fc_layers"]
        self.feature_extractor = SiglipPatchFeatureExtractor(**config["rgb_encoder"])
        self.feature_encoder = create_mlp(self.feature_extractor.embed_dim, hidden_dim, feature_layers)

        self.pos_encoding = PositionalEncoding(hidden_dim)

        self.xyz_feature_extractor = ViTEncoder(
            image_size=self.feature_extractor.image_size,
            patch_size=self.feature_extractor.patch_size,
            **config["xyz_encoder"]
        )
        self.xyz_feature_encoder = create_mlp(self.xyz_feature_extractor.embed_dim, hidden_dim, xyz_feature_layers)

        # encoder for grasp pose
        self.grasp_pose_encoder = create_mlp(12, hidden_dim, grasp_layers)

        self.feature_ln = nn.LayerNorm(hidden_dim)
        self.class_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.trf_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=self.config["transformer"]["nhead"],
                dim_feedforward=int(hidden_dim * 4),
                dropout=0.0,
                norm_first=True,
                batch_first=True
            ),
            num_layers=self.config["transformer"]["num_encoder_layers"]
        )
        self.final_fc = create_mlp(hidden_dim, embed_dim, final_fc_layers, final_activation=False)

    def create_rgb_processor(self):
        return self.feature_extractor.create_processor()

    def forward(self, rgbs: torch.Tensor, xyzs: torch.Tensor, grasp_poses: torch.Tensor):
        """
        Expects (B, 3, H, W) rgbs, (B, 3, H, W) xyzs, (B, 4, 4) grasp_poses
        Returns (B, embed_dim) grasp_features
        """
        patch_features = self.feature_extractor(rgbs)  # (B, n_patches, siglip_dim)
        patch_features = self.feature_encoder(patch_features)  # (B, n_patches, hidden_dim)

        xyz_features = self.xyz_feature_extractor(xyzs)  # (B, n_patches, xyz_feature_dim)
        xyz_features = self.xyz_feature_encoder(xyz_features)  # (B, n_patches, hidden_dim)

        assert len(patch_features) == len(xyz_features)
        if len(patch_features) == 1 and len(grasp_poses) > 1:
            patch_features = patch_features.expand(len(grasp_poses), -1, -1)
            xyz_features = xyz_features.expand(len(grasp_poses), -1, -1)

        grasp_poses = torch.cat([grasp_poses[:, :3, 3], grasp_poses[:, :3, :3].reshape(-1, 9)], dim=1)  # (B, 12)
        grasp_features: torch.Tensor = self.grasp_pose_encoder(grasp_poses)  # (B, hidden_dim)
        grasp_features = grasp_features.unsqueeze(1)  # (B, 1, hidden_dim)

        input_sequence = torch.cat([patch_features, xyz_features, grasp_features], dim=1)  # (B, *, hidden_dim)
        input_sequence = self.feature_ln(input_sequence)

        class_tokens = self.class_token.repeat(input_sequence.shape[0], 1, 1)  # (B, 1, hidden_dim)
        sequence = torch.cat([class_tokens, input_sequence], dim=1)
        sequence = self.pos_encoding(sequence)

        output = self.trf_encoder(sequence)
        output = output[:, 0, :]
        output = self.final_fc(output)
        return output / torch.linalg.norm(output, dim=-1, keepdim=True)

class GraspClassifier(Model):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

        hidden_dim: int = config["hidden_dim"]
        feature_layers: list[int] = config["feature_layers"]
        xyz_feature_layers: list[int] = config["xyz_feature_layers"]
        grasp_layers: list[int] = config["grasp_layers"]
        final_fc_layers: list[int] = config["final_fc_layers"]
        text_feature_layers: list[int] = config["text_feature_layers"]

        self.pos_encoding = PositionalEncoding(hidden_dim)

        self.rgb_feature_extractor = SiglipPatchFeatureExtractor(**config["rgb_encoder"])
        self.rgb_feature_encoder = create_mlp(self.rgb_feature_extractor.embed_dim, hidden_dim, feature_layers)

        self.xyz_feature_extractor = ViTEncoder(
            image_size=self.rgb_feature_extractor.image_size,
            patch_size=self.rgb_feature_extractor.patch_size,
            **config["xyz_encoder"]
        )
        self.xyz_feature_encoder = create_mlp(self.xyz_feature_extractor.embed_dim, hidden_dim, xyz_feature_layers)

        # encoder for grasp pose
        self.grasp_pose_encoder = create_mlp(12, hidden_dim, grasp_layers)

        self.text_feature_extractor = T5TextEncoder(**config["text_encoder"])
        self.text_feature_encoder = create_mlp(self.text_feature_extractor.embed_dim, hidden_dim, text_feature_layers)
        self.feature_ln = nn.LayerNorm(hidden_dim)

        self.class_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.trf_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=self.config["transformer"]["nhead"],
                dim_feedforward=int(hidden_dim * 4),
                dropout=0.0,
                norm_first=True,
                batch_first=True
            ),
            num_layers=self.config["transformer"]["num_encoder_layers"]
        )
        self.final_fc = create_mlp(hidden_dim, 1, final_fc_layers, final_activation=False)

    def create_rgb_processor(self):
        return self.rgb_feature_extractor.create_processor()

    def create_text_processor(self):
        return self.text_feature_extractor.create_processor()

    def forward(self,
        rgbs: torch.Tensor,
        xyzs: torch.Tensor,
        grasp_poses: torch.Tensor,
        text_input_ids: torch.LongTensor,
        text_attention_mask: Optional[torch.FloatTensor] = None
    ):
        """
        Expects (B, 3, H, W) rgbs, (B, 3, H, W) xyzs, (B, 4, 4) grasp_poses, (B, max_length) text_input_ids, (B, max_length) text_attention_mask
        Returns (B, 1) grasp classification logits
        """
        with torch.no_grad():
            patch_features = self.rgb_feature_extractor(rgbs)  # (B, n_patches, siglip_dim)
        patch_features = self.rgb_feature_encoder(patch_features)  # (B, n_patches, hidden_dim)

        xyz_features = self.xyz_feature_extractor(xyzs)  # (B, n_patches, xyz_feature_dim)
        xyz_features = self.xyz_feature_encoder(xyz_features)  # (B, n_patches, hidden_dim)

        grasp_poses = torch.cat([grasp_poses[:, :3, 3], grasp_poses[:, :3, :3].reshape(-1, 9)], dim=1)  # (B, 12)
        grasp_features: torch.Tensor = self.grasp_pose_encoder(grasp_poses)  # (B, hidden_dim)
        grasp_features = grasp_features.unsqueeze(1)  # (B, 1, hidden_dim)

        text_features = self.text_feature_extractor(text_input_ids, text_attention_mask)  # (B, n_tokens, t5_dim)
        text_features = self.text_feature_encoder(text_features)  # (B, n_tokens, hidden_dim)
    
        assert len(patch_features) == len(xyz_features) == len(text_features)
        if len(patch_features) == 1 and len(grasp_poses) > 1:
            patch_features = patch_features.expand(len(grasp_poses), -1, -1)
            xyz_features = xyz_features.expand(len(grasp_poses), -1, -1)
            text_features = text_features.expand(len(grasp_poses), -1, -1)

        # input_sequence = torch.cat([patch_features, text_features, grasp_features], dim=1)  # (B, *, hidden_dim)
        input_sequence = torch.cat([patch_features, xyz_features, grasp_features, text_features], dim=1)  # (B, *, hidden_dim)
        input_sequence = self.feature_ln(input_sequence)

        class_tokens = self.class_token.expand(input_sequence.shape[0], -1, -1)  # (B, 1, hidden_dim)
        sequence = torch.cat([class_tokens, input_sequence], dim=1)
        sequence = self.pos_encoding(sequence)

        output = self.trf_encoder(sequence)
        output = output[:, 0, :]
        output = self.final_fc(output)
        return output


def main_regression():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    import yaml
    with open("config/regression.yaml", "r") as f:
        config = yaml.safe_load(f)
    model = GraspEncoder(config["model"]).to(device)
    model.eval()
    print(model.__class__.__name__)
    print(f"\tTotal number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\tTotal number of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    batch_size = 12
    with torch.autocast(device_type=device, dtype=torch.float16):
        rgbs = torch.rand(batch_size, 3, 512, 512).to(device)
        xyzs = torch.rand(batch_size, 3, 512, 512).to(device)
        grasp_poses = torch.randn(batch_size, 4, 4).to(device)
        with torch.no_grad():
            out = model(rgbs, xyzs, grasp_poses)
    print(f"\tOutput shape: {out.shape}")

def main_classification():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    import yaml
    with open("config/classification.yaml", "r") as f:
        config = yaml.safe_load(f)
    model = GraspClassifier(config["model"]).to(device)
    model.eval()
    print(model.__class__.__name__)
    print(f"\tTotal number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\tTotal number of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    batch_size = 12
    with torch.autocast(device_type=device, dtype=torch.float16):
        rgbs = torch.rand(batch_size, 3, 512, 512).to(device)
        xyzs = torch.rand(batch_size, 3, 512, 512).to(device)
        grasp_poses = torch.randn(batch_size, 4, 4).to(device)
        texts = ["This is a test"] * batch_size
        text_input_ids, text_attention_mask = [x.to(device) for x in model.create_text_processor()(texts)]
        with torch.no_grad():
            out = model(rgbs, xyzs, grasp_poses, text_input_ids, text_attention_mask)
    print(f"\tOutput shape: {out.shape}")

if __name__ == "__main__":
    main_regression()
    main_classification()
