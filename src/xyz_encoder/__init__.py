from typing import Any, Protocol, Callable

import torch

from .vit import ViTEncoder

class XYZEncoder(Protocol, torch.nn.Module):
    def create_processor(self) -> Callable[[Any], torch.Tensor] | None:
        ...

    @property
    def embed_dim(self) -> int:
        ...

def create_xyz_encoder(**config: Any) -> XYZEncoder:
    if "type" in config:
        enc_type = config["type"]
        config = config.copy()
        del config["type"]
    else:
        enc_type = "vit"

    match enc_type:
        case "vit":
            return ViTEncoder(**config)
        case _:
            raise ValueError(f"Unknown xyz encoder type: {enc_type}")
