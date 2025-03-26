from typing import Any

from .vit import ViTEncoder

def create_xyz_encoder(**config: Any):
    enc_type = config["type"]
    config = config.copy()
    del config["type"]
    match enc_type:
        case "vit":
            return ViTEncoder(**config)
        case _:
            raise ValueError(f"Unknown xyz encoder type: {enc_type}")
