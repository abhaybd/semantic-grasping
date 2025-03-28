from typing import Any

from .vit import ViTEncoder

def create_xyz_encoder(**config: Any):
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
