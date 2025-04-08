import numpy as np
import torch

import sonata

class SonataEncoder(torch.nn.Module):
    def __init__(self, grid_size=0.001, **kwargs):
        super().__init__()
        self.grid_size = grid_size
        self.model = sonata.model.load("facebook/sonata")

    def forward(
        self,
        coord: torch.Tensor,
        grid_coord: torch.Tensor,
        color: torch.Tensor,
        inverse: torch.Tensor,
        offset: torch.Tensor,
        feat: torch.Tensor,
        **kwargs
    ):
        out_point = self.model(
            coord=coord,
            grid_coord=grid_coord,
            color=color,
            inverse=inverse,
            offset=offset,
            feat=feat,
        )
        for _ in range(2):
            assert "pooling_parent" in out_point.keys()
            assert "pooling_inverse" in out_point.keys()
            parent = out_point.pop("pooling_parent")
            inverse = out_point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, out_point.feat[inverse]], dim=-1)
            out_point = parent
        while "pooling_parent" in out_point.keys():
            assert "pooling_inverse" in out_point.keys()
            parent = out_point.pop("pooling_parent")
            inverse = out_point.pop("pooling_inverse")
            parent.feat = out_point.feat[inverse]
            out_point = parent

        feat = out_point.feat[out_point.inverse]  # (n_points, embed_dim)
        # TODO: figure out how to batch?
        # TODO: somehow remap to image frame and return with corresponding chunks? We should max pool within chunks

    def create_processor(self):
        config = [
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=self.grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color", "inverse"),
                feat_keys=("coord", "color", "normal"),
            ),
        ]
        transform = sonata.transform.Compose(config)
        return transform

    @property
    def embed_dim(self):
        return 1088
