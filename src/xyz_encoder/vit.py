import torch
from torchvision.models import VisionTransformer

class ViTEncoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = VisionTransformer(**kwargs)

    def forward(self, xyz: torch.Tensor, **kwargs):
        """
        Expects (B, 3, H, W) input, returns (B, n_patches, hidden_dim) features
        """
        x = self.model._process_input(xyz)
        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(len(x), -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.model.encoder(x)
        x = x[:, 1:]  # Remove the class token
        return x

    def create_processor(self):
        return None

    @property
    def embed_dim(self):
        return self.model.hidden_dim
