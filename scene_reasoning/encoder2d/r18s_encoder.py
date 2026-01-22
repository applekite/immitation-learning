import os
import torch
import torch.nn as nn
from torchvision import models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResNet18Encoder(nn.Module):
    """
    Lightweight ResNet18-based encoder that maps an input image to a latent vector.
    Designed as a drop-in replacement for the previous VAE encoder in scene reasoning.
    """

    def __init__(self, latent_dims: int, pretrained: bool = True):
        super().__init__()
        self.latent_dims = latent_dims
        self.model_file = os.path.join(
            "scene_reasoning/encoder2d/model", "resnet18-5c106cde.pth"
        )

        # Load resnet18 backbone with backwards-compatible API
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        except AttributeError:
            # Older torchvision version
            self.backbone = models.resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features

        # Replace classification head with identity and projection to latent space
        self.backbone.fc = nn.Identity()
        self.projection = nn.Linear(in_features, latent_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get the device of the model (which may have been moved to GPU)
        model_device = next(self.parameters()).device
        x = x.to(model_device)
        features = self.backbone(x)
        latent = self.projection(features)
        return latent

    def save(self):
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
        torch.save(self.state_dict(), self.model_file)

    def load(self):
        if not os.path.exists(self.model_file):
            raise FileNotFoundError(
                f"ResNet18 encoder checkpoint not found: {self.model_file}"
            )
        # Get the device of the model (which may have been moved to GPU)
        model_device = next(self.parameters()).device
        # Load checkpoint to CPU first, then move tensors to the model's device
        # This avoids device mismatch errors
        state_dict = torch.load(self.model_file, map_location='cpu')
        
        # Handle old checkpoint format (keys without "backbone." prefix)
        # Check if we need to remap keys
        needs_remap = False
        for key in list(state_dict.keys()):
            if not key.startswith('backbone.') and key not in ['projection.weight', 'projection.bias']:
                needs_remap = True
                break
        
        if needs_remap:
            # Remap old format to new format
            remapped_dict = {}
            for key, value in state_dict.items():
                # Skip old FC layer weights (we replaced it with Identity + projection)
                if key in ['fc.weight', 'fc.bias']:
                    continue
                # Add "backbone." prefix to ResNet18 layers
                if key.startswith(('conv1.', 'bn1.', 'layer1.', 'layer2.', 'layer3.', 'layer4.', 'maxpool.')):
                    remapped_dict['backbone.' + key] = value
                # Keep projection layer as-is
                elif key.startswith('projection.'):
                    remapped_dict[key] = value
            state_dict = remapped_dict
        
        # Move all tensors in state_dict to the model's device
        state_dict = {k: v.to(model_device) for k, v in state_dict.items()}
        
        # Load with strict=False to allow missing keys (e.g., if projection dimension changed)
        self.load_state_dict(state_dict, strict=False)
