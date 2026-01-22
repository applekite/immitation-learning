import os
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class CNN3D(nn.Module):
    """
    3D CNN Variational Encoder for spatio-temporal reasoning.
    Expects input of shape [B, C, T, H, W], e.g. C=3 (RGB), T=#frames.
    """
    def __init__(self, latent_dims: int, in_channels: int = 3):
        super().__init__()

        self.model_file = os.path.join('autoencoder/model', 'var_encoder3d_model.pth')

        # 3D convolutional encoder (downsamples spatially and temporally)
        self.encoder = nn.Sequential(
            # Downsample spatially by 2x; keep time
            nn.Conv3d(in_channels, 32, kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(inplace=True),

            # Downsample time and space
            nn.Conv3d(32, 64, kernel_size=3, stride=(2,2,2), padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=True),

            # Focus more on spatial compression
            nn.Conv3d(64, 128, kernel_size=3, stride=(1,2,2), padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=True),

            # Downsample time (again) and space
            nn.Conv3d(128, 256, kernel_size=3, stride=(2,2,2), padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=True),
        )

        # Collapse to a fixed-length representation regardless of input size
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # MLP head
        self.linear = nn.Sequential(
            nn.Flatten(),               # 256 * 1 * 1 * 1  -> 256
            nn.Linear(256, 1024),
            nn.LeakyReLU(inplace=True),
        )

        # VAE parameters
        self.mu = nn.Linear(1024, latent_dims)
        self.logvar = nn.Linear(1024, latent_dims)  # log(sigma^2)

        # Will store the KL term after forward()
        self.kl = torch.tensor(0.0, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns latent sample z. Also sets self.kl with the batch-summed KL term.
        """
        x = x.to(device, non_blocking=True)
        x = self.encoder(x)
        x = self.pool(x)
        x = self.linear(x)

        mu = self.mu(x)
        logvar = self.logvar(x)

        # Reparameterization: z = mu + std * eps
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps

        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        self.kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum())

        return z

    def save(self):
        torch.save(self.state_dict(), self.model_file)

    def load(self):
        self.load_state_dict(torch.load(self.model_file, map_location=device))
