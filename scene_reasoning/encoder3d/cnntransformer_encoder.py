import os
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class CNNTransformer(nn.Module):
    """
    CNN + Transformer Variational Encoder for global spatio-temporal reasoning.
    Input:  [B, C, T, H, W]
    Output: z ~ N(mu, sigma^2), with self.kl storing the summed KL term.
    """
    def __init__(
        self,
        latent_dims: int,
        in_channels: int = 3,
        stem_channels: int = 256,
        embed_dim: int = 256,
        t_bins: int = 8,
        h_bins: int = 7,
        w_bins: int = 7,
        n_heads: int = 8,
        depth: int = 6,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.model_file = os.path.join('autoencoder/model', 'var_encoder_cnntr_model.pth')

        # ---- 3D CNN stem (local spatio-temporal encoding) ----
        # Keeps time with first block; later blocks may compress it slightly if desired.
        # We rely on AdaptiveAvgPool3d to produce fixed (t_bins, h_bins, w_bins) later.
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=True),

            nn.Conv3d(128, stem_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            nn.BatchNorm3d(stem_channels),
            nn.LeakyReLU(inplace=True),
        )

        # Fix token grid regardless of input size
        self.pool = nn.AdaptiveAvgPool3d((t_bins, h_bins, w_bins))

        # Project stem features to Transformer embedding dim
        self.proj = nn.Conv3d(stem_channels, embed_dim, kernel_size=1)

        # ---- Positional encodings ----
        # Learnable factorized space/time embeddings: broadcast and sum
        s_tokens = h_bins * w_bins
        self.pos_space = nn.Parameter(torch.zeros(1, 1, s_tokens, embed_dim))   # [1, 1, S, D]
        self.pos_time  = nn.Parameter(torch.zeros(1, t_bins, 1, embed_dim))     # [1, T, 1, D]

        # [CLS] token for global aggregation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ---- Transformer encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_mult * embed_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,   # [B, N, D]
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm_out = nn.LayerNorm(embed_dim)

        # ---- VAE head ----
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.LeakyReLU(inplace=True),
        )
        self.mu = nn.Linear(1024, latent_dims)
        self.logvar = nn.Linear(1024, latent_dims)

        self.kl = torch.tensor(0.0, device=device)

        # Parameter init
        self._init_parameters()

    def _init_parameters(self):
        nn.init.trunc_normal_(self.pos_space, std=0.02)
        nn.init.trunc_normal_(self.pos_time, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, (nn.BatchNorm3d, nn.LayerNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T, H, W]
        returns: z [B, latent_dims]; sets self.kl
        """
        x = x.to(device, non_blocking=True)

        # CNN stem
        feats = self.stem(x)                       # [B, C', T', H', W']
        feats = self.pool(feats)                   # [B, C', T_bins, H_bins, W_bins]
        feats = self.proj(feats)                   # [B, D, T_bins, H_bins, W_bins]

        # To tokens
        B, D, T, H, W = feats.shape
        S = H * W
        feats = feats.permute(0, 2, 3, 4, 1).contiguous()  # [B, T, H, W, D]
        feats = feats.view(B, T, S, D)                     # [B, T, S, D]

        # Add factorized positional encodings
        pos = self.pos_space + self.pos_time               # [1, T, S, D] via broadcast
        tokens = feats + pos                               # [B, T, S, D]

        # Flatten (time, space) -> sequence
        tokens = tokens.view(B, T * S, D)                  # [B, N, D], N = T*S

        # Prepend CLS
        cls = self.cls_token.expand(B, -1, -1)             # [B, 1, D]
        seq = torch.cat([cls, tokens], dim=1)              # [B, 1+N, D]

        # Transformer
        y = self.transformer(seq)                          # [B, 1+N, D]
        y = self.norm_out(y)
        cls_out = y[:, 0, :]                               # [B, D]

        # VAE head
        h = self.mlp(cls_out)
        mu = self.mu(h)
        logvar = self.logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps

        # KL divergence (summed over batch & dims)
        self.kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum())

        return z

    def save(self):
        torch.save(self.state_dict(), self.model_file)

    def load(self):
        self.load_state_dict(torch.load(self.model_file, map_location=device))
