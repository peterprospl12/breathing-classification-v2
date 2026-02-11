from typing import Optional

import torch
import torch.nn as nn
import math

from torch import Tensor


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding added to the token embeddings.
    Input: embeddings tensor of shape [batch_size, sequence_length, d_model]
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model) # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('pe', pe)

    def forward(self, input_embeddings):
        # input_embeddings: [batch_size, sequence_length, d_model]
        seq_len = input_embeddings.size(1)
        encoded_input = input_embeddings + self.pe[:, :seq_len, :]
        return self.dropout(encoded_input)


class ConvPositionalEncoding(nn.Module):
    """
    Convolutional positional encoding using depthwise separable convolution.
    Provides relative position information — better than sinusoidal for
    variable-length sequences and audio tasks.
    """
    def __init__(self, d_model: int, kernel_size: int = 31):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,  # depthwise
        )
        nn.init.normal_(self.conv.weight, 0, 0.01)
        nn.init.zeros_(self.conv.bias)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, d_model]
        residual = x
        x = x.permute(0, 2, 1)  # [B, d_model, T]
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # [B, T, d_model]
        return self.norm(residual + x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: channel attention for 2D feature maps."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ConvBlock(nn.Module):
    """
    Residual CNN block with optional Squeeze-and-Excitation attention.
    Pools only along the frequency axis to preserve temporal resolution.
    Uses two convolutions per block for richer feature extraction.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 pool_freq: bool = True, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)) if pool_freq else nn.Identity()
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

        # Skip connection: project channels if dimensions change
        if in_channels != out_channels:
            self.skip_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip_proj = nn.Identity()
        self.skip_pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)) if pool_freq else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        skip = self.skip_pool(self.skip_proj(x))

        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.pool(out)

        return self.act(out + skip)


class BreathPhaseTransformerSeq(nn.Module):
    """
    Improved Transformer for per-frame breathing phase classification:
    0 = exhale, 1 = inhale, 2 = silence.

    Architecture improvements over baseline:
    - Deeper CNN frontend (4 blocks) with residual connections and SE attention
    - Pre-norm Transformer (norm_first=True) for stable training
    - GELU activation in Transformer feedforward layers
    - Properly sized dim_feedforward (4× d_model instead of default 2048)
    - Convolutional positional encoding for better position generalization
    - Post-projection LayerNorm for feature normalization
    - Richer classification head with hidden layer
    """
    def __init__(self,
                 n_mels: int = 128,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 num_classes: int = 3,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.15):
        super().__init__()

        # --- CNN Frontend (4 blocks, frequency-only pooling) ---
        self.conv_layers = nn.Sequential(
            ConvBlock(1, 32, pool_freq=True, use_se=False),
            ConvBlock(32, 64, pool_freq=True, use_se=True),
            ConvBlock(64, 128, pool_freq=True, use_se=True),
            ConvBlock(128, 256, pool_freq=True, use_se=True),
        )

        # After 4× freq pooling: n_mels // 16
        self.out_freq = n_mels // 16
        self.cnn_feature_dim = 256 * self.out_freq

        # --- Feature Projection with normalization ---
        self.feature_projection = nn.Sequential(
            nn.Linear(self.cnn_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- Convolutional Positional Encoding ---
        self.pos_encoder = ConvPositionalEncoding(d_model=d_model, kernel_size=31)

        # --- Pre-Norm Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN: more stable training
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),  # final normalization
        )

        # --- Classification Head ---
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self,
                spectrogram_batch: Tensor,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            spectrogram_batch: [batch_size, channels=1, n_mels, time_frames]
            src_key_padding_mask: Optional[Tensor] [batch_size, time_frames] (bool),
                                  where True = padded frame to ignore.
        Returns:
            logits: [batch_size, time_frames, num_classes]
        """
        # 1) CNN feature extraction -> [B, 256, out_freq, T]
        cnn_features = self.conv_layers(spectrogram_batch)

        # 2) Reshape to time-major and flatten freq+channel
        time_major = cnn_features.permute(0, 3, 1, 2)  # [B, T, 256, out_freq]
        B, T, C, F = time_major.size()
        flat = time_major.contiguous().view(B, T, C * F)  # [B, T, cnn_feature_dim]

        # 3) Project to d_model
        projected = self.feature_projection(flat)  # [B, T, d_model]

        # 4) Convolutional positional encoding
        encoded = self.pos_encoder(projected)  # [B, T, d_model]

        # 5) Transformer encoder
        transformer_out = self.transformer_encoder(
            encoded, src_key_padding_mask=src_key_padding_mask
        )  # [B, T, d_model]

        # 6) Per-frame classification
        logits = self.head(transformer_out)  # [B, T, num_classes]

        return logits
