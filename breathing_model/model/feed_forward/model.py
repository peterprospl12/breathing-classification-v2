from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class BreathPhaseFeedForward(nn.Module):
    """
    Simple feed-forward (MLP) model for per-frame breathing phase classification.

    Each frame is classified independently through fully-connected layers
    using only its own mel-frequency features (no temporal context).

    Input:  [batch_size, 1, n_mels, time_frames]  (mel-spectrogram)
    Output: [batch_size, time_frames, num_classes]
    """

    def __init__(self,
                 n_mels: int = 128,
                 hidden_dim: int = 256,
                 num_classes: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        self.n_mels = n_mels

        self.mlp = nn.Sequential(
            nn.Linear(n_mels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self,
                spectrogram_batch: Tensor,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            spectrogram_batch: [batch_size, 1, n_mels, time_frames]
            src_key_padding_mask: Optional, ignored by this model but kept
                                  for API compatibility with transformer.
        Returns:
            logits: [batch_size, time_frames, num_classes]
        """
        # Remove channel dim: [B, n_mels, T]
        x = spectrogram_batch.squeeze(1)
        batch_size, n_mels, time_frames = x.size()

        # Permute to [B, T, n_mels]
        x = x.permute(0, 2, 1)

        # Reshape for BatchNorm1d: [B*T, n_mels]
        flat = x.reshape(batch_size * time_frames, n_mels)

        # MLP forward
        logits_flat = self.mlp(flat)  # [B*T, num_classes]

        # Reshape back: [B, T, num_classes]
        logits = logits_flat.view(batch_size, time_frames, -1)

        return logits
