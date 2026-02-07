from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class BreathPhaseFeedForward(nn.Module):
    """
    Simple feed-forward (MLP) model for per-frame breathing phase classification.

    Uses a small temporal context window around each frame to provide
    some neighboring information. Each frame (with context) is classified
    independently through fully-connected layers.

    Input:  [batch_size, 1, n_mels, time_frames]  (mel-spectrogram)
    Output: [batch_size, time_frames, num_classes]
    """

    def __init__(self,
                 n_mels: int = 128,
                 context_frames: int = 5,
                 hidden_dim: int = 256,
                 num_classes: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        self.n_mels = n_mels
        self.context_frames = context_frames
        self.window_size = 2 * context_frames + 1  # total frames per window
        input_dim = n_mels * self.window_size

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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

        # Pad temporally so we can gather context windows for every frame
        # Pad context_frames on each side
        # x shape: [B, T, n_mels] -> pad along T dimension
        x_padded = torch.nn.functional.pad(x, (0, 0, self.context_frames, self.context_frames), mode='constant', value=0.0)
        # x_padded: [B, T + 2*context_frames, n_mels]

        # Build context windows using unfold
        # unfold(dim, size, step) on dim=1
        windows = x_padded.unfold(1, self.window_size, 1)
        # windows: [B, T, n_mels, window_size]

        # Flatten the last two dims to get feature vector per frame
        windows = windows.contiguous().view(batch_size, time_frames, -1)
        # windows: [B, T, n_mels * window_size]

        # Reshape for BatchNorm1d: [B*T, features]
        flat = windows.view(batch_size * time_frames, -1)

        # MLP forward
        logits_flat = self.mlp(flat)  # [B*T, num_classes]

        # Reshape back: [B, T, num_classes]
        logits = logits_flat.view(batch_size, time_frames, -1)

        return logits
