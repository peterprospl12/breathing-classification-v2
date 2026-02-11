from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class BreathPhaseCNN(nn.Module):
    """
    Pure CNN model for per-frame breathing phase classification.

    Uses 2D convolutions to extract spectral features from the mel-spectrogram,
    pooling only along the frequency axis to preserve temporal resolution.
    Then applies 1D convolutions along the time axis for temporal modeling
    before per-frame classification.

    Input:  [batch_size, 1, n_mels, time_frames]  (mel-spectrogram)
    Output: [batch_size, time_frames, num_classes]
    """

    def __init__(self,
                 n_mels: int = 128,
                 num_classes: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        self.n_mels = n_mels

        # 2D CNN feature extractor - pools only on frequency axis
        self.feature_extractor = nn.Sequential(
            # Block 1: [B, 1, 128, T] -> [B, 32, 64, T]
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Block 2: [B, 32, 64, T] -> [B, 64, 32, T]
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Block 3: [B, 64, 32, T] -> [B, 128, 16, T]
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Block 4: [B, 128, 16, T] -> [B, 128, 8, T]
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

        # After 4x frequency pooling: n_mels // 16 = 128 // 16 = 8
        self.freq_out = n_mels // 16
        self.flat_features = 128 * self.freq_out  # 128 * 8 = 1024

        # 1D temporal convolutions - operate along the time axis
        self.temporal_conv = nn.Sequential(
            # [B, 1024, T] -> [B, 256, T]
            nn.Conv1d(self.flat_features, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            # [B, 256, T] -> [B, 128, T]
            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # [B, 128, T] -> [B, 64, T]
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Classification head
        self.classifier = nn.Conv1d(64, num_classes, kernel_size=1)

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
        batch_size = spectrogram_batch.size(0)
        time_frames = spectrogram_batch.size(-1)

        # 2D feature extraction: [B, 1, n_mels, T] -> [B, 128, freq_out, T]
        features_2d = self.feature_extractor(spectrogram_batch)

        # Flatten frequency dimension: [B, 128, freq_out, T] -> [B, 128*freq_out, T]
        features_flat = features_2d.view(batch_size, self.flat_features, -1)

        # 1D temporal convolutions: [B, flat_features, T] -> [B, 64, T]
        temporal_features = self.temporal_conv(features_flat)

        # Per-frame classification: [B, 64, T] -> [B, num_classes, T]
        logits = self.classifier(temporal_features)

        # Permute to [B, T, num_classes] and make contiguous for .view() in training
        logits = logits.permute(0, 2, 1).contiguous()

        return logits
