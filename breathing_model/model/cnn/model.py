from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class BreathPhaseCNN(nn.Module):
    """
    Pure CNN model for per-frame breathing phase classification.

    Uses 2D convolutions to extract spectral features from the mel-spectrogram,
    pooling only along the frequency axis to preserve temporal resolution.
    Each frame is classified independently after feature extraction (no temporal
    context modeling).

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
        # Uses kernel_size=(3, 1) to avoid temporal context
        self.feature_extractor = nn.Sequential(
            # Block 1: [B, 1, 128, T] -> [B, 32, 64, T]
            nn.Conv2d(1, 32, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Block 2: [B, 32, 64, T] -> [B, 64, 32, T]
            nn.Conv2d(32, 64, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Block 3: [B, 64, 32, T] -> [B, 128, 16, T]
            nn.Conv2d(64, 128, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Block 4: [B, 128, 16, T] -> [B, 128, 8, T]
            nn.Conv2d(128, 128, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

        # After 4x frequency pooling: n_mels // 16 = 128 // 16 = 8
        self.freq_out = n_mels // 16
        self.flat_features = 128 * self.freq_out  # 128 * 8 = 1024

        # Per-frame classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
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
        batch_size = spectrogram_batch.size(0)
        time_frames = spectrogram_batch.size(-1)

        # 2D feature extraction: [B, 1, n_mels, T] -> [B, 128, freq_out, T]
        features_2d = self.feature_extractor(spectrogram_batch)

        # Reshape to [B, T, flat_features] for per-frame classification
        # features_2d: [B, 128, freq_out, T] -> permute to [B, T, 128, freq_out] -> flatten
        features = features_2d.permute(0, 3, 1, 2).contiguous()
        features = features.view(batch_size, time_frames, self.flat_features)

        # Reshape for classifier: [B*T, flat_features]
        features_flat = features.view(batch_size * time_frames, self.flat_features)

        # Per-frame classification: [B*T, flat_features] -> [B*T, num_classes]
        logits_flat = self.classifier(features_flat)

        # Reshape to [B, T, num_classes]
        logits = logits_flat.view(batch_size, time_frames, -1)

        return logits
