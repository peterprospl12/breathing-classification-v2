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
        pe = torch.zeros(max_len, d_model)  # positional encoding
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


class BreathPhaseTransformerSeq(nn.Module):
    """
    Transformer model that classifies each spectrogram frame into two classes:
    0 = exhale, 1 = inhale, 2 = silence.

    Forward accepts src_key_padding_mask with shape [batch_size, time_frames] (bool),
    where True indicates a padded frame that should be ignored by attention.
    """
    def __init__(self,
                 d_model: int = 192,
                 nhead: int = 8,
                 num_layers: int = 6,
                 num_classes: int = 3):
        super().__init__()

        self.conv_layers = self._build_cnn_layers()

        self.cnn_feature_dim = 128

        self.feature_projection = nn.Linear(self.cnn_feature_dim, d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dropout=0.1,
                                                   batch_first=True)

        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        # Classification head
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def _build_cnn_layers(self):
        # Convolution only in time dimension (there are no dependencies between mfcc features) (1D)
        conv1 = nn.Sequential(
            nn.Conv1d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1),  # 20 MFCC jako kanały
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        conv3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        return nn.Sequential(conv1, conv2, conv3)

    def forward(self, spectrogram_batch: Tensor, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            spectrogram_batch: [batch_size, 1, n_mfcc=20, time_frames]
        """

        # 1) Delete channel dimension -> [batch_size, n_mfcc=20, time_frames]
        mfcc_features = spectrogram_batch.squeeze(1)  # [batch_size, 20, time_frames]

        # 2) CNN 1D on time dimension -> [batch_size, 128, time_frames]
        cnn_features = self.conv_layers(mfcc_features)

        # 3) Permute to transformer format [batch_size, time_frames, 128]
        time_major_features = cnn_features.permute(0, 2, 1)

        # 4) Projection to d_model -> [batch_size, time_frames, d_model]
        projected_features = self.feature_projection(time_major_features)  # 128 → 192

        # 5) Transformer Encoder
        encoded_features = self.pos_encoder(projected_features)
        transformer_output = self.transformer_encoder(encoded_features, src_key_padding_mask=src_key_padding_mask)
        dropout_output = self.dropout(transformer_output)
        logits = self.head(dropout_output)

        return logits


class BreathPhaseTransformerSeqNew(nn.Module):
    def __init__(self, d_model: int = 192, nhead: int = 8, num_layers: int = 6,
                 num_classes: int = 3, use_delta: bool = True):
        super().__init__()

        self.input_channels = 60 if use_delta else 20

        # SPECTRAL relationships (Conv2D)
        self.spectral_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))  # [B, 32, 1, T]
        )

        # TEMPORAL patterns (Conv1D)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=15, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.feature_projection = nn.Linear(128, d_model)

        # Transformer (bez zmian)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dropout=0.1, batch_first=True)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.dropout = nn.Dropout(0.3)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, spectrogram_batch: Tensor, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        # 1) Conv2D: relacje między MFCC features
        spectral_features = self.spectral_conv(spectrogram_batch)  # [B, 32, 1, T]
        spectral_features = spectral_features.squeeze(2)  # [B, 32, T]

        # 2) Conv1D: temporal patterns
        temporal_features = self.temporal_conv(spectral_features)  # [B, 128, T]
        temporal_features = temporal_features.permute(0, 2, 1)  # [B, T, 128]

        # 3) Projection do d_model
        projected = self.feature_projection(temporal_features)  # [B, T, d_model]

        # 4) Transformer
        encoded = self.pos_encoder(projected)
        output = self.transformer_encoder(encoded, src_key_padding_mask=src_key_padding_mask)
        output = self.dropout(output)

        return self.head(output)
