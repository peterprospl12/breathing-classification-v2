import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Adding sinusoidal positional encoding to the input sequence.
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


class BreathPhaseTransformerSeq(nn.Module):
    """
    Transformer model for breath phase classification at spectrogram frame level.
    Input: [batch_size, channels=1, n_mels, time_frames]
    Output: [batch_size, time_frames, num_classes] (logits for each frame)
    """

    def __init__(self,
                 n_mels: int = 128,
                 d_model: int = 192,
                 nhead: int = 8,
                 num_layers: int = 6,
                 num_classes: int = 2):
        super().__init__()

        self.conv_layers = self._build_cnn_layers()

        self.out_freq = n_mels // 8  # after three pooling operations (vertical only)
        self.cnn_feature_dim = 128 * self.out_freq

        self.feature_projection = nn.Linear(self.cnn_feature_dim, d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dropout=0.1,
                                                   batch_first=True)

        self.transformer_layer = nn.Sequential(
            PositionalEncoding(d_model=d_model, dropout=0.1),
            nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        )

        # Classification head
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def _build_cnn_layers(self):
        conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )

        conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )

        conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )

        return nn.Sequential(conv1, conv2, conv3)

    def forward(self, spectrogram):
        # spectrogram: [batch_size, channels=1, n_mels, time_frames]

        # cnn_features: [batch_size, channels=128, n_mels//8, time_frames]
        cnn_features = self.conv_layers(spectrogram)

        # time_major_features: [batch_size, time_frames, channels, freq_bins]
        time_major_features = cnn_features.permute(0, 3, 1, 2)

        batch_size, time_frames, channels, freq_bins = time_major_features.size()
        flattened_features = time_major_features.contiguous().view(batch_size,
                                                                   time_frames,
                                                                   channels * freq_bins)

        projected_features = self.feature_projection(flattened_features)  # [batch_size, time_frames, d_model]
        transformer_output = self.transformer_layer(projected_features)  # [batch_size, time_frames, d_model]
        dropout_output = self.dropout(transformer_output)
        logits = self.head(dropout_output)  # [batch_size, time_frames, num_classes]

        return logits
