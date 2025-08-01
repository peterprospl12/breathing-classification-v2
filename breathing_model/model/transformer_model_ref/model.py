import torch
import torch.nn as nn
import math



class PositionalEncoding(nn.Module):
    """
    Dodaje sinusoidalne kodowanie pozycyjne do sekwencji wejściowej.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class BreathPhaseTransformerSeq(nn.Module):
    """
    Model Transformerowy do klasyfikacji fazy oddechu na poziomie ramek spektrogramu.
    Wejście: [B, 1, n_mels, T]
    Wyjście: [B, T, num_classes] (logity dla każdej ramki)
    """
    def __init__(self,
                 n_mels: int = 128,
                 d_model: int = 192,
                 nhead: int = 8,
                 num_layers: int = 6,
                 num_classes: int = 3):
        super().__init__()

        conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        )

        conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        )

        conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        )

        self.conv_layer = nn.Sequential(
            conv1,
            conv2,
            conv3
        )

        self.out_freq = n_mels // 8  # po trzech poolingach (tylko w pionie)
        self.cnn_feature_dim = 128 * self.out_freq

        self.fc_proj = nn.Linear(self.cnn_feature_dim, d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.1, batch_first=True)

        self.transformer_layer = nn.Sequential(
            PositionalEncoding(d_model=d_model, dropout=0.1),
            nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        )

        # Klasyfikator
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):

        # x: [B, 1, n_mels, T]
        x = self.conv_layer(x)

        x = x.permute(0, 3, 1, 2)  # [B, T, C, F]
        B, T, C, F = x.size()
        x = x.contiguous().view(B, T, C * F)  # [B, T, C*F]

        x = self.fc_proj(x)  # [B, T, d_model]
        x = self.transformer_layer(x)  # [B, T, d_model]
        x = self.dropout(x)
        logits = self.head(x)  # [B, T, num_classes]

        return logits
