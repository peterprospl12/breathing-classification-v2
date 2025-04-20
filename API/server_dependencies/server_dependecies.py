import math
from enum import Enum

import numpy as np
import torch
import torchaudio

RATE = 44100           # sampling rate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BreathPhase(Enum):
    INHALE = 0
    EXHALE = 1
    SILENCE = 2

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class BreathPhaseTransformerSeq(torch.nn.Module):
    def __init__(self, n_mels=40, num_classes=3, d_model=128, nhead=4, num_transformer_layers=2):
        super(BreathPhaseTransformerSeq, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.out_freq = n_mels // 8
        cnn_feature_dim = 128 * self.out_freq

        self.fc_proj = torch.nn.Linear(cnn_feature_dim, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model,
                                                          nhead=nhead,
                                                          dropout=0.1,
                                                          batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=0.1)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc_out = torch.nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, 1, n_mels, time_steps)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        # shape: (batch, 128, out_freq, time_steps)
        x = x.permute(0, 3, 1, 2)  # (batch, time_steps, channels, out_freq)
        batch_size, time_steps, channels, freq = x.size()
        x = x.contiguous().view(batch_size, time_steps, channels * freq)  # (batch, time_steps, cnn_feature_dim)
        x = self.fc_proj(x)  # (batch, time_steps, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)  # (batch, time_steps, d_model)
        x = self.dropout(x)
        logits = self.fc_out(x)  # (batch, time_steps, num_classes)
        return logits

class MelTransformer:
    def __init__(self):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=RATE,
            n_fft=1024,
            hop_length=512,
            n_mels=40
        )

    def get_mel_transform(self, y):
        # y: int16 signal; convert to float32 in the range [-1, 1]
        y = y.astype(np.float32) / 32768.0
        # Ensure the signal is mono
        if y.ndim != 1:
            raise Exception("Otrzymano sygnał nie-mono!")
        # Convert to tensor (shape: [1, num_samples])
        waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        # Compute Mel spectrogram – result: [1, n_mels, time_steps]
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-9)
        # Add channel dimension – expected shape: (batch, 1, n_mels, time_steps)d
        mel_spec = mel_spec.unsqueeze(0)
        return mel_spec


#############################################
# Prediction class
#############################################
class RealTimeAudioClassifier:
    def __init__(self, model_path):
        self.model = BreathPhaseTransformerSeq(n_mels=40, num_classes=3, d_model=128, nhead=4,
                                               num_transformer_layers=2).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.mel_transformer = MelTransformer()

    def predict(self, y,dont_calc_mel = False, sr=RATE):
        # y: int16 signal; convert to float32 in the range [-1, 1]

        if dont_calc_mel:
            mel_spec = y
        else:
            mel_spec = self.mel_transformer.get_mel_transform(y, sr)

        mel_spec = mel_spec.to(device)
        with torch.no_grad():
            logits = self.model(mel_spec)  # shape: (1, time_steps, num_classes)
            probabilities = torch.softmax(logits, dim=2)
            probs_np = probabilities.squeeze(0).cpu().numpy()  # (time_steps, num_classes)
            # Aggregate predictions by frames – choose the most frequent class
            preds = np.argmax(probs_np, axis=1)
            predicted_class = int(np.bincount(preds).argmax())
        return predicted_class, BreathPhase(predicted_class).name, probs_np.max()
