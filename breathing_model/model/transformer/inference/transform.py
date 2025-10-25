import torch
import torchaudio
import numpy as np
from breathing_model.model.transformer.utils import DataConfig


class MelSpectrogramTransform:
    def __init__(self, config: DataConfig):
        self.config = config
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __call__(self, signal: np.ndarray) -> torch.Tensor:
        if signal.ndim > 1:
            signal = signal.mean(axis=1)

        waveform = torch.tensor(signal).unsqueeze(0)
        mel = self.transform(waveform)
        mel = self.db_transform(mel)
        mel = mel.unsqueeze(0)
        return mel
