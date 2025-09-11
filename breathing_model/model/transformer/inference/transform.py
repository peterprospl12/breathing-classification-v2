import torch
import torchaudio
import numpy as np
from breathing_model.model.transformer.utils import DataConfig


class MelSpectrogramTransform:
    def __init__(self, config: DataConfig):
        self.config = config
        self.transform = torchaudio.transforms.MFCC(
            sample_rate=config.sample_rate,
            n_mfcc=config.n_mels,
            melkwargs={
                'n_fft': config.n_fft,
                'hop_length': config.hop_length,
                'n_mels': config.n_mels,
                'center': True,
                'power': 2.0
            }
        )

    def __call__(self, signal: np.ndarray) -> torch.Tensor:
        if signal.ndim > 1:  # stereo -> mono
            signal = signal.mean(axis=1)
        waveform = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)  # [1, samples]
        mfcc = self.transform(waveform)  # [1, n_mfcc, T]
        mfcc = mfcc.unsqueeze(0)  # [1, 1, n_mfcc, T]
        return mfcc
