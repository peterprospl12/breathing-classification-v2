import torch
import torchaudio
import numpy as np
from breathing_model.model.exhale_only_detection.utils import DataConfig


class MelSpectrogramTransform:
    def __init__(self, config: DataConfig):
        self.config = config
        self.transform = torchaudio.transforms.MFCC(
            sample_rate=config.sample_rate,
            n_mfcc=config.n_mfcc,
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


def compute_delta_features(mfcc: torch.Tensor) -> torch.Tensor:
    # Delta (tempo zmian)
    delta = torch.diff(mfcc, dim=-1, prepend=mfcc[:, :, :1])
    # Delta-Delta (przyspieszenie)
    delta_delta = torch.diff(delta, dim=-1, prepend=delta[:, :, :1])
    return torch.cat([mfcc, delta, delta_delta], dim=1)


class MelSpectrogramTransformNew:
    def __init__(self, config: DataConfig, use_delta: bool = True):
        self.config = config
        self.use_delta = use_delta
        self.transform = torchaudio.transforms.MFCC(
            sample_rate=config.sample_rate,
            n_mfcc=config.n_mfcc,
            melkwargs={
                'n_fft': config.n_fft,
                'hop_length': config.hop_length,
                'n_mels': config.n_mels,
                'center': True,
                'power': 2.0
            }
        )

    def __call__(self, signal: np.ndarray) -> torch.Tensor:
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        waveform = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        mfcc = self.transform(waveform)  # [1, 20, T]

        if self.use_delta:
            mfcc = compute_delta_features(mfcc)  # [1, 60, T]

        return mfcc.unsqueeze(0)  # [1, 1, 60, T]
