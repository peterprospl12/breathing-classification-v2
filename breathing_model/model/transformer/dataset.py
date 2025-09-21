import os
import random
from typing import Optional, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import torchaudio

from utils import BreathType  


class BreathDataset(Dataset):
    """
    Dataset for breathing sequences with on-the-fly augmentation.
    Augmentations: random Gaussian noise, random volume change, random time-shift.
    Augmentations preserve mel-spectrogram output sizes.
    """

    def __init__(
        self,
        data_dir: str,
        label_dir: str,
        sample_rate: int = 44100,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        transforms: Optional[Iterable] = None,
        # Augmentation params:
        augment: bool = True,
        p_noise: float = 0.2,
        p_volume: float = 0.2,
        p_shift: float = 0.2,
        noise_factor_range: Tuple[float, float] = (1e-5, 5e-4),
        volume_range: Tuple[float, float] = (0.5, 1.2),
        max_shift_seconds: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.transforms = transforms

        self.augment = augment
        self.p_noise = p_noise
        self.p_volume = p_volume
        self.p_shift = p_shift
        self.noise_factor_range = noise_factor_range
        self.volume_range = volume_range
        self.max_shift_seconds = max_shift_seconds

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype='power')

        # Find all .wav files
        self.wav_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.wav')])
        if len(self.wav_files) == 0:
            raise ValueError(f"No .wav files found in {data_dir}")

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        wav_name = self.wav_files[idx]
        base = os.path.splitext(wav_name)[0]
        wav_path = os.path.join(self.data_dir, wav_name)
        csv_path = os.path.join(self.label_dir, f'{base}.csv')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing label file: {csv_path}")

        # Load audio
        waveform, sr = torchaudio.load(wav_path)  # shape [channels, samples]

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # [1, samples]

        # Apply augmentation (on waveform). Returns (waveform, shift_frames)
        shift_frames = 0
        if self.augment:
            waveform, shift_frames = self._maybe_augment_waveform(waveform)

        # Compute mel-spectrogram (keeps shapes consistent)
        mel = self.mel_transform(waveform)  # (1, n_mels, T)
        mel = self.db_transform(mel)

        # Apply optional mel transforms (e.g., normalization, SpecAugment later)
        if self.transforms:
            for transform in self.transforms:
                mel = transform(mel)

        # Parse labels aligned to mel frames (before applying shift fix)
        num_frames = mel.shape[-1]
        labels = self.__parse_intervals(csv_path, num_frames=num_frames)  # (num_frames,)

        # If we applied time-shift, roll labels accordingly and fill rolled-in area with SILENCE
        if shift_frames != 0:
            # If shift_frames magnitude >= num_frames => make all silence
            if abs(shift_frames) >= num_frames:
                labels = torch.full((num_frames,), int(BreathType.SILENCE), dtype=torch.int64)
            else:
                labels = torch.roll(labels, shifts=shift_frames)
                if shift_frames > 0:
                    # content moved right; fill first shift_frames with SILENCE
                    labels[:shift_frames] = int(BreathType.SILENCE)
                else:
                    # shift_frames < 0: content moved left; fill last abs(shift_frames) with SILENCE
                    labels[shift_frames:] = int(BreathType.SILENCE)

        return mel, labels

    def _maybe_augment_waveform(self, waveform: Tensor) -> Tuple[Tensor, int]:
        """
        Apply zero or more augmentations (noise, volume, time-shift) to the input waveform.
        Returns (waveform, shift_frames) where shift_frames is integer number of spectrogram frames shifted.
        """
        # waveform: Tensor shape [1, samples], dtype float
        x = waveform.clone()
        shift_frames = 0

        # 1) Add noise
        if random.random() < self.p_noise:
            f = random.uniform(self.noise_factor_range[0], self.noise_factor_range[1])
            noise = torch.randn_like(x) * f
            x = x + noise
            # avoid overflow: clamp to [-1, 1] (common for audio floats)
            x = torch.clamp(x, -1.0, 1.0)

        # 2) Random volume
        if random.random() < self.p_volume:
            factor = random.uniform(self.volume_range[0], self.volume_range[1])
            x = x * factor
            x = torch.clamp(x, -1.0, 1.0)

        # 3) Time shift (circular roll + zero-fill at the wrapped area to avoid wrap-around artifacts)
        if random.random() < self.p_shift:
            max_shift_samples = int(self.max_shift_seconds * self.sample_rate)
            if max_shift_samples > 0:
                shift_samples = random.randint(-max_shift_samples, max_shift_samples)
                if shift_samples != 0:
                    x = torch.roll(x, shifts=shift_samples, dims=-1)
                    # zero out wrapped region
                    if shift_samples > 0:
                        x[..., :shift_samples] = 0.0
                    else:
                        x[..., shift_samples:] = 0.0
                    # compute equivalent frame shift (rounded)
                    shift_frames = int(np.round(shift_samples / float(self.hop_length)))

        return x, shift_frames

    def __parse_intervals(self, csv_path: str, num_frames: int) -> Tensor:
        """
        Parse CSV file and map labels to spectrogram frames.
        CSV columns expected: class, start_sample, end_sample
        Returns: (num_frames,) tensor of labels (dtype int64).
        """
        df = pd.read_csv(csv_path, header=0)
        labels = torch.full((num_frames,), int(BreathType.SILENCE), dtype=torch.int64)

        label_map = {
            "exhale": int(BreathType.EXHALE),
            "inhale": int(BreathType.INHALE),
            "silence": int(BreathType.SILENCE)
        }

        for _, row in df.iterrows():
            start_sample = int(row['start_sample'])
            end_sample = int(row['end_sample'])
            start_frame = start_sample // self.hop_length
            # ceil-like end_frame calculation:
            end_frame = (end_sample + self.hop_length - 1) // self.hop_length
            end_frame = min(end_frame, num_frames)

            if start_frame < num_frames:
                code = label_map.get(row['class'])
                if code is None:
                    raise ValueError(f"Unknown class: {row['class']}")
                labels[start_frame:end_frame] = code

        return labels


def collate_fn(batch):
    spectrograms, labels = zip(*batch)
    original_lengths = [spec.shape[-1] for spec in spectrograms]

    spectrograms_transposed = [spec.squeeze(0).permute(1, 0) for spec in spectrograms]
    spectrograms_padded = pad_sequence(spectrograms_transposed, batch_first=True, padding_value=0.0)
    spectrograms_padded = spectrograms_padded.permute(0, 2, 1)
    spectrograms_batch = spectrograms_padded.unsqueeze(1)

    labels_padded = pad_sequence(labels, batch_first=True, padding_value=int(BreathType.SILENCE))

    max_len = spectrograms_batch.shape[-1]
    padding_mask = torch.arange(max_len)[None, :] >= torch.tensor(original_lengths)[:, None]

    return spectrograms_batch, labels_padded, padding_mask



def analyze_label_distribution(dataset: 'BreathDataset', breath_type_class=None) -> dict:
    """
    Analyzes class distribution (breathing phases) in the entire dataset.

    Args:
        dataset: BreathDataset instance
        breath_type_class: Enum class with breath type values (e.g. BreathType.EXHALE = 0 etc.)
                         If None, integers are used as keys.

    Returns:
        Dict containing frame count and percentage for each class.
    """
    from collections import Counter

    if breath_type_class is None:
        # Domyślne nazwy, jeśli brak BreathType
        reverse_map = {0: "exhale", 1: "inhale", 2: "silence"}
    else:
        reverse_map = {
            breath_type_class.EXHALE: "exhale",
            breath_type_class.INHALE: "inhale",
            breath_type_class.SILENCE: "silence"
        }

    print("Analyzing label distribution in dataset...")
    all_labels = []

    for i in range(len(dataset)):
        _, labels = dataset[i]
        all_labels.extend(labels.tolist())

        if (i + 1) % 10 == 0:  # Informacja co 10 plików
            print(f"Processed {i + 1}/{len(dataset)} files")

    # Zlicz etykiety
    label_counts = Counter(all_labels)
    total_frames = sum(label_counts.values())

    results = {}
    print("\n" + "=" * 50)
    print("DISTRIBUTION IN DATASET CLASS")
    print("=" * 50)

    for label_code in sorted(label_counts.keys()):
        class_name = reverse_map.get(label_code, f"Unknown_{label_code}")
        count = label_counts[label_code]
        percentage = (count / total_frames) * 100
        results[class_name] = {"count": count, "percentage": round(percentage, 2)}

        print(f"{class_name:>10}: {count:>8} frames ({percentage:>6.2f}%)")

    print("-" * 50)
    print(f"{'TOTAL':>10}: {total_frames:>8} frames (100.00%)")

    # Ocena zrównoważenia
    percentages = [info["percentage"] for info in results.values()]
    max_diff = max(percentages) - min(percentages)

    print("\n" + "ASSESMENT OF BALANCE DISTRIBUTION:")
    if max_diff < 10:
        print("✅ Very well balanced!")
    elif max_diff < 20:
        print("⚠️  Moderately balanced")
    else:
        print("❌ Weakly balanced – may require augmentation or focal loss")

    return results


if __name__ == '__main__':
    deprecated_data_dir = "../../archive/data-sequences"
    deprecated_label_dir = "../../archive/data-sequences"
    data_dir = "../../data/train/raw"
    label_dir = "../../data/train/label"
    dataset = BreathDataset(data_dir,label_dir, augment=False)

    analyze_label_distribution(dataset)