import torchaudio
import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from typing import Iterable, Optional
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from utils import BreathType


class BreathDataset(Dataset):
    """
    Each element is a MFCC and a sequence of frame-level labels (0: exhale, 1: inhale, 2: silence).
    Sequences are kept at their original length; padding is applied during batching via collate_fn.
    """

    def __init__(
        self,
        data_dir: str,
        label_dir: str,
        sample_rate: int = 44100,
        n_mels: int = 128,
        n_mfcc: int = 20,
        n_fft: int = 2048,
        hop_length: int = 512,
        transforms: Optional[Iterable] = None
    ):
        """
        Args:
            data_dir: Path to a directory with .wav files containing audio (same base name as labels).
            label_dir: Path to a directory with .csv files containing labels (same base name as audio).
            sample_rate: Target sampling rate.
            n_mels: Number of mel filters.
            n_mfcc: Number of mfcc coefficients.
            n_fft: FFT window size (larger → better frequency resolution).
            hop_length: Hop length for STFT.
            transforms: Optional list of transforms applied to mel spectrogram.
        """
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.transforms = transforms

        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'n_mels': self.n_mels,
                'center': True,
                'power': 2.0
            }
        )

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
        waveform, sr = torchaudio.load(wav_path)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        mfcc = self.mfcc_transform(waveform)  # [1, n_mfcc, T]

        # Apply optional transforms
        if self.transforms:
            for transform in self.transforms:
                mfcc = transform(mfcc)

        # Parse labels
        labels = self.__parse_intervals(csv_path, num_frames=mfcc.shape[-1])

        return mfcc, labels

    def __parse_intervals(self, csv_path: str, num_frames: int) -> Tensor:
        """
        Parse CSV file and map labels to spectrogram frames.
        CSV columns: class, start_sample, end_sample
        Returns: (num_frames,) tensor of labels.
        """
        df = pd.read_csv(csv_path, header=0)
        labels = torch.full((num_frames,), BreathType.OTHER, dtype=torch.int64)  # Default: OTHER (0)

        label_map = {"exhale": BreathType.EXHALE,
                     "inhale": BreathType.OTHER,
                     "silence": BreathType.OTHER}

        for _, row in df.iterrows():
            start_frame = int(row['start_sample']) // self.hop_length
            end_frame = (int(row['end_sample']) + self.hop_length - 1) // self.hop_length
            end_frame = min(end_frame, num_frames)

            if start_frame < num_frames:
                code = label_map.get(row['class'])
                if code is None:
                    raise ValueError(f"Unknown class: {row['class']}")
                labels[start_frame:end_frame] = code

        return labels


def collate_fn(batch):
    """
    Collate function for DataLoader to handle variable-length sequences.
    Pads both mel-spectrograms and labels to the same length (longest in batch).
    Returns:
        spectrograms_batch: [batch_size, 1, n_mfcc, time_frames_padded]
        labels_padded: [batch_size, time_frames_padded]
        padding_mask: [batch_size, time_frames_padded] (bool) True where padding
    """
    spectrograms, labels = zip(*batch)

    # Remember original lengths before padding
    original_lengths = [spec.shape[-1] for spec in spectrograms]

    # Convert MFCC from shape (1, n_mfcc, T) -> (T, n_mfcc) for pad_sequence,
    # then pad and permute back to [B, 1, n_mfcc, T_max], where T_max is the padded (max) length in the batch
    spectrograms_transposed = [spectrogram.squeeze(0).permute(1, 0) for spectrogram in spectrograms]
    spectrograms_padded = pad_sequence(spectrograms_transposed, batch_first=True, padding_value=0.0)  # [B, T_max, n_mfcc]
    spectrograms_padded = spectrograms_padded.permute(0, 2, 1)  # [B, n_mfcc, T_max]
    spectrograms_batch = spectrograms_padded.unsqueeze(1)       # [B, 1, n_mfcc, T_max]

    # Pad labels with OTHER_LABEL
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=float(BreathType.OTHER))  # [B, T_max]

    # Build padding mask (True where silence/pad)
    max_len = spectrograms_batch.shape[-1]
    padding_mask = torch.arange(max_len)[None, :] >= torch.tensor(original_lengths)[:, None] # [B, T_max]

    return spectrograms_batch, labels_padded, padding_mask

if __name__ == '__main__':
    data_dir = "../../data/train/raw"
    label_dir = "../../data/train/label"
    dataset = iter(BreathDataset(data_dir,label_dir))

    for item, label in dataset:
        pass
