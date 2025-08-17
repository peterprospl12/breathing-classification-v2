import yaml
from dataclasses import dataclass, asdict
from torch.utils.data import Dataset, random_split
from typing import Dict, Any
from enum import IntEnum


def load_yaml(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def split_dataset(dataset: Dataset) -> (Dataset, Dataset):
    num_samples = len(dataset)
    num_training_samples = int(num_samples * 0.8)
    num_val_samples = num_samples - num_training_samples

    train_data, val_data = random_split(dataset, [num_training_samples, num_val_samples])

    return train_data, val_data


@dataclass
class DataConfig:
    data_dir: str
    label_dir: str
    sample_rate: int
    n_mels: int
    hop_length: int
    n_fft: int


@dataclass
class AudioConfig:
    sample_rate: int
    channels: int
    chunk_length: float  # length in seconds of context gathered for inference
    device_index: int


@dataclass
class TrainConfig:
    batch_size: int
    learning_rate: float
    num_epochs: int
    patience: int


@dataclass
class ModelConfig:
    num_classes: int
    n_mels: int
    d_model: int
    nhead: int
    num_layers: int


@dataclass
class PlotConfig:
    history_seconds: int
    update_interval: float


@dataclass
class Config:
    data: DataConfig
    audio: AudioConfig
    train: TrainConfig
    model: ModelConfig
    plot: PlotConfig

    @classmethod
    def from_yaml(cls, path: str):
        raw_data = load_yaml(path)

        # Build each sub-config
        data = DataConfig(**raw_data['data'])
        audio = AudioConfig(**raw_data['audio'])
        train = TrainConfig(**raw_data['train'])
        model = ModelConfig(**raw_data['model'])
        plot = PlotConfig(**raw_data['plot'])

        return cls(data=data, audio=audio, train=train, model=model, plot=plot)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'data': asdict(self.data),
            'audio': asdict(self.audio),
            'train': asdict(self.train),
            'model': asdict(self.model),
        }


class BreathType(IntEnum):
    EXHALE = 0
    INHALE = 1
    SILENCE = 2

    def get_label(self):
        labels = {
            self.EXHALE: "exhale",
            self.INHALE: "inhale",
            self.SILENCE: "silence"
        }
        return labels[self]

    def get_color(self):
        colors = {
            self.EXHALE: 'green',
            self.INHALE: 'red',
            self.SILENCE: 'blue'
        }
        return colors[self]
