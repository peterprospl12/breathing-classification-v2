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
    weight_decay: float
    save_dir: str


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
class SchedulerConfig:
    type: str
    max_lr: float = 3e-4
    pct_start: float = 0.3
    anneal_strategy: str = "cos"
    div_factor: int = 25
    final_div_factor: int = 10000
    step_size: int = 5
    gamma: float = 0.5

@dataclass
class AugmentConfig:
    enabled: bool
    p_noise: float
    p_volume: float
    p_shift: float
    volume_range: tuple[float, float]
    noise_factor_range: tuple[float, float]
    max_shift_seconds: float
    seed: int


@dataclass
class Config:
    data: DataConfig
    audio: AudioConfig
    train: TrainConfig
    model: ModelConfig
    plot: PlotConfig
    scheduler: SchedulerConfig
    augment: AugmentConfig

    @classmethod
    def from_yaml(cls, path: str):
        raw_data = load_yaml(path)

        data = DataConfig(**raw_data['data'])
        audio = AudioConfig(**raw_data['audio'])
        train = TrainConfig(**raw_data['train'])
        model = ModelConfig(**raw_data['model'])
        plot = PlotConfig(**raw_data['plot'])
        scheduler = SchedulerConfig(**raw_data['scheduler'])
        augment = AugmentConfig(**raw_data['augment'])

        return cls(
            data=data,
            audio=audio,
            train=train,
            model=model,
            plot=plot,
            scheduler=scheduler,
            augment=augment
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'data': asdict(self.data),
            'audio': asdict(self.audio),
            'train': asdict(self.train),
            'model': asdict(self.model),
            'plot': asdict(self.plot),
            'scheduler': asdict(self.scheduler),
            'augment': asdict(self.augment),
        }

class BreathType(IntEnum):
    EXHALE = 0
    OTHER = 1

    def get_label(self):
        labels = {
            self.EXHALE: "exhale",
            self.OTHER: "other"
        }
        return labels[self]

    def get_color(self):
        colors = {
            self.EXHALE: 'green',
            self.OTHER: 'blue'
        }
        return colors[self]
