from dataclasses import dataclass
from breathing_model.model.transformer.utils import (
    load_yaml,
    DataConfig,
    AudioConfig,
    TrainConfig,
    PlotConfig,
    SchedulerConfig,
    AugmentConfig,
    BreathType,
)


@dataclass
class FFModelConfig:
    num_classes: int
    n_mels: int
    context_frames: int
    hidden_dim: int
    dropout: float


@dataclass
class FFConfig:
    data: DataConfig
    audio: AudioConfig
    train: TrainConfig
    model: FFModelConfig
    plot: PlotConfig
    scheduler: SchedulerConfig
    augment: AugmentConfig

    @classmethod
    def from_yaml(cls, path: str):
        raw_data = load_yaml(path)

        return cls(
            data=DataConfig(**raw_data['data']),
            audio=AudioConfig(**raw_data['audio']),
            train=TrainConfig(**raw_data['train']),
            model=FFModelConfig(**raw_data['model']),
            plot=PlotConfig(**raw_data['plot']),
            scheduler=SchedulerConfig(**raw_data['scheduler']),
            augment=AugmentConfig(**raw_data['augment']),
        )
