from typing import Optional, Tuple

import torch
import numpy as np

from breathing_model.model.feed_forward.model import BreathPhaseFeedForward
from breathing_model.model.feed_forward.utils import FFModelConfig
from breathing_model.model.transformer.utils import DataConfig


class BreathPhaseClassifierFF:
    def __init__(self,
                 model_path: str,
                 model_config: FFModelConfig,
                 data_config: DataConfig,
                 device: Optional[torch.device] = None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_config = model_config
        self.model = self._load_model(model_path, model_config)
        self.last_mel_frames = int(0.2 * data_config.sample_rate / data_config.hop_length)

    def _load_model(self, model_path: str, config: FFModelConfig) -> BreathPhaseFeedForward:
        model = BreathPhaseFeedForward(
            n_mels=config.n_mels,
            context_frames=config.context_frames,
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
            dropout=config.dropout,
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    @torch.no_grad()
    def predict(self,
                mel_tensor: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[int, np.ndarray]:
        """
        Args:
            mel_tensor: [1, 1, n_mels, T]
            src_key_padding_mask: optional, ignored (API compatibility)
        Returns:
            (predicted_class, mean_probs)
        """
        mel_tensor = mel_tensor.to(self.device)

        logits = self.model(mel_tensor)  # [1, T, C]
        probs = torch.softmax(logits, dim=-1)  # [1, T, C]
        probs_np = probs.squeeze(0).cpu().numpy()  # [T, C]

        if probs_np.shape[0] == 0:
            num_classes = self.model_config.num_classes
            return 0, np.full((num_classes,), 1.0 / num_classes, dtype=np.float32)

        recent_probs = probs_np[-self.last_mel_frames:]
        mean_probs = recent_probs.mean(axis=0)
        predicted_class = int(np.argmax(mean_probs))
        return predicted_class, mean_probs
