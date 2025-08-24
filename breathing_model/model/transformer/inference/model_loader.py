import torch
import numpy as np
from breathing_model.model.transformer.model import BreathPhaseTransformerSeq
from breathing_model.model.transformer.utils import ModelConfig


class BreathPhaseClassifier:
    def __init__(self,
                 model_path: str,
                 config: ModelConfig,
                 device=None):
        self.device = device if device else torch.device('cpu')
        self.config = config
        self.model = self._load_model(model_path, config)

    def _load_model(self, model_path, config: ModelConfig):
        model = BreathPhaseTransformerSeq(
            n_mels=config.n_mels,
            num_classes=config.num_classes,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers
        ).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model

    @torch.no_grad()
    def predict(self, mel_tensor: torch.Tensor):
        mel_tensor = mel_tensor.to(self.device)  #[1,1,n_mels,T]

        logits = self.model(mel_tensor)  #[1,T, 3]
        probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze(0)

        if probs.shape[0] == 0:
            predicted_class = 2
            class_probs = np.array([0.0, 0.0, 1.0])
        else:
            preds = probs.argmax(axis=1)
            predicted_class = int(np.bincount(preds).argmax())
            class_probs = probs.mean(axis=0)

        return predicted_class, class_probs
