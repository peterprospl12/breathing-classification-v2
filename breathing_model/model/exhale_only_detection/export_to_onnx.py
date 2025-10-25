import torch
import torch.onnx
from breathing_model.model.exhale_only_detection.model import BreathPhaseTransformerSeq
from breathing_model.model.exhale_only_detection.utils import Config, DataConfig
from breathing_model.model.transformer.inference.transform import MelSpectrogramTransform


class AudioToBreathClassifier(torch.nn.Module):
    """Wrapper model that includes MelSpectrogram preprocessing."""
    def __init__(self, classifier_model, config: DataConfig):
        super().__init__()

        self.mel_transform = MelSpectrogramTransform(config)
        self.classifier = classifier_model

    def forward(self, audio_signal):
        """
        audio_signal: [batch, audio_length] - raw audio, float32 in [-1, 1]
        Returns: [batch, time_frames, num_classes] - logits
        """
        mel_spec = self.mel_transform(audio_signal)
        return self.classifier(mel_spec)


def export_breath_classifier_to_onnx(model_path, onnx_path, audio_length=154350):
    print("Exporting breath classifier to ONNX...")

    config = Config.from_yaml('./config.yaml')

    # Load model transformer
    model = BreathPhaseTransformerSeq(
        n_mels=config.model.n_mels,
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_layers=config.model.num_layers,
        num_classes=config.model.num_classes
    )

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Wrap model with preprocessing
    full_model = AudioToBreathClassifier(model, config.data)
    full_model.eval()

    # Dummy input: [batch=1, audio_length]
    dummy_input = torch.randn(audio_length, dtype=torch.float32)

    input_names = ["audio_input"]
    output_names = ["logits"]

    torch.onnx.export(
        full_model,
        dummy_input,
        onnx_path,
        export_params=True,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        opset_version=18,
        verbose=False,
    )

    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"Breath classifier model exported and verified: {onnx_path}")


if __name__ == "__main__":
    # Model paths - using the model from realtime inference
    model_path = "best_models/best_model_epoch_21.pth"
    onnx_path = "best_models/best_model_epoch_21.onnx"

    # Export model
    export_breath_classifier_to_onnx(model_path, onnx_path)

    print("ONNX export complete.")
