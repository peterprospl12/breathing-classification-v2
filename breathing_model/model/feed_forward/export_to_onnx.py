import torch
import torch.onnx
from breathing_model.model.feed_forward.model import BreathPhaseFeedForward
from breathing_model.model.transformer.utils import load_yaml, DataConfig
from breathing_model.model.transformer.inference.transform import MelSpectrogramTransform


class AudioToBreathClassifierFF(torch.nn.Module):
    """Wrapper model that includes MelSpectrogram preprocessing for Feed-Forward model."""
    def __init__(self, classifier_model, data_config):
        super().__init__()
        self.mel_transform = MelSpectrogramTransform(data_config)
        self.classifier = classifier_model

    def forward(self, audio_signal):
        """
        audio_signal: [audio_length] - raw audio, float32 in [-1, 1]
        Returns: [1, time_frames, num_classes] - logits
        """
        mel = self.mel_transform(audio_signal)
        return self.classifier(mel)


def export_breath_classifier_to_onnx(model_path, onnx_path, config_path='./config.yaml', audio_length=154350):
    print("Exporting Feed-Forward breath classifier to ONNX...")

    config = load_yaml(config_path)
    model_cfg = config['model']
    data_cfg = config['data']

    model = BreathPhaseFeedForward(
        n_mels=model_cfg['n_mels'],
        context_frames=model_cfg['context_frames'],
        hidden_dim=model_cfg['hidden_dim'],
        num_classes=model_cfg['num_classes'],
        dropout=model_cfg['dropout'],
    )

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    data_config = DataConfig(**data_cfg)
    full_model = AudioToBreathClassifierFF(model, data_config)
    full_model.eval()

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
    print(f"Feed-Forward model exported and verified: {onnx_path}")


if __name__ == "__main__":
    model_path = "checkpoints/best_model_epoch_1.pth"
    onnx_path = "checkpoints/best_model_ff.onnx"

    export_breath_classifier_to_onnx(model_path, onnx_path)
    print("ONNX export complete.")
