import torch
import torch.onnx
import torchaudio
from breathing_model.model.exhale_only_detection.model import BreathPhaseTransformerSeq
from breathing_model.model.exhale_only_detection.utils import Config, DataConfig


class AudioToBreathClassifier(torch.nn.Module):
    """Wrapper model that includes MelSpectrogram preprocessing."""
    def __init__(self, classifier_model, sample_rate=44100, n_fft=2048, hop_length=512, n_mels=128):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

        self.classifier = classifier_model

    def forward(self, audio_signal):
        """
        audio_signal: [batch, audio_length] - raw audio, float32 in [-1, 1]
        Returns: [batch, time_frames, num_classes] - logits
        """
        # Konwertuj audio do Mel spektrogramu
        mel_spec = self.mel_transform(audio_signal)  # [batch, n_mels, time_frames]

        # Zastosuj log (zamiast AmplitudeToDB które może nie być wspierane w ONNX)
        mel_spec = torch.log(mel_spec + 1e-9)

        # Dodaj channel dimension: [batch, 1, n_mels, time_frames]
        mel_spec = mel_spec.unsqueeze(1)

        # Forward przez classifier
        return self.classifier(mel_spec)


def export_breath_classifier_to_onnx(model_path, onnx_path, audio_length=154350):
    """
    Eksportuje model transformer do ONNX z preprocessing'iem Mel spektrogramu.
    Wejście: surowy audio [1, audio_length]
    Wyjście: logity [1, time_frames, num_classes]
    """
    print("Exporting breath classifier to ONNX...")

    config = Config.from_yaml('./config.yaml')

    print("EXPORT config.data:", config.data.sample_rate, config.data.n_fft, config.data.hop_length, config.data.n_mels)

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
    full_model = AudioToBreathClassifier(
        model,
        sample_rate=config.data.sample_rate,
        n_fft=config.data.n_fft,
        hop_length=config.data.hop_length,
        n_mels=config.data.n_mels
    )
    full_model.eval()

    # Dummy input: [batch=1, audio_length]
    dummy_input = torch.randn(1, audio_length, dtype=torch.float32)

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
        opset_version=18,  # Zwiększ do 18
        verbose=False,
        dynamic_axes={
            "audio_input": {0: "batch"},
            "logits": {0: "batch"}
        }
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
