import torch
import torch.onnx
import torchaudio
from model import BreathPhaseTransformerSeq


class AudioToBreathClassifier(torch.nn.Module):
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
        # audio_signal: [1, audio_length] float32 normalized
        mel_spec = self.mel_transform(audio_signal)
        mel_spec = torch.log(mel_spec + 1e-9)
        mel_spec = mel_spec.unsqueeze(1)  # Add channel dimension: [1, 1, n_mels, T]
        return self.classifier(mel_spec)


def export_breath_classifier_to_onnx(model_path, onnx_path, audio_length=154350):
    print("Exporting breath classifier to ONNX...")

    classifier = BreathPhaseTransformerSeq(
        n_mels=128,
        d_model=192,
        nhead=8,
        num_layers=6,
        num_classes=3
    )

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    classifier.load_state_dict(state_dict)
    classifier.eval()

    full_model = AudioToBreathClassifier(classifier)
    full_model.eval()

    # Dummy input as float32 and already normalized (range [-1, 1])
    dummy_input = torch.randn(1, audio_length, dtype=torch.float32)

    input_names = ["input"]
    output_names = ["output"]

    torch.onnx.export(
        full_model,
        dummy_input,
        onnx_path,
        opset_version=5,  # DODANE: opset_version=5 (było w działającym kodzie)
        dynamo=True,
        export_params=True,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        verbose=False
    )

    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"Breath classifier model exported and verified: {onnx_path}")


if __name__ == "__main__":
    # Model paths - using the model from realtime inference
    model_path = "trained_models/augmented_data_model/best_model_epoch_31.pth"
    onnx_path = "trained_models/augmented_data_model/best_model_epoch_31.onnx"

    # Export model
    export_breath_classifier_to_onnx(model_path, onnx_path)

    print("ONNX export complete.")
