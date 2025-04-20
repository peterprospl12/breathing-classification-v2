import torch
import torch.onnx
import torchaudio
from realtime import BreathPhaseTransformerSeq

class AudioToBreathClassifier(torch.nn.Module):
    def __init__(self, classifier_model, sample_rate=44100, n_fft=1024, hop_length=512, n_mels=40):
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
        # Zakładamy, że audio_signal jest już znormalizowany (float32 w zakresie [-1, 1])
        mel_spec = self.mel_transform(audio_signal)
        mel_spec = torch.log(mel_spec + 1e-9)
        mel_spec = mel_spec.unsqueeze(1)
        return self.classifier(mel_spec)


def export_model_to_onnx(model_path, onnx_path, audio_length=13230):
    classifier = BreathPhaseTransformerSeq(n_mels=40, num_classes=3, d_model=128, nhead=4, num_transformer_layers=2)
    classifier.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    classifier.eval()

    full_model = AudioToBreathClassifier(classifier)
    full_model.eval()

    # Dummy input jako float32 i już znormalizowany (zakres [-1, 1])
    dummy_input = torch.randn(1, audio_length, dtype=torch.float32)

    input_names = ["audio_signal"]
    output_names = ["output"]

    torch.onnx.export(
        full_model,
        dummy_input,
        onnx_path,
        opset_version=5,
        export_params=True,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamo=True,
        verbose=False
    )

    import onnx
    onnx_model = onnx.load(onnx_path)
    print(onnx_model.ir_version)
    onnx.checker.check_model(onnx_model)
    print("Model ONNX zweryfikowany pomyślnie!")

if __name__ == "__main__":
    model_path = "best_breath_seq_transformer_model_CURR_BEST.pth"
    onnx_path = "breath_classifier_model_audio_input.onnx"
    export_model_to_onnx(model_path, onnx_path)
