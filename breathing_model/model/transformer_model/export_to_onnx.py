import torch
import torch.onnx
import torchaudio
import json
from realtime import BreathPhaseTransformerSeq
from breathing_model.model.invalid_data_filter_model.anomaly_detection_autoencoder import SimplerBreathingAutoencoder, \
    EnhancedReconstructionLoss


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


class AudioToAnomalyDetector(torch.nn.Module):
    def __init__(self, anomaly_model, threshold=1.1, sample_rate=44100, n_fft=1024, hop_length=512, n_mels=40):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.threshold = threshold

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

        self.anomaly_model = anomaly_model
        self.criterion = EnhancedReconstructionLoss()

    def forward(self, audio_signal):
        # Convert audio to mel spectrogram
        mel_spec = self.mel_transform(audio_signal)
        mel_spec = torch.log(mel_spec + 1e-9)
        mel_spec = mel_spec.unsqueeze(1)

        # Get reconstruction from autoencoder
        reconstruction, _ = self.anomaly_model(mel_spec)

        # Calculate reconstruction error
        error = self.criterion(reconstruction, mel_spec)

        # Determine if it's an anomaly (1 for anomaly, 0 for normal)
        is_anomaly = torch.gt(error, self.threshold).float()

        # Return error value and anomaly flag
        return torch.stack([error, is_anomaly])


def export_breath_classifier_to_onnx(model_path, onnx_path, audio_length=13230):
    print("Exporting breath classifier to ONNX...")
    classifier = BreathPhaseTransformerSeq(n_mels=40, num_classes=3, d_model=128, nhead=4, num_transformer_layers=2)
    classifier.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    classifier.eval()

    full_model = AudioToBreathClassifier(classifier)
    full_model.eval()

    # Dummy input as float32 and already normalized (range [-1, 1])
    dummy_input = torch.randn(1, audio_length, dtype=torch.float32)

    input_names = ["audio_signal"]
    output_names = ["output"]

    torch.onnx.export(
        full_model,
        dummy_input,
        onnx_path,
        opset_version=11,  # Using a more recent opset version for better compatibility
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


def export_anomaly_detector_to_onnx(model_path, threshold_path, onnx_path, audio_length=13230):
    print("Exporting anomaly detector to ONNX...")
    # Load anomaly model
    anomaly_model = SimplerBreathingAutoencoder(n_mels=40, latent_dim=8)

    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict) and 'autoencoder_state_dict' in checkpoint:
            anomaly_model.load_state_dict(checkpoint['autoencoder_state_dict'])
            threshold = checkpoint.get('anomaly_threshold', 1.1)
        else:
            anomaly_model.load_state_dict(checkpoint)
            # Load threshold from separate file
            try:
                with open(threshold_path, 'r') as f:
                    threshold_data = json.load(f)
                threshold = threshold_data["threshold"]
            except Exception as e:
                print(f"Error loading threshold: {e}. Using default value 1.1")
                threshold = 1.1
    except Exception as e:
        print(f"Error loading anomaly model: {e}")
        print("Creating untrained model for export")
        threshold = 1.1

    anomaly_model.eval()

    # Create wrapper with preprocessing
    full_model = AudioToAnomalyDetector(anomaly_model, threshold=threshold)
    full_model.eval()

    # Dummy input
    dummy_input = torch.randn(1, audio_length, dtype=torch.float32)

    input_names = ["audio_signal"]
    output_names = ["error_and_flag"]  # [error_value, is_anomaly]

    torch.onnx.export(
        full_model,
        dummy_input,
        onnx_path,
        opset_version=11,
        export_params=True,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        verbose=False
    )

    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"Anomaly detector model exported and verified: {onnx_path}")


if __name__ == "__main__":
    # Breath classifier paths
    breath_model_path = "best_breath_seq_transformer_model_CURR_BEST.pth"
    breath_onnx_path = "breath_classifier_model.onnx"

    # Anomaly detector paths
    anomaly_model_path = "../invalid_data_filter_model/best_breathing_anomaly_detector.pth"
    anomaly_threshold_path = "../invalid_data_filter_model/anomaly_threshold.json"
    anomaly_onnx_path = "anomaly_detector_model.onnx"

    # Export both models
    export_breath_classifier_to_onnx(breath_model_path, breath_onnx_path)
    export_anomaly_detector_to_onnx(anomaly_model_path, anomaly_threshold_path, anomaly_onnx_path)

    print("ONNX export complete. Both models have been exported.")