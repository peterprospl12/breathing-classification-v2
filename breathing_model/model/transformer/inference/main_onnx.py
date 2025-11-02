import time
import logging
import numpy as np
import torch
import onnxruntime as ort

from breathing_model.model.transformer.utils import Config, BreathType
from breathing_model.model.transformer.inference.audio import AudioStream
from breathing_model.model.transformer.inference.audio_buffer import AudioBuffer
from breathing_model.model.transformer.inference.counter import BreathCounter
from breathing_model.model.transformer.inference.visualization import RealTimePlot


class OnnxBreathPhaseClassifier:
    """ONNX classifier expecting raw audio input [batch, audio_length]. Preprocessing (MelSpectrogram) is inside ONNX model."""
    def __init__(self, onnx_path: str, config: Config):
        self.config = config

        self.last_mel_frames = int(0.2 * config.data.sample_rate / config.data.hop_length)

        self.session_options = ort.SessionOptions()
        self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=self.session_options,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        print(f"Model loaded with input name: {self.input_name}")

    def _pad_audio_to_length(self, audio: np.ndarray, target_length: int = 154350) -> np.ndarray:
        """
        Pad or trim audio to exact target length.
        target_length = 154350 (corresponds to approx. 3.5s at sample_rate=44100)
        """
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        if audio.ndim != 1:
            raise ValueError(f"Expected 1D audio, got shape {audio.shape}")

        if len(audio) < target_length:
            padding = np.zeros(target_length - len(audio), dtype=np.float32)
            audio = np.concatenate([audio, padding])
        elif len(audio) > target_length:
            audio = audio[-target_length:]

        return audio

    def predict(self, audio_waveform: np.ndarray) -> tuple[int, np.ndarray]:
        """
        audio_waveform: np.ndarray shape [T], float32 in [-1, 1]
        Returns (pred_class, mean_probs[classes]).
        """
        audio_padded = self._pad_audio_to_length(audio_waveform)  # [target_length]

        ort_inputs = {self.input_name: audio_padded}
        ort_outputs = self.session.run(None, ort_inputs)

        logits = ort_outputs[0]  # [1, time_frames, num_classes]
        if logits.ndim != 3 or logits.shape[0] != 1:
            raise RuntimeError(f"Unexpected ONNX output shape: {logits.shape}")

        logits_tensor = torch.from_numpy(logits).float()
        probs_tensor = torch.softmax(logits_tensor, dim=-1)  # [1, time_frames, num_classes]
        probs_np = probs_tensor.squeeze(0).numpy()  # [time_frames, num_classes]

        if probs_np.shape[0] == 0:
            num_classes = probs_np.shape[1] if probs_np.ndim == 2 else 3
            return 0, np.full((num_classes,), 1.0 / num_classes, dtype=np.float32)

        recent_probs = probs_np[-self.last_mel_frames:]

        mean_probs = recent_probs.mean(axis=0)

        predicted_class = int(np.argmax(mean_probs))
        return predicted_class, mean_probs


def main():
    logging.basicConfig(level=logging.INFO)
    config = Config.from_yaml('../config.yaml')
    onnx_path = '../best_models/best_model_epoch_31.onnx'

    audio = AudioStream(config.audio)
    classifier = OnnxBreathPhaseClassifier(onnx_path, config)
    counter = BreathCounter()
    plot = RealTimePlot(config)
    buffer = AudioBuffer(config.audio.sample_rate, 3.5)

    try:
        while True:
            start_time = time.time()

            raw_audio = audio.read()
            if raw_audio is None:
                time.sleep(config.audio.chunk_length)
                continue

            buffer.append(raw_audio)
            buf_audio = buffer.get()

            pred_class, probs = classifier.predict(buf_audio)

            counter.update(pred_class)
            plot.update(raw_audio, pred_class)

            print(f"Pred: {BreathType(pred_class).get_label()} | Prob: {probs.round(3)} | "
                  f"Counters: Inhale={counter.inhale}, Exhale={counter.exhale} | "
                  f"Time: {time.time() - start_time:.3f}s")

    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        audio.close()
        plot.close()
        print("Cleanup done")


if __name__ == '__main__':
    main()

