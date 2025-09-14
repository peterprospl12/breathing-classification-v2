import time
import logging
import numpy as np
import matplotlib.pyplot as plt

from breathing_model.model.transformer.inference.audio import AudioStream
from breathing_model.model.transformer.inference.transform import MelSpectrogramTransform
from breathing_model.model.transformer.inference.model_loader import BreathPhaseClassifier
from breathing_model.model.transformer.inference.counter import BreathCounter
from breathing_model.model.transformer.inference.visualization import RealTimePlot
from breathing_model.model.transformer.utils import Config, BreathType


class VolumeBasedSilenceDetector:
    def __init__(self, threshold: float = 0.02):
        self.threshold = threshold
        self._calib_buffers = []

    @staticmethod
    def rms(audio: np.ndarray) -> float:
        if audio.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))

    def is_silence(self, audio: np.ndarray):
        r = self.rms(audio)
        return r < self.threshold, r

    def set_threshold(self, value: float):
        self.threshold = max(1e-5, float(value))

    def push_for_calibration(self, audio: np.ndarray):
        self._calib_buffers.append(audio.astype(np.float32))

    def calibrate_now(self) -> bool:
        if not self._calib_buffers:
            return False
        concatenated = np.concatenate(self._calib_buffers)
        frame = 1024
        frames = len(concatenated) // frame
        if frames == 0:
            return False
        rms_frames = [self.rms(concatenated[i*frame:(i+1)*frame]) for i in range(frames)]
        mean_r = float(np.mean(rms_frames))
        std_r = float(np.std(rms_frames))
        self.threshold = max(1e-4, mean_r + 2.0 * std_r)
        self._calib_buffers.clear()
        return True


def run():
    logging.basicConfig(level=logging.INFO)
    config = Config.from_yaml('../transformer/config.yaml')

    audio = AudioStream(config.audio)
    mel_transform = MelSpectrogramTransform(config.data)
    classifier = BreathPhaseClassifier('../transformer/checkpoints/best_model_epoch_16.pth', config.model)
    counter = BreathCounter()
    plot = RealTimePlot(config)
    silence = VolumeBasedSilenceDetector(threshold=0.02)

    print("[Keys] [ / ] threshold adjust | c calibrate | r reset counters | SPACE exit")

    def on_key(event):
        if event.key == ' ':
            plt.close(plot.fig)
        elif event.key == '[':
            silence.set_threshold(silence.threshold * 0.9)
            print(f"[Threshold] {silence.threshold:.5f}")
        elif event.key == ']':
            silence.set_threshold(silence.threshold * 1.1)
            print(f"[Threshold] {silence.threshold:.5f}")
        elif event.key == 'c':
            if silence.calibrate_now():
                print(f"[Calibrated] New threshold: {silence.threshold:.5f}")
            else:
                print("[Calibrate] Not enough silent data.")
        elif event.key == 'r':
            counter.reset()
            print("[Counters] Reset")

    plot.fig.canvas.mpl_connect('key_press_event', on_key)

    try:
        while plt.fignum_exists(plot.fig.number):
            loop_start = time.time()
            audio_start = time.time()
            raw_audio = audio.read()
            print(f"[Audio] Read time: {(time.time()-audio_start)*1000:.1f} ms")
            if raw_audio is None:
                time.sleep(config.audio.chunk_length)
                continue

            silence_time = time.time()
            is_sil, rms_val = silence.is_silence(raw_audio)
            print(f"[Silence] Check time: {(time.time()-silence_time)*1000:.1f} ms")

            transform_start = time.time()
            mel = mel_transform(raw_audio)
            model_class, probs = classifier.predict(mel)
            print(f"[Model] Inference time: {(time.time()-transform_start)*1000:.1f} ms")

            if is_sil:
                final_class = BreathType.SILENCE
                silence.push_for_calibration(raw_audio)
            else:
                final_class = BreathType(model_class)

            if final_class != BreathType.SILENCE:
                counter.update(int(final_class))

            # Aktualizacja wykresu (z liniami progu)
            start = time.time()
            plot.update(raw_audio, int(final_class), silence.threshold)
            print(f"[Plot] Update time: {(time.time()-start)*1000:.1f} ms")

            print(
                f"Display:{final_class.get_label():7s} | "
                f"Model:{BreathType(model_class).get_label():7s} | "
                f"RMS:{rms_val:.5f} Thr:{silence.threshold:.5f} | "
                f"Inh:{counter.inhale} Exh:{counter.exhale} | "
                f"Loop:{(time.time()-loop_start)*1000:.1f} ms"
            )

        print("Window closed - exiting.")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        audio.close()
        plot.close()
        plt.close('all')
        print("Cleanup done.")


if __name__ == "__main__":
    run()