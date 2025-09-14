import time
import logging
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

from breathing_model.model.transformer.inference.audio import AudioStream
from breathing_model.model.transformer.inference.transform import MelSpectrogramTransform
from breathing_model.model.transformer.inference.model_loader import BreathPhaseClassifier
from breathing_model.model.transformer.inference.counter import BreathCounter
from breathing_model.model.transformer.inference.visualization import RealTimePlot
from breathing_model.model.transformer.utils import Config, BreathType

class ArtifactDetector:
    """
    Artifact detector (cough, speech, clicks). Returns True only after several
    consecutive suspicious chunks (hysteresis) to avoid over-silencing.
    Suspicion criteria (OR with grouping):
      1. (low_conf AND low_rms)   -> model uncertain and quiet
      2. spike                    -> sudden energy spike
      3. (low_rms AND zcr_out)    -> noise/click with unusual structure
    """
    def __init__(self,
                 min_prob: float = 0.45,
                 spike_factor: float = 3.5,
                 rms_window: int = 60,
                 zcr_min: float = 0.005,
                 zcr_max: float = 0.12,
                 hysteresis: int = 2):
        self.min_prob = min_prob
        self.spike_factor = spike_factor
        self.zcr_min = zcr_min
        self.zcr_max = zcr_max
        self.hysteresis = max(1, hysteresis)
        self.rms_hist = deque(maxlen=rms_window)
        self._pending = 0  # how many suspicious in a row

    @staticmethod
    def _zcr(x: np.ndarray) -> float:
        if x.size < 2:
            return 0.0
        return float(np.mean(np.abs(np.diff(np.signbit(x)))))

    def check(self,
              audio: np.ndarray,
              model_class: int,
              probs: np.ndarray,
              rms_val: float,
              silence_threshold: float):

        prob = float(probs[model_class])
        self.rms_hist.append(rms_val)
        med_rms = np.median(self.rms_hist) if self.rms_hist else rms_val

        zcr = self._zcr(audio)
        peak = float(np.max(np.abs(audio)) + 1e-12)
        energy = float(np.sum(audio ** 2) + 1e-12)
        mean_e = energy / max(1, audio.size)
        spike_ratio = (peak * peak) / mean_e

        low_conf = prob < self.min_prob
        low_rms = rms_val < silence_threshold
        zcr_out = (zcr < self.zcr_min) or (zcr > self.zcr_max)
        rms_spike = (rms_val > med_rms * self.spike_factor) and (len(self.rms_hist) > 10)
        spike = rms_spike or (spike_ratio > 10.0 and peak > 5 * rms_val)

        suspicious = (low_conf and low_rms) or spike or (low_rms and zcr_out)

        if suspicious:
            self._pending += 1
        else:
            self._pending = 0

        override = self._pending >= self.hysteresis

        reasons = []
        if low_conf and low_rms: reasons.append("low_conf&low_rms")
        if spike: reasons.append("spike")
        if low_rms and zcr_out: reasons.append("low_rms&zcr")
        print(f"[ArtifactDbg] prob={prob:.3f} rms={rms_val:.5f} thr={silence_threshold:.5f} "
              f"med_rms={med_rms:.5f} zcr={zcr:.4f} spike_ratio={spike_ratio:.2f} "
              f"pend={self._pending} override={override} reasons={reasons}")

        return override, reasons


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
    artifact_detector = ArtifactDetector(min_prob=0.55,
                                         spike_factor=3.5,
                                         rms_window=60,
                                         zcr_min=0.005,
                                         zcr_max=0.12)

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
            final_class = BreathType(model_class)

            if final_class != BreathType.SILENCE:
                override, reasons = artifact_detector.check(
                    raw_audio, model_class, probs, rms_val, silence.threshold
                )
                if override:
                    final_class = BreathType.SILENCE
                    print(f"[Artifact] -> SILENCE ({','.join(reasons)})")

            if final_class == BreathType.SILENCE:
                silence.push_for_calibration(raw_audio)
            else:
                counter.update(int(final_class))

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