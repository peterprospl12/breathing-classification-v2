from breathing_model.model.exhale_only_detection.utils import Config
from breathing_model.model.transformer.inference.audio import AudioStream
from breathing_model.model.transformer.inference.transform import MelSpectrogramTransform
from model_loader import BreathPhaseClassifier
from counter import BreathCounter, BreathType
from visualization import RealTimePlot
from breathing_model.model.transformer.inference.audio_buffer import AudioBuffer
import time
import logging

def main():
    logging.basicConfig(level=logging.INFO)
    config = Config.from_yaml('../config.yaml')

    audio = AudioStream(config.audio)
    mel_transform = MelSpectrogramTransform(config.data)
    classifier = BreathPhaseClassifier('../best_models/best_model_epoch_21.pth', config.model, config.data)
    counter = BreathCounter()
    plot = RealTimePlot(config)
    buffer = AudioBuffer(config.audio.sample_rate, 3.5)

    try:
        while True:
            start_time = time.time()

            raw_audio = audio.read()
            if raw_audio is None:
                time.sleep(config.audio.chunk_length)

            buffer.append(raw_audio)
            buf_audio = buffer.get()

            mel = mel_transform(buf_audio)
            pred_class, probs = classifier.predict(mel)

            counter.update(pred_class)
            plot.update(raw_audio, pred_class)

            print(f"Pred: {BreathType(pred_class).get_label()} | Prob: {probs.round(3)} | "
                  f"Counters: Exhale={counter.exhale} | "
                  f"Time: {time.time() - start_time:.3f}s")

    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        audio.close()
        plot.close()
        print("Cleanup done")


if __name__ == '__main__':
    main()
