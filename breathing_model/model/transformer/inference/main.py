from breathing_model.model.transformer.utils import Config
from audio import AudioStream
from transform import MelSpectrogramTransform
from model_loader import BreathPhaseClassifier
from counter import BreathCounter, BreathType
from visualization import RealTimePlot
import time
import logging

def main():
    logging.basicConfig(level=logging.INFO)
    config = Config.from_yaml('../config.yaml')

    audio = AudioStream(config.audio)
    mel_transform = MelSpectrogramTransform(config.data)
    classifier = BreathPhaseClassifier('../../trained_models/3/best_model_epoch13.pth', config.model)
    counter = BreathCounter()
    plot = RealTimePlot(config)

    try:
        while True:
            start_time = time.time()

            raw_audio = audio.read()
            if raw_audio is None:
                time.sleep(config.audio.chunk_length)

            mel = mel_transform(raw_audio)
            pred_class, probs = classifier.predict(mel)

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
