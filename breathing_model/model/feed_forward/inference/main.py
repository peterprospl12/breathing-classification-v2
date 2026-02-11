import time
import logging

from breathing_model.model.feed_forward.utils import FFConfig
from breathing_model.model.transformer.utils import BreathType
from breathing_model.model.transformer.inference.audio import AudioStream
from breathing_model.model.transformer.inference.audio_buffer import AudioBuffer
from breathing_model.model.transformer.inference.transform import MelSpectrogramTransform
from breathing_model.model.transformer.inference.counter import BreathCounter
from breathing_model.model.transformer.inference.visualization import RealTimePlot
from breathing_model.model.feed_forward.inference.model_loader import BreathPhaseClassifierFF


def main():
    logging.basicConfig(level=logging.INFO)
    config = FFConfig.from_yaml('../config.yaml')

    audio = AudioStream(config.audio)
    mel_transform = MelSpectrogramTransform(config.data)
    classifier = BreathPhaseClassifierFF('../checkpoints/best_model_epoch_14.pth', config.model, config.data)
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

            mel = mel_transform(buf_audio)
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
