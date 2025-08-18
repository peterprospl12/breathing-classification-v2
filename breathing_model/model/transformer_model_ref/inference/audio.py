from typing import Optional
import pyaudio
import numpy as np
import logging
from model.transformer_model_ref.utils import AudioConfig

logger = logging.getLogger(__name__)

class AudioStream:
    def __init__(self, config: AudioConfig):
        self.config = config
        self.audio_client = pyaudio.PyAudio()
        self.chunk_size = int(config.chunk_length * config.sample_rate)

        self._list_devices()
        self.stream = self.audio_client.open(
            format=pyaudio.paFloat32,
            channels=config.channels,
            rate=config.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=config.device_index
        )

    def _list_devices(self) -> None:
        device_count = self.audio_client.get_device_count()
        logger.info("Available audio devices:")

        for i in range(device_count):
            info = self.audio_client.get_device_info_by_index(i)
            logger.info(
                f"[{i}] {info['name']} | "
                f"ch: {info['maxInputChannels']} | "
                f"sr: {int(info['defaultSampleRate'])}"
            )

    def read(self) -> Optional[np.ndarray]:
        try:
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            return np.frombuffer(data, dtype=np.float32)
        except Exception as e:
            logger.error(f"Audio reading error: {e}")
            return None

    def close(self) -> None:
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        if self.audio_client:
            self.audio_client.terminate()
            self.audio_client = None


class SharedAudioResource:
    def __init__(self, chunk_size=1024, format=pyaudio.paInt16, channels=1, rate=44100, device_index=1):
        self.p = pyaudio.PyAudio()
        self.buffer_size = chunk_size
        # Print available devices
        for i in range(self.p.get_device_count()):
            print(self.p.get_device_info_by_index(i))
        self.stream = self.p.open(format=format, channels=channels, rate=rate,
                                  input=True, frames_per_buffer=self.buffer_size,
                                  input_device_index=device_index)
    def read(self):
        try:
            data = self.stream.read(self.buffer_size, exception_on_overflow=True)
            return np.frombuffer(data, dtype=np.int16)
        except IOError as e:
            print(f"Error reading audio: {e}")
            return None

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
