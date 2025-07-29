import pyaudio
import numpy as np
import torch

from breathing_model.model.transformer_model_ref.utils import AudioConfig

class AudioStream:
    def __init__(self, config: AudioConfig):
        self.config = config
        self.p = pyaudio.PyAudio()
        self.chunk_size = int(config.chunk_length * config.sample_rate)

        self._list_devices()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=config.channels,
            rate=config.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=config.device_index
        )


    def _list_devices(self):
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            print(f"[{i}] {info['name']} | ch: {info['maxInputChannels']} | sr: {int(info['defaultSampleRate'])}")

    def read(self) -> torch.Tensor | None:
        try:
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            return np.frombuffer(data, dtype=np.float32)
        except Exception as e:
            print(f"Audio error: {e}")
            return None

    def close(self) -> None:
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()