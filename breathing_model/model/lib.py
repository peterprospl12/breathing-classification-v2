import pyaudio
import numpy as np

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
            print(f"Błąd odczytu audio: {e}")
            return None

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()