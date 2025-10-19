import numpy as np
from collections import deque

class AudioBuffer:
    def __init__(self, sample_rate: int, n_seconds: float):
        self.max_size = int(sample_rate * n_seconds)
        self.buffer = deque(maxlen=self.max_size)

    def append(self, chunk: np.ndarray):
        self.buffer.extend(chunk)

    def get(self) -> np.ndarray:
        if len(self.buffer) == 0:
            return np.array([], dtype=np.float32)
        return np.array(self.buffer, dtype=np.float32)
