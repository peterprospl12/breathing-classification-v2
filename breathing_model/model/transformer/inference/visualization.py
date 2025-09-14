import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from breathing_model.model.transformer.utils import Config, BreathType


class RealTimePlot:
    """
    Colored waveform: each audio chunk is a separate line segment whose color
    reflects predicted class. No background rectangles. Threshold lines kept.
    """
    def __init__(self, config: Config,
                 amplitude_gain: float = 0.95,
                 refresh_every: int = 1):
        self.config = config
        self.sample_rate = config.audio.sample_rate
        self.chunk_size = int(config.audio.sample_rate * config.audio.chunk_length)
        self.history_samples = int(self.sample_rate * config.plot.history_seconds)
        self.history_chunks = self.history_samples // self.chunk_size
        self.amplitude_gain = amplitude_gain
        self.refresh_every = max(1, refresh_every)

        # Bufory
        self.buffer = np.zeros(self.history_samples, dtype=np.float32)
        self.predictions = np.full(self.history_chunks, BreathType.SILENCE, dtype=int)

        # Pre‑obliczone X-y dla chunków
        self.x_chunks = []
        for i in range(self.history_chunks):
            start = i * self.chunk_size
            self.x_chunks.append(np.arange(self.chunk_size) + start)

        # Okno
        plt.ion()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 5))
        try:
            self.fig.canvas.manager.set_window_title("Breath phase detector (colored waveform)")
        except Exception:
            pass
        self.fig.suptitle("Live Breath Detection (SPACE quit, R reset)")

        # Oś
        self.ax.set_xlim(0, self.history_samples)
        self.ax.set_ylim(-1.0, 1.0)
        self.ax.set_facecolor('black')
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Linie progu (placeholdery)
        self.thr_pos = self.ax.axhline(0, color='yellow', linestyle='--', linewidth=0.8, zorder=3)
        self.thr_neg = self.ax.axhline(0, color='yellow', linestyle='--', linewidth=0.8, zorder=3)

        # Linie chunków (kolorowane)
        self.chunk_lines = []
        for _ in range(self.history_chunks):
            line, = self.ax.plot([], [], linewidth=1.0, solid_capstyle='round', zorder=2)
            self.chunk_lines.append(line)

        self._frame = 0
        self._connect_keys()

    def _connect_keys(self):
        def on_key(event):
            if event.key == ' ':
                plt.close(self.fig)
            elif event.key == 'r':
                self.reset()
        self.fig.canvas.mpl_connect('key_press_event', on_key)

    def reset(self):
        self.buffer.fill(0)
        self.predictions.fill(BreathType.SILENCE)
        for ln in self.chunk_lines:
            ln.set_data([], [])
            ln.set_color((1, 1, 1, 0.0))
        self.fig.canvas.draw_idle()

    @staticmethod
    def _color_for(pred: int):
        c = BreathType(pred).get_color()
        if isinstance(c, str):
            r, g, b = to_rgb(c)
        else:
            r, g, b = c[:3]
        # Silence -> półprzezroczyste wygaszenie
        if pred == BreathType.SILENCE:
            return (0.4, 0.4, 0.4, 0.35)
        return (r, g, b, 0.9)

    def update(self, audio_chunk: np.ndarray, prediction: int, threshold: float):
        # Roll audio
        L = len(audio_chunk)
        if L != self.chunk_size:
            # W razie różnic (ostatnia ramka) dopasuj
            audio_chunk = audio_chunk[:self.chunk_size] if L > self.chunk_size else np.pad(
                audio_chunk, (0, self.chunk_size - L)
            )
            L = self.chunk_size
        self.buffer = np.roll(self.buffer, -L)
        self.buffer[-L:] = audio_chunk * self.amplitude_gain  # pozostaje w [-1,1]

        # Roll predictions
        self.predictions = np.roll(self.predictions, -1)
        self.predictions[-1] = prediction

        # Aktualizacja linii chunków
        for idx in range(self.history_chunks):
            start = idx * self.chunk_size
            end = start + self.chunk_size
            y_seg = self.buffer[start:end]
            self.chunk_lines[idx].set_data(self.x_chunks[idx], y_seg)
            self.chunk_lines[idx].set_color(self._color_for(self.predictions[idx]))

        # Linie progu (przycięte do osi)
        ylim = self.ax.get_ylim()
        thr_clamped = np.clip(threshold, ylim[0], ylim[1])
        self.thr_pos.set_ydata([thr_clamped, thr_clamped])
        self.thr_neg.set_ydata([-thr_clamped, -thr_clamped])

        # Odświeżanie
        self._frame += 1
        if self._frame % self.refresh_every == 0:
            self.fig.canvas.draw_idle()
            plt.pause(0.0001)

    def close(self):
        plt.close(self.fig)
        plt.ioff()