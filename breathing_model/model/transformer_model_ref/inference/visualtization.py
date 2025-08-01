import matplotlib.pyplot as plt
import numpy as np
from breathing_model.model.transformer_model_ref.utils import Config



class RealTimePlot:
    def __init__(self, config: Config):
        self.config = config
        self.history_samples = int(config.audio.sample_rate * config.plot.history_seconds)
        self.chunk_size = int(config.audio.sample_rate * config.audio.chunk_length)
        self.buffer = np.zeros(self.history_samples)
        self.predictions = np.full(self.history_samples // self.chunk_size, 2, dtype=int)

        self.fig, self.ax = plt.subplots(1,1,figsize=(12,6))
        self.fig.canvas.manager.set_window_title("Breath phase detector")
        self.fig.suptitle("Live Breath Detection (Space: quit, R: reset)")

        self.ax.set_facecolor('black')
        self.ax.set_ylim(-500, 500)
        self.ax.set_xlim(0, self.history_samples)
        self.ax.axis('off')


    def _set_key_events(self):
        def on_key(event):
            if event.key == ' ':
                plt.close(self.fig)
            elif event.key == 'r':
                self.reset_data()

        self.fig.canvas.mpl_connect('key_press_event', on_key)

    def reset_data(self):
        self.buffer.fill(0)
        self.predictions.fill(2)

    def update(self, audio_chunk: np.ndarray, prediction: int):
        # Przesuń bufor
        self.buffer = np.roll(self.buffer, -len(audio_chunk))
        self.buffer[-len(audio_chunk):] = audio_chunk

        self.predictions = np.roll(self.predictions, -1)
        self.predictions[-1] = prediction

        self.ax.clear()
        self.ax.set_facecolor('black')
        self.ax.set_ylim(-500, 500)
        self.ax.set_xlim(0, self.history_samples)
        self.ax.axis('off')

        colors = {0: 'green', 1: 'red', 2: 'blue'}
        for i, pred in enumerate(self.predictions):
            start = i * self.chunk_size
            end = start + self.chunk_size
            if end <= len(self.buffer):
                x = np.arange(start, end)
                y = self.buffer[start:end] / 4
                self.ax.plot(x, y, color=colors[pred], linewidth=1.2)

        plt.draw()
        plt.pause(0.001)

    def close(self):
        plt.close(self.fig)