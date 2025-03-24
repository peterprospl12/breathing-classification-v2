import os
import datetime
import time
import threading
import wave
import pygame
import pygame.freetype
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyaudio

# ###########################################################################################
# If there's an issue with the microphone, find the index of the microphone you want to use
# in the terminal, along with its sampleRate.
# Then, change the variable RATE and INPUT_DEVICE_INDEX below.
# ###########################################################################################

# Audio configuration
AUDIO_CHUNK = 1024
PLOT_CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
INPUT_DEVICE_INDEX = 0

# Paths for saving WAV files
WAV_PATHS = {
    "inhale": "../data-raw/raw/manual/inhale/",
    "exhale": "../data-raw/raw/manual/exhale/",
    "silence": "../data-raw/raw/manual/silence/",
}


def initialize_paths():
    """Creates necessary directories if they do not exist."""
    for path in WAV_PATHS.values():
        os.makedirs(path, exist_ok=True)


class SharedAudioResource:
    """Manages shared access to the audio device."""

    def __init__(self):
        self.p = pyaudio.PyAudio()
        self._log_available_devices()
        self.stream = self.p.open(
            format=FORMAT, channels=CHANNELS, rate=RATE,
            input=True, frames_per_buffer=AUDIO_CHUNK,
            input_device_index=INPUT_DEVICE_INDEX
        )
        self.buffer = None
        self.read(AUDIO_CHUNK)

    def _log_available_devices(self):
        """Logs available audio devices."""
        for i in range(self.p.get_device_count()):
            print(self.p.get_device_info_by_index(i))

    def read(self, size):
        """Reads audio samples."""
        self.buffer = self.stream.read(size, exception_on_overflow=False)
        return self.buffer

    def close(self):
        """Closes the audio stream and releases resources."""
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


def draw_text(text, pos, font, screen):
    """Draws text on the screen."""
    text_surface, _ = font.render(text, (255, 255, 255))
    screen.blit(text_surface, (pos[0] - text_surface.get_width() // 2, pos[1] - text_surface.get_height() // 2))


def pygame_thread(audio):
    """Handles the graphical interface and audio recording."""
    pygame.init()
    initialize_paths()

    WIDTH, HEIGHT = 1366, 768
    FONT_SIZE = 24
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.freetype.SysFont(None, FONT_SIZE)
    clock = pygame.time.Clock()

    recording = False
    running = True
    frames = []
    wf = None
    start_time = None

    def start_recording(label):
        nonlocal start_time, recording, frames, wf
        start_time = time.time()
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = os.path.join(WAV_PATHS[label], f"{timestamp}.wav")
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        frames = []
        recording = True

    def save_audio():
        """Saves the recorded audio to a file."""
        if wf:
            wf.writeframes(b''.join(frames))
            wf.close()

    while running:
        screen.fill((0, 0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s and recording:
                    save_audio()
                    recording = False
                elif event.key in (pygame.K_w, pygame.K_e, pygame.K_r):
                    label = {pygame.K_w: "inhale", pygame.K_e: "exhale", pygame.K_r: "silence"}[event.key]
                    if recording:
                        save_audio()
                    start_recording(label)

        if recording:
            elapsed_time = time.time() - start_time
            draw_text(f"Recording: {elapsed_time:.2f}s", (WIDTH // 2, HEIGHT // 2), font, screen)
            frames.append(audio.read(AUDIO_CHUNK))
        else:
            draw_text("W: Inhale | E: Exhale | R: Silence | S: Stop", (WIDTH // 2, HEIGHT // 2), font, screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def plot_audio(audio):
    """Displays a real-time audio plot."""
    def animate(i):
        if audio.buffer:
            data = np.frombuffer(audio.buffer, dtype=np.int16)
            left_channel = data[::2]
            right_channel = data[1::2]
            line1.set_ydata(left_channel)
            line2.set_ydata(right_channel)
        return line1, line2

    fig, axs = plt.subplots(2)
    x = np.arange(0, 2 * PLOT_CHUNK, 2)
    line1, = axs[0].plot(x, np.random.rand(PLOT_CHUNK))
    line2, = axs[1].plot(x, np.random.rand(PLOT_CHUNK))

    for ax in axs:
        ax.set_ylim(-1500, 1500)
        ax.set_xlim(0, PLOT_CHUNK / 2)

    ani = animation.FuncAnimation(fig, animate, frames=100, blit=True)
    plt.show()


if __name__ == "__main__":
    audio = SharedAudioResource()
    pygame_thread_instance = threading.Thread(target=pygame_thread, args=(audio,))
    pygame_thread_instance.start()
    plot_audio(audio)
    pygame_thread_instance.join()
    audio.close()
