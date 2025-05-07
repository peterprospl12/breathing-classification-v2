import os
import datetime
import time
import threading
import wave
import csv
import pygame
import pygame.freetype
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyaudio
from enum import Enum

# ###########################################################################################
# If there's an issue with the microphone, find the index of the microphone you want to use
# in the terminal, along with its sampleRate.
# Then, change the variable RATE and INPUT_DEVICE_INDEX below.
# ###########################################################################################

# Audio configuration
AUDIO_CHUNK = 1024
PLOT_CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1  # Changed to mono for simplicity
RATE = 44100
INPUT_DEVICE_INDEX = 0
RECORDING_DURATION = 10  # Duration in seconds to save each recording

# Paths for saving files
DATA_DIR = "../data/"
RAW_DIR = os.path.join(DATA_DIR, "raw")
CSV_DIR = os.path.join(DATA_DIR, "label")

class NoseMouth(Enum):
    Nose = 0
    Mouth = 1

class MicrophoneQuality(Enum):
    Good = 0
    Medium = 1
    Bad = 2

def initialize_paths():
    """Creates necessary directories if they do not exist."""
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)


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
        print("Available audio devices:")
        for i in range(self.p.get_device_count()):
            dev_info = self.p.get_device_info_by_index(i)
            print(f"Device {i}: {dev_info['name']}")
            print(f"  Max Input Channels: {dev_info['maxInputChannels']}")
            print(f"  Default Sample Rate: {dev_info['defaultSampleRate']}")

    def read(self, size):
        """Reads audio samples."""
        self.buffer = self.stream.read(size, exception_on_overflow=False)
        return self.buffer

    def close(self):
        """Closes the audio stream and releases resources."""
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


class BreathingRecorder:
    """Records audio and tracks breathing classes with timestamps."""

    def __init__(self, audio, noseMouthMode, microphoneQuality, personName):
        self.audio = audio
        self.current_class = "silence"  # Default class
        self.frames = []
        self.recording = False
        self.events = []  # List to store class changes with sample positions
        self.current_sample = 0
        self.recording_start_time = None
        self.last_save_time = None
        self.filename_base = None
        self.noseMouthMode = noseMouthMode
        self.microphoneQuality = microphoneQuality
        self.personName = personName

    def start_recording(self):
        """Starts a new recording session."""
        self.recording = True
        self.frames = []
        self.events = []
        self.current_sample = 0
        self.recording_start_time = time.time()
        self.last_save_time = self.recording_start_time
        self.filename_base = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Add initial class
        self.events.append((self.current_class, 0))

        print(f"Recording started. Current class: {self.current_class}")

    def change_class(self, new_class):
        """Changes the breathing class and records the transition."""
        if self.recording:
            self.events.append((new_class, self.current_sample))
            self.current_class = new_class
            print(f"Class changed to: {new_class} at sample {self.current_sample}")

    def record_chunk(self):
        """Records a chunk of audio and updates sample count."""
        if self.recording:
            chunk = self.audio.read(AUDIO_CHUNK)
            self.frames.append(chunk)
            self.current_sample += AUDIO_CHUNK

            # Check if it's time to save (every RECORDING_DURATION seconds)
            current_time = time.time()
            if current_time - self.last_save_time >= RECORDING_DURATION:
                self.save_sequence()
                self.last_save_time = current_time

    def save_sequence(self):
        """Saves the current audio sequence and corresponding CSV file."""
        if not self.recording or not self.frames:
            return

        # Generate filename prefix with person name, breathing mode, and mic quality
        nose_mouth_str = "nose" if self.noseMouthMode == NoseMouth.Nose else "mouth"
        mic_quality_str = self.microphoneQuality.name.lower()
        file_prefix = f"{self.personName}_{nose_mouth_str}_{mic_quality_str}"

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        wav_filename = os.path.join(RAW_DIR, f"{file_prefix}_{timestamp}.wav")
        csv_filename = os.path.join(CSV_DIR, f"{file_prefix}_{timestamp}.csv")

        # Save WAV file
        wf = wave.open(wav_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        # Save CSV file with class, start_sample, end_sample format
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['class', 'start_sample', 'end_sample'])

            # Write each breathing event
            for i in range(len(self.events) - 1):
                current_class, start_sample = self.events[i]
                _, end_sample = self.events[i + 1]
                writer.writerow([current_class, start_sample, end_sample])

            # Write the last event (if there's at least one event)
            if self.events:
                last_class, last_start = self.events[-1]
                writer.writerow([last_class, last_start, self.current_sample])

        print(f"Saved sequence: {file_prefix}_{timestamp}")
        print(f"  - Audio file: {wav_filename}")
        print(f"  - CSV file: {csv_filename}")

        # Reset frames but keep recording
        self.frames = []
        # Keep events but adjust sample positions to be relative to the new segment
        last_sample = self.current_sample
        self.events = [(self.current_class, 0)]  # Start the new segment with current class
        self.current_sample = AUDIO_CHUNK  # Reset current sample count but account for first chunk

    def stop_recording(self):
        """Stops the recording and saves final data."""
        if self.recording:
            self.save_sequence()
            self.recording = False
            print("Recording stopped.")


def draw_text(text, pos, font, screen):
    """Draws text on the screen."""
    text_surface, _ = font.render(text, (255, 255, 255))
    screen.blit(text_surface, (pos[0] - text_surface.get_width() // 2,
                               pos[1] - text_surface.get_height() // 2))


def pygame_thread(recorder):
    """Handles the graphical interface and user input."""
    pygame.init()

    WIDTH, HEIGHT = 1200, 700
    FONT_SIZE = 24
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Breathing Recorder")
    font = pygame.freetype.SysFont(None, FONT_SIZE)
    clock = pygame.time.Clock()

    running = True

    # Define button positions and sizes
    button_width, button_height = 180, 60
    button_margin = 20
    button_y = HEIGHT - button_height - 30

    # Define buttons [x, y, width, height, text, key, action]
    buttons = [
        [WIDTH // 2 - button_width * 2 - button_margin * 1.5, button_y, button_width, button_height,
         "START (SPACE)", pygame.K_SPACE, "start"],
        [WIDTH // 2 - button_width // 2, button_y, button_width, button_height,
         "STOP (S)", pygame.K_s, "stop"],
        [WIDTH // 2 + button_width + button_margin * 1.5, button_y, button_width, button_height,
         "QUIT (ESC)", pygame.K_ESCAPE, "quit"],
    ]

    # Class buttons
    class_buttons = [
        [WIDTH // 4 - button_width // 2, HEIGHT // 2, button_width, button_height,
         "INHALE (w)", pygame.K_i, "inhale"],
        [WIDTH // 2, HEIGHT // 2, button_width, button_height,
         "EXHALE (E)", pygame.K_e, "exhale"],
        [WIDTH * 3 // 4 + button_width // 2, HEIGHT // 2, button_width, button_height,
         "SILENCE (R)", pygame.K_c, "silence"],
    ]

    def draw_button(button):
        x, y, w, h, text, _, _ = button
        color = (100, 100, 100)

        # Highlight the current breathing class button
        if len(button) > 6 and button[6] == recorder.current_class and recorder.recording:
            color = (0, 200, 0)

        pygame.draw.rect(screen, color, (x, y, w, h))
        pygame.draw.rect(screen, (200, 200, 200), (x, y, w, h), 2)
        draw_text(text, (x + w // 2, y + h // 2), font, screen)

    while running:
        screen.fill((0, 0, 0))

        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE and not recorder.recording:
                    recorder.start_recording()
                elif event.key == pygame.K_s and recorder.recording:
                    recorder.stop_recording()
                elif event.key == pygame.K_w and recorder.recording:
                    recorder.change_class("inhale")
                elif event.key == pygame.K_e and recorder.recording:
                    recorder.change_class("exhale")
                elif event.key == pygame.K_r and recorder.recording:
                    recorder.change_class("silence")

        # Record audio chunk if recording
        if recorder.recording:
            recorder.record_chunk()

        # Draw status
        if recorder.recording:
            elapsed = time.time() - recorder.recording_start_time
            next_save = RECORDING_DURATION - (time.time() - recorder.last_save_time)
            status_text = f"RECORDING | Class: {recorder.current_class} | Time: {elapsed:.1f}s | Next save: {next_save:.1f}s"
            status_color = (255, 50, 50)
        else:
            status_text = "NOT RECORDING | Press SPACE to start"
            status_color = (200, 200, 200)

        # Draw status text
        text_surface, _ = font.render(status_text, status_color)
        screen.blit(text_surface, (WIDTH // 2 - text_surface.get_width() // 2, 50))

        # Draw class name in big font if recording
        if recorder.recording:
            class_font = pygame.freetype.SysFont(None, 72)
            class_text, _ = class_font.render(recorder.current_class.upper(), (0, 200, 0))
            screen.blit(class_text, (WIDTH // 2 - class_text.get_width() // 2, 120))

        # Draw all buttons
        for button in buttons:
            draw_button(button)

        # Draw class buttons
        for button in class_buttons:
            draw_button(button)

        # Draw instructions
        instructions = [
            "Press SPACE to start recording",
            "Press S to stop recording",
            "Press W/E/R to mark INHALE/EXHALE/SILENCE",
            f"Files will be saved every {RECORDING_DURATION} seconds"
        ]

        for i, instruction in enumerate(instructions):
            instr_surface, _ = font.render(instruction, (180, 180, 180))
            screen.blit(instr_surface, (WIDTH // 2 - instr_surface.get_width() // 2,
                                        HEIGHT - 200 + i * 30))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def plot_audio(audio):
    """Displays a real-time audio plot."""

    def animate(i):
        if audio.buffer:
            data = np.frombuffer(audio.buffer, dtype=np.int16)
            line.set_ydata(data[:PLOT_CHUNK])
        return line,

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(0, PLOT_CHUNK)
    line, = ax.plot(x, np.random.rand(PLOT_CHUNK))

    ax.set_ylim(-32000, 32000)
    ax.set_xlim(0, PLOT_CHUNK)
    ax.set_title("Real-time Audio Waveform")

    ani = animation.FuncAnimation(fig, animate, frames=100, blit=True)
    plt.show()


if __name__ == "__main__":
    initialize_paths()
    audio = SharedAudioResource()

    MODE = NoseMouth.Nose
    MICROPHONEQUALITY = MicrophoneQuality.Bad
    PERSONNAME = "Iwo"

    recorder = BreathingRecorder(audio, MODE, MICROPHONEQUALITY, PERSONNAME)

    # Start the pygame interface in a separate thread
    pygame_thread_instance = threading.Thread(target=pygame_thread, args=(recorder,))
    pygame_thread_instance.daemon = True
    pygame_thread_instance.start()

    # Start the audio plot
    plot_audio(audio)

    # Clean up
    audio.close()