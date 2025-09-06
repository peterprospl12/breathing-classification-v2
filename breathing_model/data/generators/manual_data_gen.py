import os
import datetime
import time
import threading
import wave
import csv
import pygame
import pygame.freetype
import numpy as np
import pyaudio
from enum import Enum

# ###########################################################
# AUDIO_CHUNK = 2205 → dokładnie 50 ms
# 200 chunków = 10.00 sekund (441_000 próbek)
# 1200 chunków = 60 sekund (2_646_000 próbek)
# ###########################################################
# Audio configuration
AUDIO_CHUNK = 2205  # Must divide 441000 exactly (441000 / 2205 = 200)
PLOT_CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
INPUT_DEVICE_INDEX = 1

# Recording durations
TRAINING_RECORDING_DURATION = 10  # seconds
EVAL_RECORDING_DURATION = 60      # seconds

# Target samples
TRAINING_TARGET_SAMPLES = RATE * TRAINING_RECORDING_DURATION  # 441_000
EVAL_TARGET_SAMPLES = RATE * EVAL_RECORDING_DURATION          # 2_646_000

# Paths
DATA_DIR = "../train"
RAW_DIR = os.path.join(DATA_DIR, "raw")
CSV_DIR = os.path.join(DATA_DIR, "label")
EVAL_DATA_DIR = "../eval/"
EVAL_RAW_DIR = os.path.join(EVAL_DATA_DIR, "raw")
EVAL_CSV_DIR = os.path.join(EVAL_DATA_DIR, "label")


class NoseMouth(Enum):
    Nose = 0
    Mouth = 1


class MicrophoneQuality(Enum):
    Good = 0
    Medium = 1
    Bad = 2


class DataMeansOfUsage(Enum):
    Evaluation = 0
    Training = 1


def initialize_paths():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(EVAL_RAW_DIR, exist_ok=True)
    os.makedirs(EVAL_CSV_DIR, exist_ok=True)


class SharedAudioResource:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self._log_available_devices()
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=AUDIO_CHUNK,
            input_device_index=INPUT_DEVICE_INDEX
        )
        self.buffer = None
        self.read(AUDIO_CHUNK)

    def _log_available_devices(self):
        print("Available audio devices:")
        for i in range(self.p.get_device_count()):
            dev_info = self.p.get_device_info_by_index(i)
            print(f"Device {i}: {dev_info['name']}")
            print(f"  Max Input Channels: {dev_info['maxInputChannels']}")
            print(f"  Default Sample Rate: {dev_info['defaultSampleRate']}")

    def read(self, size):
        self.buffer = self.stream.read(size, exception_on_overflow=False)
        return self.buffer

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


class BreathingRecorder:
    def __init__(self, audio, noseMouthMode, microphoneQuality, personName, meansOfUsage):
        self.audio = audio
        self.current_class = "silence"
        self.frames = []
        self.recording = False
        self.events = []
        self.current_sample = 0
        self.recording_start_time = None
        self.last_save_time = None
        self.filename_base = None
        self.noseMouthMode = noseMouthMode
        self.microphoneQuality = microphoneQuality
        self.personName = personName
        self.meansOfUsage = meansOfUsage

        # Set target samples
        self.target_samples = TRAINING_TARGET_SAMPLES if self.meansOfUsage is DataMeansOfUsage.Training else EVAL_TARGET_SAMPLES

    def start_recording(self):
        self.recording = True
        self.frames = []
        self.events = [(self.current_class, 0)]
        self.current_sample = 0
        self.recording_start_time = time.time()
        self.last_save_time = self.recording_start_time
        self.filename_base = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        print(f"Recording started. Current class: {self.current_class}")

    def change_class(self, new_class):
        if self.recording:
            self.events.append((new_class, self.current_sample))
            self.current_class = new_class
            print(f"Class changed to: {new_class} at sample {self.current_sample}")

    def record_chunk(self):
        if self.recording:
            chunk = self.audio.read(AUDIO_CHUNK)
            self.frames.append(chunk)
            self.current_sample += AUDIO_CHUNK

            # ✅ Save ONLY if we have exactly the target number of samples
            if self.current_sample == self.target_samples:
                self.save_sequence()
                # Reset for next segment
                self.frames = []
                self.events = [(self.current_class, 0)]
                self.current_sample = 0
                self.last_save_time = time.time()
            elif self.current_sample > self.target_samples:
                # Overshot — reset without saving
                print(f"⚠️  Overshot: {self.current_sample} > {self.target_samples}. Resetting.")
                self.frames = []
                self.events = [(self.current_class, 0)]
                self.current_sample = 0
                self.last_save_time = time.time()

    def save_sequence(self):
        if not self.recording or not self.frames:
            return

        nose_mouth_str = "nose" if self.noseMouthMode == NoseMouth.Nose else "mouth"
        mic_quality_str = self.microphoneQuality.name.lower()
        file_prefix = f"{self.personName}_{nose_mouth_str}_{mic_quality_str}"
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        if self.meansOfUsage is DataMeansOfUsage.Training:
            wav_path = os.path.join(RAW_DIR, f"{file_prefix}_{timestamp}.wav")
            csv_path = os.path.join(CSV_DIR, f"{file_prefix}_{timestamp}.csv")
        else:
            wav_path = os.path.join(EVAL_RAW_DIR, f"{file_prefix}_{timestamp}.wav")
            csv_path = os.path.join(EVAL_CSV_DIR, f"{file_prefix}_{timestamp}.csv")

        # Save WAV
        wf = wave.open(wav_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.audio_client.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        # Save CSV
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['class', 'start_sample', 'end_sample'])
            for i in range(len(self.events) - 1):
                cls, start = self.events[i]
                _, end = self.events[i + 1]
                writer.writerow([cls, start, end])
            if self.events:
                last_cls, last_start = self.events[-1]
                writer.writerow([last_cls, last_start, self.current_sample])

        print(f"✅ SAVED EXACT: {file_prefix}_{timestamp} ({self.current_sample} samples)")
        print(f"    Audio: {wav_path}")
        print(f"    Labels: {csv_path}")


def pygame_thread(recorder, means_of_usage):
    pygame.init()
    WIDTH, HEIGHT = 1200, 700
    FONT_SIZE = 24
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Breathing Recorder")
    font = pygame.freetype.SysFont(None, FONT_SIZE)
    clock = pygame.time.Clock()
    running = True

    button_width, button_height = 180, 60
    button_margin = 20
    button_y = HEIGHT - button_height - 30

    buttons = [
        [WIDTH // 2 - button_width * 2 - button_margin * 1.5, button_y, button_width, button_height,
         "START (SPACE)", pygame.K_SPACE, "start"],
        [WIDTH // 2 - button_width // 2, button_y, button_width, button_height,
         "STOP (S)", pygame.K_s, "stop"],
        [WIDTH // 2 + button_width + button_margin * 1.5, button_y, button_width, button_height,
         "QUIT (ESC)", pygame.K_ESCAPE, "quit"],
    ]

    class_buttons = [
        [WIDTH // 4 - button_width // 2, HEIGHT // 2, button_width, button_height,
         "INHALE (W)", pygame.K_w, "inhale"],
        [WIDTH // 2 - button_width // 2, HEIGHT // 2, button_width, button_height,
         "EXHALE (E)", pygame.K_e, "exhale"],
        [WIDTH * 3 // 4 - button_width // 2, HEIGHT // 2, button_width, button_height,
         "SILENCE (R)", pygame.K_r, "silence"],
    ]

    def draw_button(button):
        x, y, w, h, text, _, _ = button
        color = (100, 100, 100)
        if len(button) > 6 and button[6] == recorder.current_class and recorder.recording:
            color = (0, 200, 0)
        pygame.draw.rect(screen, color, (x, y, w, h))
        pygame.draw.rect(screen, (200, 200, 200), (x, y, w, h), 2)
        text_surface, _ = font.render(text, (255, 255, 255))
        screen.blit(text_surface, (x + w // 2 - text_surface.get_width() // 2,
                                   y + h // 2 - text_surface.get_height() // 2))

    while running:
        screen.fill((0, 0, 0))
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

        if recorder.recording:
            recorder.record_chunk()

        # Status
        if recorder.recording:
            elapsed = time.time() - recorder.recording_start_time
            remaining = (recorder.target_samples - recorder.current_sample) / RATE
            status_text = f"RECORDING | Class: {recorder.current_class.upper()} | Time: {elapsed:.1f}s"
            status_color = (255, 50, 50)
        else:
            status_text = f"NOT RECORDING | Mode: {means_of_usage.name} | Press SPACE to start"
            status_color = (200, 200, 200)

        text_surface, _ = font.render(status_text, status_color)
        screen.blit(text_surface, (WIDTH // 2 - text_surface.get_width() // 2, 50))

        if recorder.recording:
            class_font = pygame.freetype.SysFont(None, 72)
            class_text, _ = class_font.render(recorder.current_class.upper(), (0, 200, 0))
            screen.blit(class_text, (WIDTH // 2 - class_text.get_width() // 2, 120))

        for button in buttons:
            draw_button(button)
        for button in class_buttons:
            draw_button(button)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    initialize_paths()
    audio = SharedAudioResource()
    MODE = NoseMouth.Nose
    MICROPHONEQUALITY = MicrophoneQuality.Medium
    PERSONNAME = "Piotr"
    MEANSOFUSAGE = DataMeansOfUsage.Training
    recorder = BreathingRecorder(audio, MODE, MICROPHONEQUALITY, PERSONNAME, MEANSOFUSAGE)

    pygame_thread_instance = threading.Thread(target=pygame_thread, args=(recorder, MEANSOFUSAGE))
    pygame_thread_instance.daemon = True
    pygame_thread_instance.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        audio.close()