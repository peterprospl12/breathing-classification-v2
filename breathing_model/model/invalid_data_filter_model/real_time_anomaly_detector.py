import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import time
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import json
from enum import Enum


# Autoencoder model definition
class BreathingAutoencoder(nn.Module):
    def __init__(self, n_mels=40, latent_dim=16):
        super(BreathingAutoencoder, self).__init__()

        # Encoder (compress time dimension while preserving mel dimension initially)
        self.encoder = nn.Sequential(
            # Layer 1: (1, n_mels, time) -> (16, n_mels, time/2)
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Layer 2: (16, n_mels, time/2) -> (32, n_mels/2, time/4)
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Layer 3: (32, n_mels/2, time/4) -> (64, n_mels/4, time/8)
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Layer 4: (64, n_mels/4, time/8) -> (latent_dim, n_mels/8, time/16)
            nn.Conv2d(64, latent_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            # Layer 1: (latent_dim, n_mels/8, time/16) -> (64, n_mels/4, time/8)
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                               output_padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Layer 2: (64, n_mels/4, time/8) -> (32, n_mels/2, time/4)
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Layer 3: (32, n_mels/2, time/4) -> (16, n_mels, time/2)
            nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Layer 4: (16, n_mels, time/2) -> (1, n_mels, time)
            nn.ConvTranspose2d(16, 1, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), output_padding=(0, 1)),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        # Ensure output has same size as input in time dimension
        if reconstructed.size() != x.size():
            reconstructed = F.interpolate(reconstructed, size=(x.size(2), x.size(3)), mode='bilinear',
                                          align_corners=False)
        return reconstructed

    def encode(self, x):
        return self.encoder(x)


# Constants and settings
MODEL_PATH = 'best_breathing_autoencoder.pth'  # Path to the trained autoencoder
THRESHOLD_PATH = 'anomaly_threshold.json'  # Path to the anomaly threshold
SILENCE_THRESHOLD_PATH = 'silence_threshold.json'  # Path to the silence threshold

REFRESH_TIME = 0.3  # time in seconds to read audio
FORMAT = pyaudio.paInt16
CHANNELS = 1  # 1 mono | 2 stereo
RATE = 44100  # sampling rate
DEVICE_INDEX = 1  # microphone device index
CHUNK_SIZE = int(RATE * REFRESH_TIME)


# Audio result states
class AudioState(Enum):
    SILENCE = 0
    VALID = 1
    ANOMALY = 2


running = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Audio handling class
class SharedAudioResource:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.buffer_size = CHUNK_SIZE
        # Print available devices
        for i in range(self.p.get_device_count()):
            print(f"Device {i}: {self.p.get_device_info_by_index(i)['name']}")
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                                  input=True, frames_per_buffer=self.buffer_size,
                                  input_device_index=DEVICE_INDEX)

    def read(self):
        data = self.stream.read(self.buffer_size, exception_on_overflow=False)
        return np.frombuffer(data, dtype=np.int16)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


# Mel spectrogram transformer
class MelTransformer:
    def __init__(self):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=RATE,
            n_fft=1024,
            hop_length=512,
            n_mels=40
        )

    def get_mel_transform(self, y, sr=RATE):
        # y: int16 signal; convert to float32 in the range [-1, 1]
        y = y.astype(np.float32) / 32768.0
        # Ensure the signal is mono
        if y.ndim != 1:
            raise Exception("Non-mono signal received!")
        # Convert to tensor (shape: [1, num_samples])
        waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        # Compute Mel spectrogram – result: [1, n_mels, time_steps]
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-9)
        # Add channel dimension – expected shape: (batch, 1, n_mels, time_steps)
        mel_spec = mel_spec.unsqueeze(0)
        return mel_spec


# Audio anomaly detector class
class RealTimeAnomalyDetector:
    def __init__(self, model_path, threshold_path, silence_threshold_path):
        # Load the autoencoder model
        self.model = BreathingAutoencoder(n_mels=40, latent_dim=16).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        # Load the anomaly threshold
        with open(threshold_path, 'r') as f:
            threshold_data = json.load(f)
        self.threshold = threshold_data["threshold"]

        # Load the silence threshold
        try:
            with open(silence_threshold_path, 'r') as f:
                silence_data = json.load(f)
            self.silence_threshold = silence_data["threshold"]
        except (FileNotFoundError, json.JSONDecodeError):
            print("Silence threshold file not found or invalid. Using default value.")
            self.silence_threshold = 0.01  # default value

        # Initialize mel transformer
        self.mel_transformer = MelTransformer()

        # For keeping error history
        self.error_history = []
        self.max_history_size = 100

    def detect_anomaly(self, audio_buffer):
        with torch.no_grad():
            # Convert audio to mel spectrogram
            mel_spec = self.mel_transformer.get_mel_transform(audio_buffer)

            # Calculate RMS amplitude from mel spectrogram
            power = torch.exp(mel_spec) - 1e-9
            rms = torch.sqrt(torch.mean(power ** 2)).item()

            # Check if this is silence first
            if rms < self.silence_threshold:
                return AudioState.SILENCE, rms, 0.0

            # If not silence, use the autoencoder for anomaly detection
            mel_spec = mel_spec.to(device)

            # Get reconstruction from autoencoder
            reconstruction = self.model(mel_spec)

            # Calculate reconstruction error
            criterion = nn.MSELoss(reduction='none')
            error = criterion(reconstruction, mel_spec)
            error_value = error.mean().item()

            # Keep error history for dynamic plotting
            self.error_history.append(error_value)
            if len(self.error_history) > self.max_history_size:
                self.error_history.pop(0)

            # Determine if it's an anomaly
            is_anomalous = error_value > self.threshold

            return AudioState.ANOMALY if is_anomalous else AudioState.VALID, rms, error_value


# Plot configuration
PLOT_TIME_HISTORY = 5  # seconds
plot_data = np.zeros((RATE * PLOT_TIME_HISTORY))
x_line_space = np.arange(0, RATE * PLOT_TIME_HISTORY)
error_values = np.zeros((int(PLOT_TIME_HISTORY / REFRESH_TIME)))
rms_values = np.zeros((int(PLOT_TIME_HISTORY / REFRESH_TIME)))
audio_states = np.zeros((int(PLOT_TIME_HISTORY / REFRESH_TIME)), dtype=int)

# Create figure with three subplots (audio, error, rms)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
fig.canvas.manager.set_window_title('Breathing Sound Anomaly Detector (Press [SPACE] to stop)')

# Audio waveform subplot
line, = ax1.plot(plot_data, color='white')
ax1.set_facecolor((0, 0, 0))
ax1.set_ylim((-5000, 5000))
ax1.set_title('Audio Waveform')

# Error plot subplot
error_line, = ax2.plot(np.arange(len(error_values)), error_values, color='cyan')
threshold_line, = ax2.plot([0, len(error_values)], [0, 0], 'r--', label='Threshold')
ax2.set_facecolor((0, 0, 0.1))
ax2.set_ylim((0, 0.05))  # Will be adjusted dynamically
ax2.set_title('Reconstruction Error')
ax2.legend()

# RMS plot subplot
rms_line, = ax3.plot(np.arange(len(rms_values)), rms_values, color='yellow')
silence_line, = ax3.plot([0, len(rms_values)], [0, 0], 'g--', label='Silence Threshold')
ax3.set_facecolor((0, 0, 0.1))
ax3.set_ylim((0, 0.05))  # Will be adjusted dynamically
ax3.set_title('RMS Amplitude')
ax3.legend()


# Event handler for key presses
def on_key(event):
    global running
    if event.key == ' ':
        plt.close()
        running = False


fig.canvas.mpl_connect('key_press_event', on_key)


def update_plot(frames, audio_state, rms, error_value, anomaly_threshold, silence_threshold):
    global plot_data, error_values, rms_values, audio_states

    # Update audio buffer
    plot_data = np.roll(plot_data, -len(frames))
    plot_data[-len(frames):] = frames

    # Update values and states
    error_values = np.roll(error_values, -1)
    error_values[-1] = error_value
    rms_values = np.roll(rms_values, -1)
    rms_values[-1] = rms
    audio_states = np.roll(audio_states, -1)
    audio_states[-1] = audio_state.value

    # Plot audio waveform with color based on audio state
    ax1.clear()
    if audio_state == AudioState.ANOMALY:
        ax1.set_title('Audio Waveform - ANOMALY DETECTED', color='red')
        line_color = 'red'
    elif audio_state == AudioState.SILENCE:
        ax1.set_title('Audio Waveform - SILENCE', color='blue')
        line_color = 'blue'
    else:
        ax1.set_title('Audio Waveform - Valid Sound', color='green')
        line_color = 'green'

    ax1.plot(x_line_space, plot_data, color=line_color)
    ax1.set_facecolor((0, 0, 0))
    ax1.set_ylim((-5000, 5000))

    # Plot error values with color coding
    ax2.clear()
    for i in range(len(error_values) - 1):
        if audio_states[i] == AudioState.ANOMALY.value:
            color = 'red'
        elif audio_states[i] == AudioState.SILENCE.value:
            color = 'blue'
        else:
            color = 'cyan'
        ax2.plot([i, i + 1], [error_values[i], error_values[i + 1]], color=color)

    # Set a dynamic y-axis limit for the error plot
    max_error = max(max(error_values) * 1.2, anomaly_threshold * 1.5, 0.001)
    ax2.set_ylim((0, max_error))

    # Plot threshold line
    ax2.axhline(y=anomaly_threshold, color='r', linestyle='--', label='Anomaly Threshold')
    ax2.set_title(f'Reconstruction Error (Current: {error_value:.5f}, Threshold: {anomaly_threshold:.5f})')
    ax2.set_facecolor((0, 0, 0.1))
    ax2.legend()

    # Plot RMS values
    ax3.clear()
    for i in range(len(rms_values) - 1):
        if audio_states[i] == AudioState.ANOMALY.value:
            color = 'red'
        elif audio_states[i] == AudioState.SILENCE.value:
            color = 'blue'
        else:
            color = 'green'
        ax3.plot([i, i + 1], [rms_values[i], rms_values[i + 1]], color=color)

    # Set dynamic y-axis limit for RMS plot
    max_rms = max(max(rms_values) * 1.2, silence_threshold * 3, 0.001)
    ax3.set_ylim((0, max_rms))

    # Plot silence threshold line
    ax3.axhline(y=silence_threshold, color='g', linestyle='--', label='Silence Threshold')
    ax3.set_title(f'RMS Amplitude (Current: {rms:.5f}, Threshold: {silence_threshold:.5f})')
    ax3.set_facecolor((0, 0, 0.1))
    ax3.legend()

    plt.draw()
    plt.pause(0.01)


# Main loop
if __name__ == '__main__':
    # Initialize audio resources and anomaly detector
    audio = SharedAudioResource()
    detector = RealTimeAnomalyDetector(MODEL_PATH, THRESHOLD_PATH, SILENCE_THRESHOLD_PATH)

    print(f"Anomaly detection threshold: {detector.threshold:.5f}")
    print(f"Silence detection threshold: {detector.silence_threshold:.5f}")
    print("Ready to analyze your breathing sounds. Press SPACE to stop.")

    while running:
        start_time = time.time()

        # Read audio data
        buffer = audio.read()
        if buffer is None:
            continue

        # Detect silence/anomalies using two-stage approach
        audio_state, rms, error_value = detector.detect_anomaly(buffer)

        # Update plot
        update_plot(buffer, audio_state, rms, error_value, detector.threshold, detector.silence_threshold)

        # Print results
        status_map = {
            AudioState.SILENCE: "SILENCE (Valid)",
            AudioState.VALID: "VALID BREATH",
            AudioState.ANOMALY: "ANOMALY"
        }
        status = status_map[audio_state]
        print(
            f"Status: {status} - RMS: {rms:.5f} - Error: {error_value:.5f} - Processed in {time.time() - start_time:.3f}s")

    # Clean up resources
    audio.close()
    print("Application closed.")