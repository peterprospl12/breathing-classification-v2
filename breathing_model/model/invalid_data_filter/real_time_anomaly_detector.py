import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import time
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum

from breathing_model.model.transformer.inference.audio import SharedAudioResource


# Updated autoencoder based on SimplerBreathingAutoencoder
class SimplerBreathingAutoencoder(nn.Module):
    def __init__(self, n_mels=40, latent_dim=8):
        super(SimplerBreathingAutoencoder, self).__init__()

        # Simplified encoder
        self.encoder = nn.Sequential(
            # Layer 1: (1, n_mels, time) -> (16, n_mels, time/2)
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2),

            # Layer 2: (16, n_mels, time/2) -> (32, n_mels/2, time/4)
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2),

            # Layer 3: (32, n_mels/2, time/4) -> (latent_dim, n_mels/4, time/8)
            nn.Conv2d(32, latent_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2),
        )

        # Simplified decoder
        self.decoder = nn.Sequential(
            # Layer 1: (latent_dim, n_mels/4, time/8) -> (32, n_mels/2, time/4)
            nn.ConvTranspose2d(latent_dim, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                               output_padding=(1, 1)),
            nn.LeakyReLU(0.2),

            # Layer 2: (32, n_mels/2, time/4) -> (16, n_mels, time/2)
            nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                               output_padding=(1, 1)),
            nn.LeakyReLU(0.2),

            # Layer 3: (16, n_mels, time/2) -> (1, n_mels, time)
            nn.ConvTranspose2d(16, 1, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1),
                               output_padding=(0, 1)),
        )

    def forward(self, x):
        # Normalize input while preserving scale
        x_scale = torch.mean(torch.abs(x)) + 1e-8
        x_normalized = x / x_scale

        # Encode and decode
        latent = self.encoder(x_normalized)
        reconstructed = self.decoder(latent)

        # Restore original scale
        reconstructed = reconstructed * x_scale

        # Ensure output matches input size
        if reconstructed.size() != x.size():
            reconstructed = F.interpolate(reconstructed, size=(x.size(2), x.size(3)),
                                          mode='bilinear', align_corners=False)
        return reconstructed, latent

    def encode(self, x):
        x_scale = torch.mean(torch.abs(x)) + 1e-8
        x_normalized = x / x_scale
        return self.encoder(x_normalized)


# Enhanced loss function from the new model
class EnhancedReconstructionLoss(nn.Module):
    def __init__(self, mse_weight=0.3, log_mse_weight=0.4, corr_weight=0.3):
        super(EnhancedReconstructionLoss, self).__init__()
        self.mse_weight = mse_weight
        self.log_mse_weight = log_mse_weight
        self.corr_weight = corr_weight
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, x, y):
        # Standard MSE
        mse_loss = self.mse(x, y)

        # Logarithmic MSE for better low-energy pattern detection
        eps = 1e-8
        log_x = torch.log(torch.abs(x) + eps)
        log_y = torch.log(torch.abs(y) + eps)
        log_mse_loss = self.mse(log_x, log_y)

        # Pearson correlation for pattern preservation
        x_flat = x.reshape(x.size(0), -1)
        y_flat = y.reshape(y.size(0), -1)

        # Center the data
        x_centered = x_flat - x_flat.mean(dim=1, keepdim=True)
        y_centered = y_flat - y_flat.mean(dim=1, keepdim=True)

        # Calculate correlation
        x_std = torch.sqrt(torch.sum(x_centered ** 2, dim=1) + eps)
        y_std = torch.sqrt(torch.sum(y_centered ** 2, dim=1) + eps)
        corr = torch.sum(x_centered * y_centered, dim=1) / (x_std * y_std)
        corr_loss = 1.0 - corr.mean()  # Maximize correlation

        # Combined loss
        total_loss = (self.mse_weight * mse_loss +
                      self.log_mse_weight * log_mse_loss +
                      self.corr_weight * corr_loss)

        return total_loss

# Constants and settings
MODEL_PATH = 'best_breathing_anomaly_detector.pth'  # Path to the trained autoencoder
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


# Volume-invariant audio anomaly detector class
class RealTimeAnomalyDetector:
    def __init__(self, model_path):
        # Load the autoencoder model
        self.model = SimplerBreathingAutoencoder(n_mels=40, latent_dim=8).to(device)

        # Handle different model saving formats
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'autoencoder_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['autoencoder_state_dict'])
                self.threshold = checkpoint.get('anomaly_threshold', None)
            else:
                self.model.load_state_dict(checkpoint)
                self.threshold = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.threshold = None

        self.model.eval()

        # # Load threshold if not included in model checkpoint
        # if self.threshold is None:
        #     try:
        #         with open(threshold_path, 'r') as f:
        #             threshold_data = json.load(f)
        #         self.threshold = threshold_data["threshold"]
        #     except (FileNotFoundError, json.JSONDecodeError) as e:
        #         print(f"Anomaly threshold file error: {e}. Using default value.")
        #         self.threshold = 0.5  # default

        self.threshold = 1.1
        # Initialize mel transformer
        self.mel_transformer = MelTransformer()

        # New enhanced loss function
        self.criterion = EnhancedReconstructionLoss()

        # For tracking error history
        self.error_history = []
        self.max_history_size = 100

    def detect_anomaly(self, audio_buffer):
        with torch.no_grad():
            # Convert audio to mel spectrogram
            mel_spec = self.mel_transformer.get_mel_transform(audio_buffer)
            mel_spec = mel_spec.to(device)

            # Calculate spectral energy (still helpful for metrics, but not for silence detection)
            power = torch.exp(mel_spec) - 1e-9
            spectral_energy = torch.mean(power, dim=1)
            energy_level = torch.quantile(spectral_energy, 0.9).item()

            # Get reconstruction from autoencoder
            reconstruction, _ = self.model(mel_spec)

            # Calculate error using the enhanced loss
            error = self.criterion(reconstruction, mel_spec)
            error_value = error.item()

            # Track error history
            self.error_history.append(error_value)
            if len(self.error_history) > self.max_history_size:
                self.error_history.pop(0)

            # Determine if it's an anomaly
            is_anomalous = error_value > self.threshold

            # Only return VALID or ANOMALY states
            return AudioState.ANOMALY if is_anomalous else AudioState.VALID, energy_level, error_value





# Plot configuration
PLOT_TIME_HISTORY = 5  # seconds
plot_data = np.zeros((RATE * PLOT_TIME_HISTORY))
x_line_space = np.arange(0, RATE * PLOT_TIME_HISTORY)
error_values = np.zeros((int(PLOT_TIME_HISTORY / REFRESH_TIME)))
energy_values = np.zeros((int(PLOT_TIME_HISTORY / REFRESH_TIME)))  # Now tracking energy instead of RMS
audio_states = np.zeros((int(PLOT_TIME_HISTORY / REFRESH_TIME)), dtype=int)

# Create figure with three subplots (audio, error, energy)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
fig.canvas.manager.set_window_title('Volume-Invariant Breathing Sound Detector (Press [SPACE] to stop)')

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
ax2.set_title('Pattern Similarity Error')
ax2.legend()

# Energy plot subplot (renamed from RMS)
energy_line, = ax3.plot(np.arange(len(energy_values)), energy_values, color='yellow')
silence_line, = ax3.plot([0, len(energy_values)], [0, 0], 'g--', label='Silence Threshold')
ax3.set_facecolor((0, 0, 0.1))
ax3.set_ylim((0, 0.05))  # Will be adjusted dynamically
ax3.set_title('Spectral Energy')
ax3.legend()


# Event handler for key presses
def on_key(event):
    global running
    if event.key == ' ':
        plt.close()
        running = False


fig.canvas.mpl_connect('key_press_event', on_key)


def update_plot(frames, audio_state, energy_level, error_value, anomaly_threshold):
    global plot_data, error_values, energy_values, audio_states

    # Update audio buffer
    plot_data = np.roll(plot_data, -len(frames))
    plot_data[-len(frames):] = frames

    # Update values and states
    error_values = np.roll(error_values, -1)
    error_values[-1] = error_value
    energy_values = np.roll(energy_values, -1)
    energy_values[-1] = energy_level
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
    ax2.set_title(f'Pattern Similarity Error (Current: {error_value:.5f}, Threshold: {anomaly_threshold:.5f})')
    ax2.set_facecolor((0, 0, 0.1))
    ax2.legend()

    # Plot energy values
    ax3.clear()
    for i in range(len(energy_values) - 1):
        if audio_states[i] == AudioState.ANOMALY.value:
            color = 'red'
        elif audio_states[i] == AudioState.SILENCE.value:
            color = 'blue'
        else:
            color = 'green'
        ax3.plot([i, i + 1], [energy_values[i], energy_values[i + 1]], color=color)


    plt.draw()
    plt.pause(0.01)


# Main loop
if __name__ == '__main__':
    # Initialize audio resources and anomaly detector
    audio = SharedAudioResource(chunk_size=CHUNK_SIZE, format=FORMAT, channels=CHANNELS,
                                rate=RATE, device_index=DEVICE_INDEX)
    detector = RealTimeAnomalyDetector(MODEL_PATH)

    print(f"Anomaly detection threshold: {detector.threshold:.5f}")
    print("Ready to analyze your breathing sounds. Press SPACE to stop.")
    print("NOTE: This detector is now volume-invariant and focuses on sound patterns.")

    while running:
        start_time = time.time()

        # Read audio data
        buffer = audio.read()
        if buffer is None:
            continue

        # Detect silence/anomalies using the volume-invariant approach
        audio_state, energy_level, error_value = detector.detect_anomaly(buffer)

        # Update plot
        update_plot(buffer, audio_state, energy_level, error_value, detector.threshold)

        # Print results
        status_map = {
            AudioState.SILENCE: "SILENCE (Valid)",
            AudioState.VALID: "VALID BREATH",
            AudioState.ANOMALY: "ANOMALY"
        }
        status = status_map[audio_state]
        print(
            f"Status: {status} - Energy: {energy_level:.5f} - Pattern Error: {error_value:.5f} - Processed in {time.time() - start_time:.3f}s")

    # Clean up resources
    audio.close()
    print("Application closed.")