import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import time
import torch
import torchaudio
from model.lib import SharedAudioResource
from model.transformer_model.transformer_model import BreathPhaseTransformerSeq

#############################################
# Settings and constants
#############################################
MODEL_PATH = '../../trained_models/1/transformer_model_88.pth'

REFRESH_TIME = 0.3  # time in seconds to read audio
FORMAT = pyaudio.paInt16
CHANNELS = 1  # 1 mono | 2 stereo
RATE = 44100  # sampling rate
DEVICE_INDEX = 1  # microphone device index (listed in the console output)
CHUNK_SIZE = int(RATE * REFRESH_TIME)

INHALE_COUNTER = 0
EXHALE_COUNTER = 0

running = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MelTransformer:
    def __init__(self):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=RATE,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )

    def get_mel_transform(self, y, sr=RATE):
        # y: int16 signal; convert to float32 in the range [-1, 1]
        y = y.astype(np.float32) / 32768.0
        # Ensure the signal is mono
        if y.ndim != 1:
            raise Exception("Otrzymano sygnał nie-mono!")
        # Convert to tensor (shape: [1, num_samples])
        waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        # Compute Mel spectrogram – result: [1, n_mels, time_steps]
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-9)
        # Add channel dimension – expected shape: (batch, 1, n_mels, time_steps)d
        mel_spec = mel_spec.unsqueeze(0)
        return mel_spec

#############################################
# Prediction class
#############################################
class RealTimeAudioClassifier:
    def __init__(self, model_path):
        self.model = BreathPhaseTransformerSeq(
            n_mels=128,
            num_classes=3,
            d_model=192,
            nhead=8,
            num_transformer_layers=6
        ).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        self.mel_transformer = MelTransformer()

    def predict(self, y):
        """Wykonuje lokalną predykcję."""
        with torch.no_grad():
            mel = self.mel_transformer.get_mel_transform(y)
            mel = mel.to(device)
            logits = self.model(mel)  # shape: (1, time_steps, num_classes)
            probabilities = torch.softmax(logits, dim=2)
            probs_np = probabilities.squeeze(0).cpu().numpy()  # (time_steps, num_classes)


            preds = np.argmax(probs_np, axis=1)

            if len(preds) == 0:
                predicted_class = 2
                class_probabilities = np.array([0.0, 0.0, 1.0])
            else:
                predicted_class = int(np.bincount(preds).argmax())
                class_probabilities = np.mean(probs_np, axis=0)


            return predicted_class, class_probabilities


#############################################
# Plot configuration
#############################################
PLOT_TIME_HISTORY = 5  # seconds
PLOT_CHUNK_SIZE = CHUNK_SIZE
plot_data = np.zeros((RATE * PLOT_TIME_HISTORY, 1))
x_line_space = np.arange(0, RATE * PLOT_TIME_HISTORY, 1)
predictions = np.zeros((int(PLOT_TIME_HISTORY / REFRESH_TIME), 1))

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(plot_data, color='white')


def on_key(event):
    global running, INHALE_COUNTER, EXHALE_COUNTER
    if event.key == ' ':
        plt.close()
        running = False
    elif event.key == 'r':
        INHALE_COUNTER = 0
        EXHALE_COUNTER = 0


fig.canvas.manager.set_window_title('Realtime Breath Detector (Naciśnij [SPACJA] aby zatrzymać, [R] aby zresetować licznik)')
fig.suptitle(
    f'Wdechy: {INHALE_COUNTER}  Wydechy: {EXHALE_COUNTER}   (Czerwony - Wdech, Zielony - Wydech, Niebieski - Cisza)')
fig.canvas.mpl_connect('key_press_event', on_key)
y_lim = (-500, 500)
face_color = (0, 0, 0)
ax.set_facecolor(face_color)
ax.set_ylim(y_lim)


def update_plot(frames, current_prediction):
    global plot_data, predictions, ax, INHALE_COUNTER, EXHALE_COUNTER
    # Update plot buffer
    plot_data = np.roll(plot_data, -len(frames))
    plot_data[-len(frames):] = frames.reshape(-1, 1)
    predictions = np.roll(predictions, -1)
    predictions[-1] = current_prediction

    if current_prediction == 0:
        EXHALE_COUNTER += 1
    elif current_prediction == 1:
        INHALE_COUNTER += 1

    ax.clear()
    # For each segment (REFRESH_TIME window) plot the signal with color based on prediction
    for i in range(len(predictions)):
        if predictions[i] == 0:
            color = 'green'  # exhale
        elif predictions[i] == 1:
            color = 'red'  # inhale
        else:
            color = 'blue'  # silence
        start = i * PLOT_CHUNK_SIZE
        end = (i + 1) * PLOT_CHUNK_SIZE
        ax.plot(x_line_space[start:end], plot_data[start:end] / 4, color=color)

    ax.set_facecolor(face_color)
    ax.set_ylim(y_lim)
    fig.suptitle(
        f'Wdechy: {INHALE_COUNTER}  Wydechy: {EXHALE_COUNTER}   (Czerwony - Wdech, Zielony - Wydech, Niebieski - Cisza)')
    plt.draw()
    plt.pause(0.01)


if __name__ == '__main__':
    audio = SharedAudioResource(chunk_size=CHUNK_SIZE, format=FORMAT, channels=CHANNELS,
                                rate=RATE, device_index=DEVICE_INDEX)
    classifier = RealTimeAudioClassifier(MODEL_PATH)

    try:
        while running:
            start_time = time.time()

            buffer = audio.read()
            if buffer is None:
                print("Iteration error: buffer is None")
                time.sleep(REFRESH_TIME)
                continue

            prediction, probability = classifier.predict(buffer)

            label_map = {0: "Exhale", 1: "Inhale", 2: "Silence"}
            print(f"Prediction: {label_map.get(prediction, 'Unknown')} ({prediction}), Probability: {probability}")

            update_plot(buffer, prediction)
            print("Iteration time:", time.time() - start_time)
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        # Clean up
        audio.close()
        plt.close(fig)
        print("Audio stream closed.")