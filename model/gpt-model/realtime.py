import os
import pyaudio
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
import time
import wave
from collections import deque

# Stałe
REFRESH_TIME = 0.5  # 0.5 sekundy na próbkę
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 48000
DEVICE_INDEX = 1  # Zmień na odpowiedni indeks twojego mikrofonu
CHUNK_SIZE = int(RATE * REFRESH_TIME)  # 22050 próbek

# Model
CLASSIFIER_MODEL_PATH = 'breath_classifier.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Definicja modelu BreathClassifier
class BreathClassifier(torch.nn.Module):
    def __init__(self, hidden_size=128, num_layers=2, num_classes=3):
        super(BreathClassifier, self).__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        self.lstm = torch.nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.dropout = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, waveform):
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = torch.log(mel_spec + 1e-9)
        log_mel_spec = log_mel_spec.transpose(1, 2)
        lstm_out, _ = self.lstm(log_mel_spec)
        last_time_step = lstm_out[:, -1, :]
        out = self.dropout(last_time_step)
        return self.fc(out)


# Klasa do obsługi dźwięku
class SharedAudioResource:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            input_device_index=DEVICE_INDEX
        )

    def read(self):
        return np.frombuffer(self.stream.read(CHUNK_SIZE, exception_on_overflow=False), dtype=np.int16)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


# Klasyfikator w czasie rzeczywistym
class RealTimeAudioClassifier:
    def __init__(self, model_path):
        self.model = BreathClassifier().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def predict(self, audio_buffer):
        waveform = audio_buffer.astype(np.float32)
        waveform = (waveform - np.mean(waveform)) / (np.std(waveform) + 1e-9)

        # Convert to mono by averaging the channels
        if waveform.ndim > 1 and waveform.shape[1] > 1:
            waveform = np.mean(waveform, axis=1)

        # Resample to 44.1 kHz
        waveform = torchaudio.transforms.Resample(orig_freq=RATE, new_freq=44100)(
            torch.from_numpy(waveform).unsqueeze(0))

        # Ensure the tensor is 3D (batch_size, num_channels, num_samples)
        tensor = waveform.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = self.model(tensor)
        return torch.argmax(logits).item()

# Konfiguracja wykresu
PLOT_TIME_HISTORY = 5
plot_data = np.zeros(RATE * PLOT_TIME_HISTORY)
predictions = deque(maxlen=int(PLOT_TIME_HISTORY / REFRESH_TIME))

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_facecolor((0, 0, 0))
ax.set_ylim(-500, 500)

# Liczniki
inhale_counter = 0
exhale_counter = 0


def update_plot(audio_data, prediction):
    global plot_data, predictions, inhale_counter, exhale_counter

    # Aktualizacja danych i predykcji
    plot_data = np.roll(plot_data, -len(audio_data))
    plot_data[-len(audio_data):] = audio_data
    predictions.append(prediction)

    # Aktualizacja wykresu
    ax.clear()
    for i, pred in enumerate(predictions):
        start = i * CHUNK_SIZE
        end = start + CHUNK_SIZE
        color = 'red' if pred == 1 else 'green' if pred == 0 else 'blue'
        ax.plot(np.arange(start, end), plot_data[start:end] / 4, color=color)

    # Aktualizacja liczników
    if prediction == 1:
        inhale_counter += 1
    elif prediction == 0:
        exhale_counter += 1

    ax.set_title(f'Inhale: {inhale_counter}  Exhale: {exhale_counter}')
    plt.pause(0.01)


# Główna pętla
if __name__ == "__main__":
    audio = SharedAudioResource()
    classifier = RealTimeAudioClassifier(CLASSIFIER_MODEL_PATH)

    try:
        while True:
            start_time = time.time()

            # Odczyt dźwięku
            buffer = audio.read()

            # Predykcja
            prediction = classifier.predict(buffer)
            print(f'Predicted: {["Exhale", "Inhale", "Silence"][prediction]}')

            # Aktualizacja wykresu
            update_plot(buffer, prediction)

            time.sleep(max(0, REFRESH_TIME - (time.time() - start_time)))

    except KeyboardInterrupt:
        audio.close()
        print("Zamykanie...")