import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import time
import onnxruntime as ort
from enum import Enum
from model.transformer_model_ref.inference.audio import SharedAudioResource

#############################################
# Settings and constants
#############################################
MODEL_PATH = 'breath_classifier_model_audio_input.onnx'  # Ścieżka do modelu ONNX
REFRESH_TIME = 0.3  # czas w sekundach odczytu audio
FORMAT = pyaudio.paInt16
CHANNELS = 1  # 1 mono | 2 stereo
RATE = 44100  # częstotliwość próbkowania
DEVICE_INDEX = 1  # indeks urządzenia mikrofonu (wyświetlany w konsoli)
CHUNK_SIZE = int(RATE * REFRESH_TIME)
AUDIO_LENGTH = 13230  # stała długość wejścia wymagana przez model (0.3s * 44100)

INHALE_COUNTER = 0
EXHALE_COUNTER = 0

running = True


class PredictionModes(Enum):
    LOCAL = 1
    HTTP_SERVER = 2
    SOCKET = 3


#############################################
# ONNX Prediction class
#############################################
class RealTimeAudioClassifier:
    def __init__(self, model_path):
        # Inicjalizacja ONNX Runtime
        self.ort_session = ort.InferenceSession(model_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        print(f"Model loaded with input name: {self.input_name}")

    def preprocess_audio(self, audio_data):
        """Przygotowanie danych audio do modelu ONNX"""
        # Konwersja z int16 na float32 w zakresie [-1, 1]
        audio_float = audio_data.astype(np.float32) / 32768.0

        # Sprawdź rozmiar - jeśli potrzeba, dostosuj do wymaganej długości
        if len(audio_float) > AUDIO_LENGTH:
            # Przytnij
            audio_float = audio_float[:AUDIO_LENGTH]
        elif len(audio_float) < AUDIO_LENGTH:
            # Uzupełnij zerami
            padding = np.zeros(AUDIO_LENGTH - len(audio_float), dtype=np.float32)
            audio_float = np.concatenate([audio_float, padding])

        # Dodaj wymiar batcha (1, audio_length)
        return np.expand_dims(audio_float, axis=0)

    def predict(self, audio_data):
        """Wykonaj predykcję na danych audio za pomocą modelu ONNX"""
        try:
            # Przygotowanie danych
            processed_audio = self.preprocess_audio(audio_data)

            # Przeprowadzenie wnioskowania
            ort_inputs = {self.input_name: processed_audio}
            ort_outputs = self.ort_session.run(None, ort_inputs)

            # Przetwarzanie wyjścia
            logits = ort_outputs[0]  # shape: (1, time_steps, num_classes)

            # Konwersja logits na prawdopodobieństwa przez softmax
            exp_logits = np.exp(logits)
            probs = exp_logits / np.sum(exp_logits, axis=2, keepdims=True)

            # Wybierz klasę z największym prawdopodobieństwem dla każdego kroku czasowego
            preds = np.argmax(probs[0], axis=1)  # (time_steps,)

            # Wybierz najczęściej występującą klasę jako ostateczną predykcję
            predicted_class = int(np.bincount(preds).argmax())

            return predicted_class
        except Exception as e:
            print(f"Błąd podczas predykcji: {e}")
            return 2  # Silence jako domyślna klasa w przypadku błędu


#############################################
# Plot configuration
#############################################
PLOT_TIME_HISTORY = 5  # seconds
PLOT_CHUNK_SIZE = CHUNK_SIZE
plot_data = np.zeros((RATE * PLOT_TIME_HISTORY, 1))
x_line_space = np.arange(0, RATE * PLOT_TIME_HISTORY, 1)
predictions = np.zeros((int(PLOT_TIME_HISTORY / REFRESH_TIME), 1))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(plot_data, color='white')


def on_key(event):
    global running, INHALE_COUNTER, EXHALE_COUNTER
    if event.key == ' ':
        plt.close()
        running = False
    elif event.key == 'r':
        INHALE_COUNTER = 0
        EXHALE_COUNTER = 0


fig.canvas.manager.set_window_title('Realtime Breath Detector (Press [SPACE] to stop, [R] to reset counter)')
fig.suptitle(f'Inhales: {INHALE_COUNTER}  Exhales: {EXHALE_COUNTER}   (Red - Inhale, Green - Exhale, Blue - Silence)')
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
        f'Inhales: {INHALE_COUNTER}  Exhales: {EXHALE_COUNTER}   (Red - Inhale, Green - Exhale, Blue - Silence)')
    plt.draw()
    plt.pause(0.01)


if __name__ == '__main__':
    audio = SharedAudioResource(chunk_size=CHUNK_SIZE, format=FORMAT, channels=CHANNELS,
                                rate=RATE, device_index=DEVICE_INDEX)
    classifier = RealTimeAudioClassifier(MODEL_PATH)

    while running:
        start_time = time.time()

        # Odczyt CHUNK_SIZE próbek z mikrofonu
        buffer = audio.read()
        if buffer is None:
            continue

        print(f"Audio buffer shape: {buffer.shape}")
        prediction = classifier.predict(buffer)

        print("Prediction:", prediction)

        update_plot(buffer, prediction)
        print(f"Iteration time: {time.time() - start_time:.4f}s")

    audio.close()