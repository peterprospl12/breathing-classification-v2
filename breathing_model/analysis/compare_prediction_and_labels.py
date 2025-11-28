import numpy as np
import matplotlib.pyplot as plt
import wave
import csv
import torch
import librosa
import os
import yaml
from matplotlib.lines import Line2D
from sklearn.metrics import classification_report, accuracy_score

# --- IMPORTY LSTM ---
from breathing_model.archive.lstm.model_classes import AudioClassifierLSTM

# --- IMPORTY TRANSFORMER ---
from breathing_model.model.transformer.utils import Config
from breathing_model.model.transformer.inference.transform import MelSpectrogramTransform
from breathing_model.model.transformer.inference.model_loader import BreathPhaseClassifier

# Jeśli AudioBuffer jest dostępny, używamy go. Jeśli nie, wrapper obsłuży buforowanie prosto w numpy.
try:
    from breathing_model.model.transformer.audio_buffer import AudioBuffer

    USE_CUSTOM_BUFFER = True
except ImportError:
    USE_CUSTOM_BUFFER = False


# ==========================================
# WRAPPERY MODELI
# ==========================================

class LSTMClassifierWrapper:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AudioClassifierLSTM()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.hidden = None

    def reset_state(self):
        self.hidden = None

    def predict(self, audio_chunk: np.ndarray, sample_rate: int = 44100) -> int:
        # Normalizacja do float32
        if audio_chunk.dtype == np.int16:
            audio_float = audio_chunk.astype(np.float32) / 32768.0
        else:
            audio_float = audio_chunk.astype(np.float32)

        # Preprocessing LSTM (Librosa)
        mfcc = librosa.feature.mfcc(y=audio_float, sr=sample_rate, n_mfcc=13)
        features = mfcc.mean(axis=1)

        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs, self.hidden = self.model(input_tensor, self.hidden)
            logits = outputs[0, 0]
            prediction_idx = torch.argmax(logits).item()

        return prediction_idx


class TransformerClassifierWrapper:
    def __init__(self, config_path, model_path):
        # Ładowanie konfiguracji
        self.config = Config.from_yaml(config_path)

        # Inicjalizacja komponentów Transformera
        self.mel_transform = MelSpectrogramTransform(self.config.data)
        self.classifier = BreathPhaseClassifier(model_path, self.config.model, self.config.data)

        # Bufor audio (3.5s kontekstu zgodnie z Twoim snippetem)
        self.context_duration = 3.5
        self.sample_rate = self.config.audio.sample_rate
        self.buffer_size = int(self.sample_rate * self.context_duration)

        if USE_CUSTOM_BUFFER:
            self.buffer = AudioBuffer(self.sample_rate, self.context_duration)
        else:
            # Fallback jeśli nie uda się zaimportować klasy AudioBuffer
            self.raw_buffer = np.zeros(self.buffer_size, dtype=np.float32)

    def reset_state(self):
        if USE_CUSTOM_BUFFER:
            self.buffer = AudioBuffer(self.sample_rate, self.context_duration)
        else:
            self.raw_buffer = np.zeros(self.buffer_size, dtype=np.float32)

    def predict(self, audio_chunk: np.ndarray, sample_rate: int = 44100) -> int:
        # Normalizacja
        if audio_chunk.dtype == np.int16:
            audio_float = audio_chunk.astype(np.float32) / 32768.0
        else:
            audio_float = audio_chunk.astype(np.float32)

        # Obsługa bufora (Transformer potrzebuje kontekstu, np. 3.5s)
        if USE_CUSTOM_BUFFER:
            self.buffer.append(audio_float)
            buf_audio = self.buffer.get()
        else:
            # Przesuń bufor i dodaj nowe dane (numpy implementation)
            self.raw_buffer = np.roll(self.raw_buffer, -len(audio_float))
            self.raw_buffer[-len(audio_float):] = audio_float
            buf_audio = self.raw_buffer

        # Predykcja
        # MelSpectrogramTransform oczekuje tensora lub numpy array
        mel = self.mel_transform(buf_audio)

        # BreathPhaseClassifier.predict zwraca (klasa, prawdopodobienstwa)
        pred_class, _ = self.classifier.predict(mel)

        return pred_class


# ==========================================
# NARZĘDZIA POMOCNICZE
# ==========================================

def read_audio_file(wav_path: str) -> tuple[np.ndarray, int]:
    with wave.open(wav_path, 'rb') as wf:
        n_frames = wf.getnframes()
        audio_data = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
        frame_rate = wf.getframerate()
    return audio_data, frame_rate


def read_labels(csv_path: str) -> list[dict[str, int | str]]:
    labels = []
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            if len(row) < 3: continue
            class_name, start_sample, end_sample = row
            labels.append({
                'class': class_name,
                'start': int(start_sample),
                'end': int(end_sample)
            })
    return labels


def get_color_for_class(class_name: str | int) -> str:
    # 0: Exhale (Green), 1: Inhale (Red), 2: Silence (Blue)
    if class_name == 'inhale' or class_name == 1:
        return 'red'
    elif class_name == 'exhale' or class_name == 0:
        return 'green'
    return 'blue'


def get_ground_truth_for_chunk(labels, start_frame, end_frame):
    """Głosowanie większościowe dla etykiety Ground Truth."""
    counts = {0: 0, 1: 0, 2: 0}
    total_len = end_frame - start_frame
    counts[2] = total_len

    for label in labels:
        lbl_class = label['class']
        cls_id = 1 if lbl_class == 'inhale' else (0 if lbl_class == 'exhale' else 2)

        l_start, l_end = label['start'], label['end']
        if l_start < end_frame and l_end > start_frame:
            overlap = min(l_end, end_frame) - max(l_start, start_frame)
            if overlap > 0:
                counts[cls_id] += overlap
                counts[2] -= overlap
    return max(counts, key=counts.get)


# ==========================================
# GENEROWANIE WYKRESÓW
# ==========================================

def generate_comparison_plots(
        wav_path: str,
        csv_path: str,
        lstm_model_path: str,
        trans_config_path: str,
        trans_model_path: str,
        output_path: str,
        plot_name: str,
        chunk_duration: float = 0.25
) -> tuple[list[int], list[int], list[int]]:
    # Wczytanie danych
    audio_data, sample_rate = read_audio_file(wav_path)
    labels = read_labels(csv_path)

    # Inicjalizacja obu modeli
    lstm_wrapper = LSTMClassifierWrapper(lstm_model_path)
    trans_wrapper = TransformerClassifierWrapper(trans_config_path, trans_model_path)

    # Reset stanów (ważne dla LSTM i Bufora Transformera)
    lstm_wrapper.reset_state()
    trans_wrapper.reset_state()

    chunk_size = int(sample_rate * chunk_duration)

    print(f"  Audio: {len(audio_data) / sample_rate:.2f}s | Labels: {len(labels)}")

    # Przygotowanie wykresu z 3 subplotami
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(plot_name, fontsize=16)

    time_axis = np.arange(len(audio_data)) / sample_rate

    # --- PLOT 1: GROUND TRUTH ---
    ax1.set_title('Ground Truth', fontweight='bold')
    ax1.set_ylabel('Amplitude')
    ax1.plot(time_axis, audio_data, color='lightgray', alpha=0.5)  # Tło

    for label in labels:
        s, e = label['start'], label['end']
        c = get_color_for_class(label['class'])
        if s < len(audio_data) and e <= len(audio_data):
            ax1.plot(time_axis[s:e], audio_data[s:e], color=c)

    # --- KONFIGURACJA PLOT 2 i 3 ---
    ax2.set_title('LSTM Model Prediction', fontweight='bold')
    ax2.set_ylabel('Pred')
    # Rysujemy "szary cień" sygnału dla kontekstu
    ax2.plot(time_axis, audio_data, color='lightgray', alpha=0.3)

    ax3.set_title('Transformer Model Prediction', fontweight='bold')
    ax3.set_ylabel('Pred')
    ax3.set_xlabel('Time [s]')
    ax3.plot(time_axis, audio_data, color='lightgray', alpha=0.3)

    # Kontenery na wyniki dla tego pliku
    y_true = []
    y_pred_lstm = []
    y_pred_trans = []

    # --- PĘTLA PRZETWARZANIA ---
    for i in range(0, len(audio_data), chunk_size):
        end = min(i + chunk_size, len(audio_data))
        chunk = audio_data[i:end]

        if len(chunk) < chunk_size:
            continue

        # 1. Ground Truth
        gt = get_ground_truth_for_chunk(labels, i, end)
        y_true.append(gt)

        # 2. Predykcja LSTM
        p_lstm = lstm_wrapper.predict(chunk, sample_rate)
        y_pred_lstm.append(p_lstm)

        # 3. Predykcja Transformer
        p_trans = trans_wrapper.predict(chunk, sample_rate)
        y_pred_trans.append(p_trans)

        # Rysowanie na wykresach
        t_seg = time_axis[i:end]

        # LSTM Plot
        ax2.plot(t_seg, chunk, color=get_color_for_class(p_lstm))

        # Transformer Plot
        ax3.plot(t_seg, chunk, color=get_color_for_class(p_trans))

    # Legenda
    custom_lines = [
        Line2D([0], [0], color='green', lw=2),
        Line2D([0], [0], color='red', lw=2),
        Line2D([0], [0], color='blue', lw=2)
    ]
    fig.legend(custom_lines, ['Exhale (0)', 'Inhale (1)', 'Silence (2)'],
               loc='upper right', bbox_to_anchor=(0.95, 0.95))

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Miejsce na tytuł główny
    plt.savefig(output_path)
    plt.close(fig)
    print(f"  Saved plot: {output_path}")

    return y_true, y_pred_lstm, y_pred_trans


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    # --- KONFIGURACJA ŚCIEŻEK ---
    raw_folder = "../data/eval/raw"
    label_folder = "../data/eval/label"
    output_folder = "plots_comparison"

    # LSTM Paths
    lstm_model_path = "../archive/lstm/model_lstm.pth"

    # Transformer Paths (Dostosuj do swoich plików!)
    trans_config_path = "../model/transformer/config.yaml"
    trans_model_path = "../model/transformer/best_models/best_model_epoch_31.pth"

    os.makedirs(output_folder, exist_ok=True)
    wav_files = [f for f in os.listdir(raw_folder) if f.endswith('.wav')]
    print(f"Found {len(wav_files)} WAV files. Processing...")

    # Globalne kontenery na metryki
    all_y_true = []
    all_y_lstm = []
    all_y_trans = []

    target_names = ['Exhale (0)', 'Inhale (1)', 'Silence (2)']

    for wav_file in wav_files:
        base_name = os.path.splitext(wav_file)[0]
        csv_file = f"{base_name}.csv"

        paths = {
            'wav': os.path.join(raw_folder, wav_file),
            'csv': os.path.join(label_folder, csv_file),
            'out': os.path.join(output_folder, f"{base_name}_compare.png")
        }

        if not os.path.exists(paths['csv']):
            print(f"Skipping {wav_file} (no CSV found)")
            continue

        plot_title = f"Comparison: {base_name.replace('_', ', ')}"
        print(f"Processing: {base_name}")

        try:
            f_true, f_lstm, f_trans = generate_comparison_plots(
                paths['wav'], paths['csv'],
                lstm_model_path,
                trans_config_path, trans_model_path,
                paths['out'], plot_title
            )

            all_y_true.extend(f_true)
            all_y_lstm.extend(f_lstm)
            all_y_trans.extend(f_trans)

        except Exception as e:
            print(f"ERROR processing {base_name}: {e}")
            import traceback

            traceback.print_exc()

    # --- RAPORT KOŃCOWY ---
    print("\n" + "#" * 60)
    print("FINAL COMPARISON REPORT")
    print("#" * 60)

    if len(all_y_true) > 0:
        print("\n--- MODEL LSTM RESULTS ---")
        print(f"Global Accuracy: {accuracy_score(all_y_true, all_y_lstm):.4f}")
        print(classification_report(all_y_true, all_y_lstm, target_names=target_names, digits=4))

        print("\n--- MODEL TRANSFORMER RESULTS ---")
        print(f"Global Accuracy: {accuracy_score(all_y_true, all_y_trans):.4f}")
        print(classification_report(all_y_true, all_y_trans, target_names=target_names, digits=4))

        print("#" * 60)
        print(f"Processed {len(all_y_true)} chunks total.")
        print(f"Visualizations saved in: {output_folder}")
    else:
        print("No data processed.")