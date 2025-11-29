import numpy as np
import matplotlib.pyplot as plt
import wave
import csv
import torch
import librosa
import os
from matplotlib.lines import Line2D
from sklearn.metrics import classification_report, accuracy_score

from breathing_model.archive.lstm.model_classes import AudioClassifierLSTM

from breathing_model.model.transformer.utils import Config
from breathing_model.model.transformer.inference.transform import MelSpectrogramTransform
from breathing_model.model.transformer.inference.model_loader import BreathPhaseClassifier
from breathing_model.model.transformer.inference.audio_buffer import AudioBuffer

class LSTMClassifierWrapper:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AudioClassifierLSTM()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.hidden = None

    def reset_state(self):
        """LSTM: Reset stanu ukrytego na początku nowego pliku."""
        self.hidden = None

    def predict(self, audio_chunk: np.ndarray, sample_rate: int = 44100) -> int:
        if audio_chunk.dtype != np.float32:
            raise ValueError("Audio chunk must be of type float32")

        mfcc = librosa.feature.mfcc(y=audio_chunk, sr=sample_rate, n_mfcc=13)
        features = mfcc.mean(axis=1)

        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs, self.hidden = self.model(input_tensor, self.hidden)
            logits = outputs[0, 0]
            return torch.argmax(logits).item()


class TransformerClassifierWrapper:
    def __init__(self, config_path, model_path):
        self.config = Config.from_yaml(config_path)
        self.mel_transform = MelSpectrogramTransform(self.config.data)
        self.classifier = BreathPhaseClassifier(model_path, self.config.model, self.config.data)

        self.sample_rate = self.config.audio.sample_rate
        self.chunk_length_s = self.config.audio.chunk_length
        self.chunk_size = int(self.sample_rate * self.chunk_length_s)

        self.buffer = AudioBuffer(self.sample_rate, 3.5)

    def reset_state(self):
        self.buffer = AudioBuffer(self.sample_rate, 3.5)

    def predict(self, audio_chunk: np.ndarray) -> int:
        if audio_chunk.dtype != np.float32:
            raise ValueError("Audio chunk must be of type float32")

        self.buffer.append(audio_chunk)
        buf_audio = self.buffer.get()

        mel = self.mel_transform(buf_audio)
        pred_class, _ = self.classifier.predict(mel)

        return pred_class

def read_audio_file(wav_path: str) -> tuple[np.ndarray, int]:
    with wave.open(wav_path, 'rb') as wf:
        n_frames = wf.getnframes()
        audio_data = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
        audio_data = audio_data.astype(np.float32, order='C') / 32768.0

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
    if class_name == 'exhale' or class_name == 0:
        return 'green'
    elif class_name == 'inhale' or class_name == 1:
        return 'red'
    return 'blue'


def get_ground_truth_for_chunk(labels, start_frame, end_frame):
    counts = {0: 0, 1: 0, 2: 0}
    counts[2] = end_frame - start_frame  # Default silence

    for label in labels:
        cls_id = 1 if label['class'] == 'inhale' else (0 if label['class'] == 'exhale' else 2)

        # Intersection
        overlap_start = max(label['start'], start_frame)
        overlap_end = min(label['end'], end_frame)
        overlap = overlap_end - overlap_start

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
        plot_name: str
) -> dict:
    # Wczytanie danych
    audio_data, sample_rate = read_audio_file(wav_path)
    labels = read_labels(csv_path)

    # Inicjalizacja modeli
    lstm_wrapper = LSTMClassifierWrapper(lstm_model_path)
    trans_wrapper = TransformerClassifierWrapper(trans_config_path, trans_model_path)

    # WAŻNE: Określenie różnych chunków dla modeli
    # LSTM: sztywne 0.25s
    chunk_size_lstm = int(sample_rate * 0.25)
    # Transformer: dynamiczne z configu (0.2s)
    chunk_size_trans = trans_wrapper.chunk_size

    print(f"  Audio: {len(audio_data) / sample_rate:.2f}s | Labels: {len(labels)}")
    print(
        f"  Chunk sizes -> LSTM: {chunk_size_lstm} samples (0.25s), Transformer: {chunk_size_trans} samples ({trans_wrapper.chunk_length_s}s)")

    # Przygotowanie wykresu
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(plot_name, fontsize=16)

    time_axis = np.arange(len(audio_data)) / sample_rate

    # --- PLOT 1: GROUND TRUTH ---
    ax1.set_title('Ground Truth', fontweight='bold')
    ax1.set_ylabel('Amplitude')
    ax1.plot(time_axis, audio_data, color='lightgray', alpha=0.5)

    for label in labels:
        s, e = label['start'], label['end']
        c = get_color_for_class(label['class'])
        if s < len(audio_data) and e <= len(audio_data):
            ax1.plot(time_axis[s:e], audio_data[s:e], color=c)

    # --- KONFIGURACJA PLOT 2 i 3 ---
    ax2.set_title(f'LSTM Prediction (Window: 0.25s)', fontweight='bold')
    ax2.set_ylabel('Pred')
    ax2.plot(time_axis, audio_data, color='lightgray', alpha=0.3)

    ax3.set_title(f'Transformer Prediction (Window: {trans_wrapper.chunk_length_s}s + Buffer)', fontweight='bold')
    ax3.set_ylabel('Pred')
    ax3.set_xlabel('Time [s]')
    ax3.plot(time_axis, audio_data, color='lightgray', alpha=0.3)

    # Kontenery na wyniki (osobne dla każdego modelu!)
    results = {
        'lstm_true': [], 'lstm_pred': [],
        'trans_true': [], 'trans_pred': []
    }

    # ==============================
    # PĘTLA 1: LSTM (0.25s stride)
    # ==============================
    lstm_wrapper.reset_state()
    for i in range(0, len(audio_data), chunk_size_lstm):
        end = min(i + chunk_size_lstm, len(audio_data))
        chunk = audio_data[i:end]

        if len(chunk) < chunk_size_lstm: continue

        # Ground Truth dla okna 0.25s
        gt = get_ground_truth_for_chunk(labels, i, end)
        results['lstm_true'].append(gt)

        # Predykcja
        pred = lstm_wrapper.predict(chunk, sample_rate)
        results['lstm_pred'].append(pred)

        # Rysowanie
        t_seg = time_axis[i:end]
        ax2.plot(t_seg, chunk, color=get_color_for_class(pred))

    # ==============================
    # PĘTLA 2: TRANSFORMER (0.20s stride)
    # ==============================
    trans_wrapper.reset_state()
    for i in range(0, len(audio_data), chunk_size_trans):
        end = min(i + chunk_size_trans, len(audio_data))
        chunk = audio_data[i:end]

        # Symulacja Real-Time: Jeśli chunk jest niepełny, czekamy (tutaj pomijamy)
        if len(chunk) < chunk_size_trans: continue

        # Ground Truth dla okna 0.2s
        gt = get_ground_truth_for_chunk(labels, i, end)
        results['trans_true'].append(gt)

        # Predykcja (z buforowaniem wewnątrz wrappera)
        pred = trans_wrapper.predict(chunk)
        results['trans_pred'].append(pred)

        # Rysowanie
        t_seg = time_axis[i:end]
        ax3.plot(t_seg, chunk, color=get_color_for_class(pred))

    # Legenda
    custom_lines = [
        Line2D([0], [0], color='green', lw=2),
        Line2D([0], [0], color='red', lw=2),
        Line2D([0], [0], color='blue', lw=2)
    ]
    fig.legend(custom_lines, ['Exhale (0)', 'Inhale (1)', 'Silence (2)'],
               loc='upper right', bbox_to_anchor=(0.95, 0.95))

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(output_path)
    plt.close(fig)
    print(f"  Saved plot: {output_path}")

    return results


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    # --- KONFIGURACJA ŚCIEŻEK ---
    raw_folder = "../data/eval/raw"
    label_folder = "../data/eval/label"
    output_folder = "plots_comparison_final"

    lstm_model_path = "../archive/lstm/model_lstm.pth"
    trans_config_path = "../model/transformer/config.yaml"
    trans_model_path = "../model/transformer/best_models/best_model_epoch_31.pth"

    os.makedirs(output_folder, exist_ok=True)
    wav_files = [f for f in os.listdir(raw_folder) if f.endswith('.wav')]
    print(f"Found {len(wav_files)} WAV files. Processing...")

    # Globalne kontenery (agregacja wszystkich plików)
    global_metrics = {
        'lstm_true': [], 'lstm_pred': [],
        'trans_true': [], 'trans_pred': []
    }

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
            res = generate_comparison_plots(
                paths['wav'], paths['csv'],
                lstm_model_path,
                trans_config_path, trans_model_path,
                paths['out'], plot_title
            )

            # Agregacja wyników
            global_metrics['lstm_true'].extend(res['lstm_true'])
            global_metrics['lstm_pred'].extend(res['lstm_pred'])
            global_metrics['trans_true'].extend(res['trans_true'])
            global_metrics['trans_pred'].extend(res['trans_pred'])

        except Exception as e:
            print(f"ERROR processing {base_name}: {e}")
            import traceback

            traceback.print_exc()

    # --- RAPORT KOŃCOWY ---
    print("\n" + "=" * 60)
    print("FINAL COMPARISON REPORT (DIFFERENT TIME BASES)")
    print("=" * 60)

    if len(global_metrics['lstm_true']) > 0:
        print(f"\n[MODEL LSTM] (Window: 0.25s, Total predictions: {len(global_metrics['lstm_true'])})")
        print(f"Global Accuracy: {accuracy_score(global_metrics['lstm_true'], global_metrics['lstm_pred']):.4f}")
        print(classification_report(global_metrics['lstm_true'], global_metrics['lstm_pred'], target_names=target_names,
                                    digits=4))

        print(f"\n[MODEL TRANSFORMER] (Window: 0.20s, Total predictions: {len(global_metrics['trans_true'])})")
        print(f"Global Accuracy: {accuracy_score(global_metrics['trans_true'], global_metrics['trans_pred']):.4f}")
        print(
            classification_report(global_metrics['trans_true'], global_metrics['trans_pred'], target_names=target_names,
                                  digits=4))

        print("=" * 60)
        print(f"Plots saved in: {output_folder}")
    else:
        print("No data processed.")