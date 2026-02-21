import numpy as np
import matplotlib.pyplot as plt
import wave
import csv
import torch
import librosa
import os
import sys
import seaborn as sns
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from breathing_model.archive.lstm.model_classes import AudioClassifierLSTM
from breathing_model.model.transformer.utils import Config
from breathing_model.model.transformer.dataset import BreathDataset, collate_fn
from breathing_model.model.transformer.model import BreathPhaseTransformerSeq


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
        if audio_chunk.dtype != np.float32:
            raise ValueError("Audio chunk must be of type float32")

        mfcc = librosa.feature.mfcc(y=audio_chunk, sr=sample_rate, n_mfcc=13)
        features = mfcc.mean(axis=1)
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs, self.hidden = self.model(input_tensor, self.hidden)
            logits = outputs[0, 0]
            return torch.argmax(logits).item()


def read_audio_file(wav_path: str) -> tuple[np.ndarray, int]:
    with wave.open(wav_path, 'rb') as wf:
        n_frames = wf.getnframes()
        audio_data = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
        audio_data = audio_data.astype(np.float32, order='C') / 32768.0
        frame_rate = wf.getframerate()
    return audio_data, frame_rate


def read_labels(csv_path: str) -> list[dict[str, int | str]]:
    labels = []
    if not os.path.exists(csv_path):
        return []
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
        overlap_start = max(label['start'], start_frame)
        overlap_end = min(label['end'], end_frame)
        overlap = overlap_end - overlap_start
        if overlap > 0:
            counts[cls_id] += overlap
            counts[2] -= overlap

    return max(counts, key=counts.get)


def save_confusion_matrix(y_true, y_pred, classes, title, filename, output_folder):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=classes, yticklabels=classes, annot_kws={"size": 14})
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('Rzeczywista klasa (Ground Truth)', fontsize=12)
    plt.xlabel('Przewidziana klasa (Prediction)', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"  Saved confusion matrix: {filepath}")


if __name__ == "__main__":
    TEST_RAW_FOLDER = "../data/eval2/raw"
    TEST_LABEL_FOLDER = "../data/eval2/label"
    CONFIG_PATH = "../model/transformer/config.yaml"
    TRANSFORMER_MODEL_PATH = "../model/transformer/best_models/best_model_epoch_31.pth"
    LSTM_MODEL_PATH = "../archive/lstm/model_lstm.pth"
    OUTPUT_FOLDER = "plots_comparison_ppl_from_train"

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if not os.path.exists(CONFIG_PATH):
        print(f"Brak pliku config: {CONFIG_PATH}")
        sys.exit(1)

    config = Config.from_yaml(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    WINDOW_LSTM = 0.25
    WINDOW_TRANS = 0.20

    SAMPLES_LSTM = int(config.data.sample_rate * WINDOW_LSTM)
    SAMPLES_TRANS = int(config.data.sample_rate * WINDOW_TRANS)

    HOP_LENGTH = config.data.hop_length

    WINDOW_INFERENCE = 10
    FRAMES_PER_INFERENCE = int(config.data.sample_rate * WINDOW_INFERENCE / HOP_LENGTH)

    print(">>> Przygotowanie danych testowych...")
    test_dataset = BreathDataset(
        data_dir=TEST_RAW_FOLDER,
        label_dir=TEST_LABEL_FOLDER,
        sample_rate=config.data.sample_rate,
        n_mels=config.data.n_mels,
        n_fft=config.data.n_fft,
        hop_length=config.data.hop_length,
        augment=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )

    print(f">>> Ładowanie modelu Transformer...")
    trans_model = BreathPhaseTransformerSeq(
        n_mels=config.model.n_mels,
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_layers=config.model.num_layers,
        num_classes=config.model.num_classes
    ).to(device)

    checkpoint = torch.load(TRANSFORMER_MODEL_PATH, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    if 'pos_encoder.pe' in state_dict: del state_dict['pos_encoder.pe']
    trans_model.load_state_dict(state_dict, strict=False)
    trans_model.eval()

    print(f">>> Ładowanie modelu LSTM...")
    lstm_wrapper = LSTMClassifierWrapper(LSTM_MODEL_PATH)

    global_metrics = {
        'lstm_true': [], 'lstm_pred': [],
        'trans_true': [], 'trans_pred': []
    }
    target_names = ['Wydech', 'Wdech', 'Cisza']  # 0, 1, 2

    print("\n>>> Rozpoczynanie pętli porównawczej...")

    with torch.no_grad():
        for i, (spectrogram, trans_labels, padding_mask) in enumerate(test_loader):
            wav_filename = test_dataset.wav_files[i]
            base_name = os.path.splitext(wav_filename)[0]
            wav_path = os.path.join(TEST_RAW_FOLDER, wav_filename)
            csv_path = os.path.join(TEST_LABEL_FOLDER, f"{base_name}.csv")

            print(f"Processing: {wav_filename}")

            total_length = spectrogram.shape[-1]
            spectrogram = spectrogram.to(device)
            file_logits_list = []

            for start in range(0, total_length, FRAMES_PER_INFERENCE):
                end = min(start + FRAMES_PER_INFERENCE, total_length)
                chunk = spectrogram[..., start:end]
                chunk_len = end - start
                chunk_mask = torch.zeros((1, chunk_len), dtype=torch.bool).to(device)

                output = trans_model(chunk, src_key_padding_mask=chunk_mask)
                file_logits_list.append(output)

            full_file_logits = torch.cat(file_logits_list, dim=1)
            valid_mask = ~padding_mask.to(device)

            raw_trans_preds = torch.argmax(full_file_logits, dim=-1)[valid_mask].cpu().numpy()

            audio_data, sr = read_audio_file(wav_path)
            csv_labels = read_labels(csv_path)
            time_axis = np.arange(len(audio_data)) / sr

            lstm_wrapper.reset_state()
            for j in range(0, len(audio_data), SAMPLES_LSTM):
                end_sample = min(j + SAMPLES_LSTM, len(audio_data))
                if (end_sample - j) < SAMPLES_LSTM: continue

                gt_class = get_ground_truth_for_chunk(csv_labels, j, end_sample)
                chunk_audio = audio_data[j:end_sample]
                pred_lstm = lstm_wrapper.predict(chunk_audio, sr)

                global_metrics['lstm_true'].append(gt_class)
                global_metrics['lstm_pred'].append(pred_lstm)

            for j in range(0, len(audio_data), SAMPLES_TRANS):
                end_sample = min(j + SAMPLES_TRANS, len(audio_data))
                if (end_sample - j) < SAMPLES_TRANS: continue

                gt_class = get_ground_truth_for_chunk(csv_labels, j, end_sample)

                start_frame = int(j / HOP_LENGTH)
                end_frame = int(end_sample / HOP_LENGTH)
                chunk_preds = raw_trans_preds[start_frame:end_frame]

                if len(chunk_preds) > 0:
                    pred_trans = np.bincount(chunk_preds).argmax()
                else:
                    pred_trans = 2

                global_metrics['trans_true'].append(gt_class)
                global_metrics['trans_pred'].append(pred_trans)

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
            plot_title = "Typowy mikrofon" if "medium" in base_name else "Dobry mikrofon"
            fig.suptitle(plot_title, fontsize=16)

            # --- PLOT 1: GROUND TRUTH ---
            ax1.set_title('Ground Truth', fontweight='bold')
            ax1.set_ylabel('Amplitude')
            ax1.plot(time_axis, audio_data, color='lightgray', alpha=0.5)
            for label in csv_labels:
                s, e = label['start'], label['end']
                c = get_color_for_class(label['class'])
                if s < len(audio_data) and e <= len(audio_data):
                    ax1.plot(time_axis[s:e], audio_data[s:e], color=c)

            # --- PLOT 2: LSTM ---
            ax2.set_title(f'LSTM Prediction', fontweight='bold')
            ax2.set_ylabel('Amplitude')
            ax2.plot(time_axis, audio_data, color='lightgray', alpha=0.3)

            lstm_wrapper.reset_state()
            for j in range(0, len(audio_data), SAMPLES_LSTM):
                end_sample = min(j + SAMPLES_LSTM, len(audio_data))
                chunk_audio = audio_data[j:end_sample]
                if len(chunk_audio) < SAMPLES_LSTM: continue
                p = lstm_wrapper.predict(chunk_audio, sr)
                t_seg = time_axis[j:end_sample]
                ax2.plot(t_seg, chunk_audio, color=get_color_for_class(p))

            # --- PLOT 3: TRANSFORMER (Blokowy 0.20s) ---
            ax3.set_title(f'Transformer Prediction', fontweight='bold')
            ax3.set_ylabel('Amplitude')
            ax3.set_xlabel('Time [s]')
            ax3.plot(time_axis, audio_data, color='lightgray', alpha=0.3)

            for j in range(0, len(audio_data), SAMPLES_TRANS):
                end_sample = min(j + SAMPLES_TRANS, len(audio_data))
                if (end_sample - j) < SAMPLES_TRANS: continue

                start_frame = int(j / HOP_LENGTH)
                end_frame = int(end_sample / HOP_LENGTH)
                chunk_preds = raw_trans_preds[start_frame:end_frame]

                if len(chunk_preds) > 0:
                    p = np.bincount(chunk_preds).argmax()
                else:
                    p = 2  # Silence

                t_seg = time_axis[j:end_sample]
                color = get_color_for_class(p)
                ax3.plot(t_seg, audio_data[j:end_sample], color=color)

            custom_lines = [
                Line2D([0], [0], color='green', lw=2),
                Line2D([0], [0], color='red', lw=2),
                Line2D([0], [0], color='blue', lw=2)
            ]
            fig.legend(custom_lines, ['Wydech (0)', 'Wdech (1)', 'Cisza (2)'],
                       loc='upper right', bbox_to_anchor=(0.95, 0.95))

            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_compare.png")
            plt.savefig(output_path)
            plt.close(fig)
            print(f"  Saved plot: {output_path}")

    print("\n" + "=" * 60)
    print("FINAL COMPARISON REPORT")
    print("=" * 60)

    if len(global_metrics['lstm_true']) > 0:
        # LSTM
        print(f"\n[MODEL LSTM] (Window: {WINDOW_LSTM}s)")
        print(f"Global Accuracy: {accuracy_score(global_metrics['lstm_true'], global_metrics['lstm_pred']):.4f}")
        print(classification_report(global_metrics['lstm_true'], global_metrics['lstm_pred'], target_names=target_names,
                                    digits=4))

        save_confusion_matrix(
            global_metrics['lstm_true'],
            global_metrics['lstm_pred'],
            target_names,
            f"Macierz pomyłek: LSTM",
            "confusion_matrix_lstm.png",
            OUTPUT_FOLDER
        )

        # TRANSFORMER
        print(f"\n[MODEL TRANSFORMER] (Aggregated Window: {WINDOW_TRANS}s)")
        print(f"Global Accuracy: {accuracy_score(global_metrics['trans_true'], global_metrics['trans_pred']):.4f}")
        print(
            classification_report(global_metrics['trans_true'], global_metrics['trans_pred'], target_names=target_names,
                                  digits=4))

        save_confusion_matrix(
            global_metrics['trans_true'],
            global_metrics['trans_pred'],
            target_names,
            f"Macierz pomyłek: Transformer)",
            "confusion_matrix_transformer.png",
            OUTPUT_FOLDER
        )
    else:
        print("Brak danych.")