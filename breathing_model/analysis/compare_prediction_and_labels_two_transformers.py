import numpy as np
import matplotlib.pyplot as plt
import wave
import csv
import torch
import os
import sys
import seaborn as sns
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from breathing_model.model.exhale_only_detection.model import BreathPhaseTransformerSeq as TwoClassTransformer
from breathing_model.model.exhale_only_detection.utils import Config as TwoClassConfig
from breathing_model.model.transformer.utils import Config
from breathing_model.model.transformer.dataset import BreathDataset, collate_fn
from breathing_model.model.transformer.model import BreathPhaseTransformerSeq


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


def get_ground_truth_for_chunk_two_class(labels, start_frame, end_frame):
    counts = {0: 0, 1: 0}
    counts[1] = end_frame - start_frame  # Default other

    for label in labels:
        cls_id = 0 if label['class'] == 'exhale' else 1
        overlap_start = max(label['start'], start_frame)
        overlap_end = min(label['end'], end_frame)
        overlap = overlap_end - overlap_start
        if overlap > 0:
            counts[cls_id] += overlap
            counts[1] -= overlap

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
    TEST_RAW_FOLDER = "../data/eval/raw"
    TEST_LABEL_FOLDER = "../data/eval/label"
    THREE_CLASS_CONFIG_PATH = "../model/transformer/config.yaml"
    THREE_CLASS_MODEL_PATH = "../model/transformer/best_models/best_model_epoch_31.pth"
    TWO_CLASS_CONFIG_PATH = "../model/exhale_only_detection/config.yaml"
    TWO_CLASS_MODEL_PATH = "../model/exhale_only_detection/best_models/best_model_epoch_21.pth"
    OUTPUT_FOLDER = "plots_comparison_two_transformers"

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if not os.path.exists(THREE_CLASS_CONFIG_PATH):
        print(f"Brak pliku config: {THREE_CLASS_CONFIG_PATH}")
        sys.exit(1)
    if not os.path.exists(TWO_CLASS_CONFIG_PATH):
        print(f"Brak pliku config: {TWO_CLASS_CONFIG_PATH}")
        sys.exit(1)

    three_class_config = Config.from_yaml(THREE_CLASS_CONFIG_PATH)
    two_class_config = TwoClassConfig.from_yaml(TWO_CLASS_CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    WINDOW_TRANS = 0.20

    SAMPLES_TRANS = int(three_class_config.data.sample_rate * WINDOW_TRANS)
    HOP_LENGTH = three_class_config.data.hop_length

    WINDOW_INFERENCE = 10
    FRAMES_PER_INFERENCE = int(three_class_config.data.sample_rate * WINDOW_INFERENCE / HOP_LENGTH)

    print(">>> Przygotowanie danych testowych...")
    test_dataset = BreathDataset(
        data_dir=TEST_RAW_FOLDER,
        label_dir=TEST_LABEL_FOLDER,
        sample_rate=three_class_config.data.sample_rate,
        n_mels=three_class_config.data.n_mels,
        n_fft=three_class_config.data.n_fft,
        hop_length=three_class_config.data.hop_length,
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

    print(f">>> Ładowanie modelu Three-Class Transformer...")
    three_class_model = BreathPhaseTransformerSeq(
        n_mels=three_class_config.model.n_mels,
        d_model=three_class_config.model.d_model,
        nhead=three_class_config.model.nhead,
        num_layers=three_class_config.model.num_layers,
        num_classes=three_class_config.model.num_classes
    ).to(device)

    checkpoint = torch.load(THREE_CLASS_MODEL_PATH, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    if 'pos_encoder.pe' in state_dict: del state_dict['pos_encoder.pe']
    three_class_model.load_state_dict(state_dict, strict=False)
    three_class_model.eval()

    print(f">>> Ładowanie modelu Two-Class Transformer...")
    two_class_model = TwoClassTransformer(
        n_mels=two_class_config.model.n_mels,
        d_model=two_class_config.model.d_model,
        nhead=two_class_config.model.nhead,
        num_layers=two_class_config.model.num_layers,
        num_classes=two_class_config.model.num_classes
    ).to(device)

    checkpoint = torch.load(TWO_CLASS_MODEL_PATH, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    if 'pos_encoder.pe' in state_dict: del state_dict['pos_encoder.pe']
    two_class_model.load_state_dict(state_dict, strict=False)
    two_class_model.eval()

    global_metrics = {
        'two_class_true': [], 'two_class_pred': [],
        'three_class_true': [], 'three_class_pred': []
    }
    target_names_three = ['Wydech', 'Wdech', 'Cisza']  # 0, 1, 2
    target_names_two = ['Wydech', 'Inne']  # 0, 1

    print("\n>>> Rozpoczynanie pętli porównawczej...")

    with torch.no_grad():
        for i, (spectrogram, three_class_labels, padding_mask) in enumerate(test_loader):
            wav_filename = test_dataset.wav_files[i]
            base_name = os.path.splitext(wav_filename)[0]
            wav_path = os.path.join(TEST_RAW_FOLDER, wav_filename)
            csv_path = os.path.join(TEST_LABEL_FOLDER, f"{base_name}.csv")

            print(f"Processing: {wav_filename}")

            total_length = spectrogram.shape[-1]
            spectrogram = spectrogram.to(device)

            # Three-class predictions
            file_logits_list_three = []
            for start in range(0, total_length, FRAMES_PER_INFERENCE):
                end = min(start + FRAMES_PER_INFERENCE, total_length)
                chunk = spectrogram[..., start:end]
                chunk_len = end - start
                chunk_mask = torch.zeros((1, chunk_len), dtype=torch.bool).to(device)

                output = three_class_model(chunk, src_key_padding_mask=chunk_mask)
                file_logits_list_three.append(output)

            full_file_logits_three = torch.cat(file_logits_list_three, dim=1)
            valid_mask = ~padding_mask.to(device)
            raw_three_class_preds = torch.argmax(full_file_logits_three, dim=-1)[valid_mask].cpu().numpy()

            # Two-class predictions
            file_logits_list_two = []
            for start in range(0, total_length, FRAMES_PER_INFERENCE):
                end = min(start + FRAMES_PER_INFERENCE, total_length)
                chunk = spectrogram[..., start:end]
                chunk_len = end - start
                chunk_mask = torch.zeros((1, chunk_len), dtype=torch.bool).to(device)

                output = two_class_model(chunk, src_key_padding_mask=chunk_mask)
                file_logits_list_two.append(output)

            full_file_logits_two = torch.cat(file_logits_list_two, dim=1)
            raw_two_class_preds = torch.argmax(full_file_logits_two, dim=-1)[valid_mask].cpu().numpy()

            audio_data, sr = read_audio_file(wav_path)
            csv_labels = read_labels(csv_path)
            time_axis = np.arange(len(audio_data)) / sr

            for j in range(0, len(audio_data), SAMPLES_TRANS):
                end_sample = min(j + SAMPLES_TRANS, len(audio_data))
                if (end_sample - j) < SAMPLES_TRANS: continue

                gt_three_class = get_ground_truth_for_chunk(csv_labels, j, end_sample)
                gt_two_class = get_ground_truth_for_chunk_two_class(csv_labels, j, end_sample)

                start_frame = int(j / HOP_LENGTH)
                end_frame = int(end_sample / HOP_LENGTH)
                chunk_preds_three = raw_three_class_preds[start_frame:end_frame]
                chunk_preds_two = raw_two_class_preds[start_frame:end_frame]

                if len(chunk_preds_three) > 0:
                    pred_three = np.bincount(chunk_preds_three).argmax()
                else:
                    pred_three = 2

                if len(chunk_preds_two) > 0:
                    pred_two = np.bincount(chunk_preds_two).argmax()
                else:
                    pred_two = 1

                global_metrics['three_class_true'].append(gt_three_class)
                global_metrics['three_class_pred'].append(pred_three)
                global_metrics['two_class_true'].append(gt_two_class)
                global_metrics['two_class_pred'].append(pred_two)

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

            # --- PLOT 2: TWO-CLASS TRANSFORMER ---
            ax2.set_title(f'Two-Class Transformer Prediction', fontweight='bold')
            ax2.set_ylabel('Amplitude')
            ax2.plot(time_axis, audio_data, color='lightgray', alpha=0.3)

            for j in range(0, len(audio_data), SAMPLES_TRANS):
                end_sample = min(j + SAMPLES_TRANS, len(audio_data))
                if (end_sample - j) < SAMPLES_TRANS: continue

                start_frame = int(j / HOP_LENGTH)
                end_frame = int(end_sample / HOP_LENGTH)
                chunk_preds = raw_two_class_preds[start_frame:end_frame]

                if len(chunk_preds) > 0:
                    p = np.bincount(chunk_preds).argmax()
                else:
                    p = 1  # Other

                t_seg = time_axis[j:end_sample]
                # For two-class: 0 green, 1 blue
                color = 'green' if p == 0 else 'blue'
                ax2.plot(t_seg, audio_data[j:end_sample], color=color)

            # --- PLOT 3: THREE-CLASS TRANSFORMER ---
            ax3.set_title(f'Three-Class Transformer Prediction', fontweight='bold')
            ax3.set_ylabel('Amplitude')
            ax3.set_xlabel('Time [s]')
            ax3.plot(time_axis, audio_data, color='lightgray', alpha=0.3)

            for j in range(0, len(audio_data), SAMPLES_TRANS):
                end_sample = min(j + SAMPLES_TRANS, len(audio_data))
                if (end_sample - j) < SAMPLES_TRANS: continue

                start_frame = int(j / HOP_LENGTH)
                end_frame = int(end_sample / HOP_LENGTH)
                chunk_preds = raw_three_class_preds[start_frame:end_frame]

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

    if len(global_metrics['two_class_true']) > 0:
        # TWO-CLASS
        print(f"\n[MODEL TWO-CLASS TRANSFORMER] (Aggregated Window: {WINDOW_TRANS}s)")
        print(f"Global Accuracy: {accuracy_score(global_metrics['two_class_true'], global_metrics['two_class_pred']):.4f}")
        print(classification_report(global_metrics['two_class_true'], global_metrics['two_class_pred'], target_names=target_names_two,
                                    digits=4))

        save_confusion_matrix(
            global_metrics['two_class_true'],
            global_metrics['two_class_pred'],
            target_names_two,
            f"Macierz pomyłek: Two-Class Transformer",
            "confusion_matrix_two_class.png",
            OUTPUT_FOLDER
        )

        # THREE-CLASS
        print(f"\n[MODEL THREE-CLASS TRANSFORMER] (Aggregated Window: {WINDOW_TRANS}s)")
        print(f"Global Accuracy: {accuracy_score(global_metrics['three_class_true'], global_metrics['three_class_pred']):.4f}")
        print(
            classification_report(global_metrics['three_class_true'], global_metrics['three_class_pred'], target_names=target_names_three,
                                  digits=4))

        save_confusion_matrix(
            global_metrics['three_class_true'],
            global_metrics['three_class_pred'],
            target_names_three,
            f"Macierz pomyłek: Three-Class Transformer",
            "confusion_matrix_three_class.png",
            OUTPUT_FOLDER
        )
    else:
        print("Brak danych.")