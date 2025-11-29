import os
import numpy as np
import torchaudio
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import deque
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from breathing_model.model.transformer.utils import Config, BreathType
from breathing_model.model.transformer.inference.transform import MelSpectrogramTransform
from breathing_model.model.transformer.inference.model_loader import BreathPhaseClassifier

CONFIG_PATH = 'config.yaml'
MODEL_PATH = 'best_models/best_model_epoch_31.pth'
WAV_DIR_OVERRIDE = '../../data/eval/raw'
LABEL_DIR_OVERRIDE = '../../data/eval/label'
OUTPUT_DIR = 'offline_plots'
BUFFER_SECONDS = 3.5
LIMIT_WAV = None
MIN_CHUNK_SAMPLES_SKIP = 0
NORMALIZE_AUDIO_FOR_PLOT = False
PRINT_PROBS = False


def load_audio(wav_path: str, target_sr: int) -> np.ndarray:
    """Load WAV file, convert to mono float32 numpy array at target_sr."""
    waveform, sr = torchaudio.load(wav_path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0).numpy().astype(np.float32)


def parse_label_csv(csv_path: str) -> list[dict]:
    """Parse CSV with columns: class,start_sample,end_sample"""
    import csv
    labels = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        _ = next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            cls, start_s, end_s = row[0], row[1], row[2]
            try:
                start_i = int(start_s)
                end_i = int(end_s)
            except ValueError:
                continue
            labels.append({'class': cls.strip().lower(), 'start': start_i, 'end': end_i})
    return labels


def label_to_breath_type(label: str) -> BreathType:
    """Map dataset labels to BreathType (EXHALE / OTHER)."""
    if label == 'exhale':
        return BreathType.EXHALE
    if label == 'inhale':
        return BreathType.INHALE
    if label == 'silence':
        return BreathType.SILENCE
    return BreathType.SILENCE


def get_color(bt: BreathType) -> str:
    return bt.get_color()


def get_gt_for_chunk(labels: list[dict], start: int, end: int) -> int:
    """Głosowanie większościowe dla chunka."""
    counts = {int(BreathType.EXHALE): 0, int(BreathType.INHALE): 0, int(BreathType.SILENCE): 0}
    counts[int(BreathType.SILENCE)] = end - start

    for lab in labels:
        l_type = label_to_breath_type(lab['class'])
        cls_id = int(l_type)
        overlap_start = max(lab['start'], start)
        overlap_end = min(lab['end'], end)

        if overlap_end > overlap_start:
            length = overlap_end - overlap_start
            counts[cls_id] += length
            counts[int(BreathType.SILENCE)] -= length

    return max(counts, key=counts.get)


def process_file(wav_path: str,
                 csv_path: str,
                 classifier: BreathPhaseClassifier,
                 mel_transform: MelSpectrogramTransform,
                 config: Config,
                 buffer_seconds: float,
                 output_dir: str) -> tuple[list[int], list[int]]:
    base_name = os.path.splitext(os.path.basename(wav_path))[0]

    audio = load_audio(wav_path, config.data.sample_rate)
    n_samples = audio.shape[0]
    sr = config.data.sample_rate

    if NORMALIZE_AUDIO_FOR_PLOT and np.max(np.abs(audio)) > 0:
        audio_for_plot = audio / np.max(np.abs(audio))
    else:
        audio_for_plot = audio

    labels = parse_label_csv(csv_path)

    chunk_size = int(config.audio.chunk_length * sr)
    max_buffer = int(buffer_seconds * sr)
    audio_buffer = deque(maxlen=max_buffer)

    predicted_segments: list[dict] = []

    file_y_true = []
    file_y_pred = []

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk = audio[start:end]

        if chunk.size < chunk_size:
            continue

        if chunk.size == 0 or chunk.size < MIN_CHUNK_SAMPLES_SKIP:
            continue

        audio_buffer.extend(chunk)
        buf_np = np.array(audio_buffer, dtype=np.float32)
        if buf_np.size == 0:
            continue

        mel = mel_transform(buf_np)
        pred_cls, probs = classifier.predict(mel)
        predicted_segments.append({'start': start, 'end': end, 'pred': pred_cls, 'probs': probs})

        true_cls = get_gt_for_chunk(labels, start, end)
        file_y_true.append(true_cls)
        file_y_pred.append(int(pred_cls))

        if PRINT_PROBS:
            print(f"Chunk {start}:{end} -> pred={BreathType(pred_cls).name} probs={probs.round(3)}")

    gt_segments: list[dict] = []
    for lab in labels:
        s = max(0, lab['start'])
        e = min(n_samples, lab['end'])
        if e <= s:
            continue
        gt_segments.append({'start': s, 'end': e, 'cls': label_to_breath_type(lab['class'])})

    time_axis = np.arange(n_samples) / sr

    fig, (ax_gt, ax_pred) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    fig.suptitle(f"Offline comparison (transformer): {base_name}")

    ax_gt.set_title('Ground truth labels')
    ax_gt.set_ylabel('Amplitude')
    for seg in gt_segments:
        ax_gt.plot(time_axis[seg['start']:seg['end']],
                   audio_for_plot[seg['start']:seg['end']],
                   color=get_color(seg['cls']))

    ax_pred.set_title('Model predictions (streaming simulation)')
    ax_pred.set_xlabel('Time [s]')
    ax_pred.set_ylabel('Amplitude')
    for seg in predicted_segments:
        try:
            btype = BreathType(seg['pred'])
        except ValueError:
            btype = BreathType.SILENCE
        ax_pred.plot(time_axis[seg['start']:seg['end']],
                     audio_for_plot[seg['start']:seg['end']],
                     color=get_color(btype))

    custom_lines = [
        Line2D([0], [0], color=BreathType.EXHALE.get_color(), lw=2),
        Line2D([0], [0], color=BreathType.INHALE.get_color(), lw=2),
        Line2D([0], [0], color=BreathType.SILENCE.get_color(), lw=2),
    ]
    fig.legend(custom_lines, ['Exhale', 'Inhale', 'Silence'], loc='lower center', ncol=3)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.13)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{base_name}_comparison.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved plot: {out_path}")

    return file_y_true, file_y_pred


def run():
    config = Config.from_yaml(CONFIG_PATH)
    wav_dir = WAV_DIR_OVERRIDE or config.data.data_dir
    label_dir = LABEL_DIR_OVERRIDE or config.data.label_dir

    mel_transform = MelSpectrogramTransform(config.data)
    classifier = BreathPhaseClassifier(MODEL_PATH, config.model, config.data)

    wav_files = sorted([f for f in os.listdir(wav_dir) if f.lower().endswith('.wav')])
    if LIMIT_WAV:
        wav_files = wav_files[:LIMIT_WAV]

    print(f"Found {len(wav_files)} wav files in {wav_dir}")

    all_y_true = []
    all_y_pred = []

    for i, wav_name in enumerate(wav_files, start=1):
        base = os.path.splitext(wav_name)[0]
        wav_path = os.path.join(wav_dir, wav_name)
        csv_path = os.path.join(label_dir, f"{base}.csv")
        if not os.path.exists(csv_path):
            print(f"[Skip] Missing label file for {wav_name}: {csv_path}")
            continue

        print(f"[{i}/{len(wav_files)}] Processing {wav_name}...", end=" ")
        try:
            file_true, file_pred = process_file(wav_path, csv_path, classifier, mel_transform, config, BUFFER_SECONDS,
                                                OUTPUT_DIR)

            # --- DODANO: Wypisanie Accuracy dla pliku ---
            if len(file_true) > 0:
                file_acc = accuracy_score(file_true, file_pred)
                print(f"File Acc: {file_acc:.2%}")
            else:
                print("No chunks processed.")
            # --------------------------------------------

            all_y_true.extend(file_true)
            all_y_pred.extend(file_pred)
        except Exception as e:
            print(f"Error processing {wav_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("FINAL EVALUATION STATISTICS")
    print("=" * 60)

    if len(all_y_true) > 0:
        target_names = ['Exhale (0)', 'Inhale (1)', 'Silence (2)']

        acc = accuracy_score(all_y_true, all_y_pred)
        print(f"Total Chunks: {len(all_y_true)}")
        print(f"Global Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
        print("-" * 60)
        print(classification_report(all_y_true, all_y_pred, target_names=target_names, digits=4))

        print("Confusion Matrix:")
        print(confusion_matrix(all_y_true, all_y_pred))
    else:
        print("No data processed.")

    print("Done.")


if __name__ == '__main__':
    run()