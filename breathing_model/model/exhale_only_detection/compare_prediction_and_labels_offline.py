import os
import numpy as np
import torchaudio
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import deque

from breathing_model.model.exhale_only_detection.utils import Config, BreathType
from breathing_model.model.transformer.inference.transform import MelSpectrogramTransform
from breathing_model.model.exhale_only_detection.inference.model_loader import BreathPhaseClassifier

CONFIG_PATH = 'config.yaml'
MODEL_PATH = 'best_models/best_model_epoch_21.pth'
WAV_DIR_OVERRIDE = '../../data/eval/raw'
LABEL_DIR_OVERRIDE = '../../data/eval/label'
OUTPUT_DIR = 'offline_plots'
BUFFER_SECONDS = 3.5
LIMIT_WAV = None
MIN_CHUNK_SAMPLES_SKIP = 0

def load_audio(wav_path: str, target_sr: int) -> np.ndarray:
    """Load WAV file, convert to mono float32 numpy array at target_sr."""
    waveform, sr = torchaudio.load(wav_path)  # waveform: [channels, samples], float32 in [-1,1]
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
        _ = next(reader)  # skip header
        for row in reader:
            if len(row) < 3:
                continue
            cls, start_s, end_s = row[0], row[1], row[2]
            try:
                start_i = int(start_s)
                end_i = int(end_s)
            except ValueError:
                continue
            labels.append({
                'class': cls.strip().lower(),
                'start': start_i,
                'end': end_i
            })
    return labels


def label_to_breath_type(label: str) -> BreathType:
    """Map dataset labels to BreathType (EXHALE / OTHER)."""
    if label == 'exhale':
        return BreathType.EXHALE
    # inhale / silence / anything else -> OTHER
    return BreathType.OTHER


def get_color(btype: BreathType) -> str:
    return btype.get_color()


def process_file(wav_path: str,
                 csv_path: str,
                 classifier: BreathPhaseClassifier,
                 mel_transform: MelSpectrogramTransform,
                 config: Config,
                 buffer_seconds: float,
                 output_dir: str) -> None:
    base_name = os.path.splitext(os.path.basename(wav_path))[0]

    # 1. Load audio
    audio = load_audio(wav_path, config.data.sample_rate)
    n_samples = audio.shape[0]
    sr = config.data.sample_rate

    # 2. Load labels
    labels = parse_label_csv(csv_path)

    # 3. Prepare streaming-like iteration (simulate realtime main.py logic)
    chunk_size = int(config.audio.chunk_length * sr)
    max_buffer = int(buffer_seconds * sr)
    audio_buffer = deque(maxlen=max_buffer)

    predicted_segments: list[dict] = []  # each: {start,end,pred}

    # Iterate over file in chunks of chunk_size (like reading microphone chunks)
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk = audio[start:end]
        if chunk.size == 0 or chunk.size < MIN_CHUNK_SAMPLES_SKIP:
            continue
        # Append to buffer
        audio_buffer.extend(chunk)
        buf_np = np.array(audio_buffer, dtype=np.float32)
        if buf_np.size == 0:
            continue
        # Mel spectrogram & classification (same as in main.py)
        mel = mel_transform(buf_np)
        pred_cls, _ = classifier.predict(mel)
        predicted_segments.append({'start': start, 'end': end, 'pred': pred_cls})

    # 4. Create ground-truth segments mapped to EXHALE/OTHER
    gt_segments: list[dict] = []
    for lab in labels:
        s = max(0, lab['start'])
        e = min(n_samples, lab['end'])
        if e <= s:
            continue
        cls_bt = label_to_breath_type(lab['class'])
        gt_segments.append({'start': s, 'end': e, 'cls': cls_bt})

    # 5. Plotting
    time_axis = np.arange(n_samples) / sr

    fig, (ax_gt, ax_pred) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    fig.suptitle(f"Offline comparison: {base_name}")

    # Ground truth plot
    ax_gt.set_title('Ground truth labels')
    ax_gt.set_ylabel('Amplitude')
    for seg in gt_segments:
        seg_time = time_axis[seg['start']:seg['end']]
        seg_audio = audio[seg['start']:seg['end']]
        ax_gt.plot(seg_time, seg_audio, color=get_color(seg['cls']))

    # Predicted plot
    ax_pred.set_title('Model predictions (streaming simulation)')
    ax_pred.set_xlabel('Time [s]')
    ax_pred.set_ylabel('Amplitude')
    for seg in predicted_segments:
        seg_time = time_axis[seg['start']:seg['end']]
        seg_audio = audio[seg['start']:seg['end']]
        # Map int -> BreathType
        try:
            btype = BreathType(seg['pred'])
        except ValueError:
            btype = BreathType.OTHER
        ax_pred.plot(seg_time, seg_audio, color=get_color(btype))

    # Legend
    custom_lines = [
        Line2D([0], [0], color=BreathType.EXHALE.get_color(), lw=2),
        Line2D([0], [0], color=BreathType.OTHER.get_color(), lw=2)
    ]
    fig.legend(custom_lines, ['Exhale', 'Other'], loc='lower center', ncol=2)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{base_name}_comparison.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved plot: {out_path}")

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
    for i, wav_name in enumerate(wav_files, start=1):
        base = os.path.splitext(wav_name)[0]
        wav_path = os.path.join(wav_dir, wav_name)
        csv_path = os.path.join(label_dir, f"{base}.csv")
        if not os.path.exists(csv_path):
            print(f"[Skip] Missing label file for {wav_name}: {csv_path}")
            continue
        print(f"[{i}/{len(wav_files)}] Processing {wav_name}")
        try:
            process_file(wav_path, csv_path, classifier, mel_transform, config, BUFFER_SECONDS, OUTPUT_DIR)
        except Exception as e:
            print(f"Error processing {wav_name}: {e}")
    print("Done.")


if __name__ == '__main__':
    run()
