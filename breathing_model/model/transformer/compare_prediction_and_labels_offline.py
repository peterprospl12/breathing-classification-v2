import os
import numpy as np
import torchaudio
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import deque

from breathing_model.model.transformer.utils import Config, BreathType
from breathing_model.model.transformer.inference.transform import MelSpectrogramTransform
from breathing_model.model.transformer.inference.model_loader import BreathPhaseClassifier

# =============================================================================
# USTAWIENIA (zmień tutaj – brak argparse)
# =============================================================================
CONFIG_PATH = 'config.yaml'  # ścieżka do config.yaml (relatywnie do tego pliku)
MODEL_PATH = 'checkpoints/best_model_epoch_30.pth'  # domyślny checkpoint (zmień jeśli inny)
WAV_DIR_OVERRIDE = '../../data/eval/raw'            # jeśli None -> użyje config.data.data_dir
LABEL_DIR_OVERRIDE = '../../data/eval/label'        # jeśli None -> użyje config.data.label_dir
OUTPUT_DIR = 'offline_plots'                       # katalog na wykresy wynikowe
BUFFER_SECONDS = 3.5                                # długość ruchomego bufora (sekundy)
LIMIT_WAV = None                                    # np. 5 aby ograniczyć liczbę przetwarzanych plików
MIN_CHUNK_SAMPLES_SKIP = 0                          # pomiń chunk jeśli krótszy niż ten próg (0 = wyłączone)
NORMALIZE_AUDIO_FOR_PLOT = False                    # jeśli True normalizuje amplitudy do [-1,1] tylko dla rysunku
PRINT_PROBS = False                                 # jeśli True wypisuje również średnie prawdopodobieństwa
# =============================================================================

# -----------------------------------------------------------------------------
# Funkcje pomocnicze
# -----------------------------------------------------------------------------

def load_audio(wav_path: str, target_sr: int) -> np.ndarray:
    """Wczytaj plik WAV -> mono float32 przy zadanym sample_rate."""
    waveform, sr = torchaudio.load(wav_path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0).numpy().astype(np.float32)


def parse_label_csv(csv_path: str) -> list[dict]:
    """CSV z kolumnami: class,start_sample,end_sample -> lista słowników."""
    import csv
    labels = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
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
    """Mapowanie pełnych etykiet (exhale/inhale/silence) na enum BreathType."""
    if label == 'exhale':
        return BreathType.EXHALE
    if label == 'inhale':
        return BreathType.INHALE
    if label == 'silence':
        return BreathType.SILENCE
    # Nieznane traktujemy jako SILENCE (najbardziej neutralne)
    return BreathType.SILENCE


def get_color(bt: BreathType) -> str:
    return bt.get_color()

# -----------------------------------------------------------------------------
# Główna logika offline (symulacja trybu realtime chunk po chunku)
# -----------------------------------------------------------------------------

def process_file(wav_path: str,
                 csv_path: str,
                 classifier: BreathPhaseClassifier,
                 mel_transform: MelSpectrogramTransform,
                 config: Config,
                 buffer_seconds: float,
                 output_dir: str) -> None:
    base_name = os.path.splitext(os.path.basename(wav_path))[0]

    # 1. Audio
    audio = load_audio(wav_path, config.data.sample_rate)
    n_samples = audio.shape[0]
    sr = config.data.sample_rate

    if NORMALIZE_AUDIO_FOR_PLOT and np.max(np.abs(audio)) > 0:
        audio_for_plot = audio / np.max(np.abs(audio))
    else:
        audio_for_plot = audio

    # 2. Ground truth etykiety
    labels = parse_label_csv(csv_path)

    # 3. Symulacja strumienia
    chunk_size = int(config.audio.chunk_length * sr)
    max_buffer = int(buffer_seconds * sr)
    audio_buffer = deque(maxlen=max_buffer)

    predicted_segments: list[dict] = []  # {start,end,pred, probs(optional)}

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk = audio[start:end]
        if chunk.size == 0 or chunk.size < MIN_CHUNK_SAMPLES_SKIP:
            continue

        audio_buffer.extend(chunk)
        buf_np = np.array(audio_buffer, dtype=np.float32)
        if buf_np.size == 0:
            continue

        mel = mel_transform(buf_np)
        pred_cls, probs = classifier.predict(mel)
        predicted_segments.append({'start': start, 'end': end, 'pred': pred_cls, 'probs': probs})

        if PRINT_PROBS:
            print(f"Chunk {start}:{end} -> pred={BreathType(pred_cls).name} probs={probs.round(3)}")

    # 4. Ground truth segmenty (mapowanie na BreathType)
    gt_segments: list[dict] = []
    for lab in labels:
        s = max(0, lab['start'])
        e = min(n_samples, lab['end'])
        if e <= s:
            continue
        gt_segments.append({'start': s, 'end': e, 'cls': label_to_breath_type(lab['class'])})

    # 5. Wykresy
    time_axis = np.arange(n_samples) / sr

    fig, (ax_gt, ax_pred) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    fig.suptitle(f"Offline comparison (transformer): {base_name}")

    # GT
    ax_gt.set_title('Ground truth labels')
    ax_gt.set_ylabel('Amplitude')
    for seg in gt_segments:
        ax_gt.plot(time_axis[seg['start']:seg['end']],
                   audio_for_plot[seg['start']:seg['end']],
                   color=get_color(seg['cls']))

    # Predictions
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

    # Legenda
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

# -----------------------------------------------------------------------------
# Uruchomienie
# -----------------------------------------------------------------------------

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

