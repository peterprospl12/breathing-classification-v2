import os
import re
import json
import csv as csv_module
import numpy as np
import torchaudio
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from collections import deque
from typing import Optional

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    roc_curve,
    auc,
    matthews_corrcoef,
    log_loss,
)
from sklearn.preprocessing import label_binarize
from scipy import stats

from breathing_model.model.transformer.utils import Config, BreathType
from breathing_model.model.transformer.inference.transform import MelSpectrogramTransform
from breathing_model.model.transformer.inference.model_loader import BreathPhaseClassifier

CONFIG_PATH = 'config.yaml'
MODEL_PATH = 'best_models/best_model_epoch_31.pth'
WAV_DIR_OVERRIDE = '../../data/eval/raw'
LABEL_DIR_OVERRIDE = '../../data/eval/label'
OUTPUT_DIR = 'offline_plots/piotr'
BUFFER_SECONDS = 3.5
LIMIT_WAV = None
MIN_CHUNK_SAMPLES_SKIP = 0
NORMALIZE_AUDIO_FOR_PLOT = False
PRINT_PROBS = False
# Regex pattern to filter filenames (e.g., r'Kinga|Adam' to match files with Kinga or Adam in name).
# Set to None to disable filtering and process all files.
FILENAME_REGEX_FILTER = r'Piotr'  # e.g., r'Kinga|Adam|Maria'

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

def get_gt_label_for_chunk(chunk_start: int, chunk_end: int,
                          gt_segments: list[dict], n_samples: int) -> int:
    """
    Determine ground-truth label for a prediction chunk by majority vote
    over the sample-level ground-truth annotations.
    """
    counts = np.zeros(3, dtype=int)  # exhale=0, inhale=1, silence=2
    chunk_len = chunk_end - chunk_start
    covered = 0
    for seg in gt_segments:
        overlap_start = max(chunk_start, seg['start'])
        overlap_end = min(chunk_end, seg['end'])
        if overlap_end > overlap_start:
            n = overlap_end - overlap_start
            counts[int(seg['cls'])] += n
            covered += n
    # Samples not covered by any annotation are silence
    counts[int(BreathType.SILENCE)] += (chunk_len - covered)
    return int(np.argmax(counts))


def process_file(wav_path: str,
                 csv_path: str,
                 classifier: BreathPhaseClassifier,
                 mel_transform: MelSpectrogramTransform,
                 config: Config,
                 buffer_seconds: float,
                 output_dir: str) -> Optional[dict]:
    """
    Process a single file: generate comparison plot AND return
    chunk-level predictions + ground truth for evaluation.

    Returns dict with keys:
        'filename', 'y_true', 'y_pred', 'y_probs', 'n_chunks'
    or None on error.
    """
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

    gt_segments: list[dict] = []
    for lab in labels:
        s = max(0, lab['start'])
        e = min(n_samples, lab['end'])
        if e <= s:
            continue
        gt_segments.append({'start': s, 'end': e, 'cls': label_to_breath_type(lab['class'])})

    # --- Build chunk-level y_true / y_pred / y_probs arrays ---
    y_true_chunks = []
    y_pred_chunks = []
    y_probs_chunks = []
    for seg in predicted_segments:
        gt_label = get_gt_label_for_chunk(seg['start'], seg['end'], gt_segments, n_samples)
        y_true_chunks.append(gt_label)
        y_pred_chunks.append(seg['pred'])
        y_probs_chunks.append(seg['probs'])

    # --- Plot (unchanged) ---
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

    return {
        'filename': base_name,
        'y_true': np.array(y_true_chunks, dtype=int),
        'y_pred': np.array(y_pred_chunks, dtype=int),
        'y_probs': np.array(y_probs_chunks, dtype=np.float64),
        'n_chunks': len(y_true_chunks),
    }

# ========================================================================================
#  EVALUATION STATISTICS  — suitable for scientific publication
# ========================================================================================

CLASS_NAMES = {0: 'exhale', 1: 'inhale', 2: 'silence'}
NUM_CLASSES = 3
BOOTSTRAP_N = 2000          # number of bootstrap resamples for confidence intervals
BOOTSTRAP_CI = 0.95         # confidence level
BOOTSTRAP_SEED = 42


def bootstrap_metric(y_true: np.ndarray, y_pred: np.ndarray,
                     metric_fn, n_iterations: int = BOOTSTRAP_N,
                     ci: float = BOOTSTRAP_CI,
                     seed: int = BOOTSTRAP_SEED) -> dict:
    """
    Non-parametric bootstrap estimation of a scalar metric.
    Returns {'mean', 'std', 'ci_lower', 'ci_upper', 'ci_level'}.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        try:
            s = metric_fn(y_true[idx], y_pred[idx])
        except Exception:
            continue
        scores.append(s)
    scores = np.array(scores)
    alpha = 1.0 - ci
    lo = np.percentile(scores, 100 * alpha / 2)
    hi = np.percentile(scores, 100 * (1 - alpha / 2))
    return {
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores, ddof=1)),
        'ci_lower': float(lo),
        'ci_upper': float(hi),
        'ci_level': ci,
    }


def compute_overall_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            y_probs: np.ndarray) -> dict:
    """Compute all aggregate metrics on the full evaluation set."""
    results = {}

    # --- Basic metrics ---
    results['accuracy'] = float(accuracy_score(y_true, y_pred))
    results['cohen_kappa'] = float(cohen_kappa_score(y_true, y_pred))
    results['matthews_corrcoef'] = float(matthews_corrcoef(y_true, y_pred))

    # --- Per-class precision, recall, F1 ---
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    for i in range(NUM_CLASSES):
        name = CLASS_NAMES[i]
        results[f'precision_{name}'] = float(prec[i])
        results[f'recall_{name}'] = float(rec[i])
        results[f'f1_{name}'] = float(f1[i])
        results[f'support_{name}'] = int(sup[i])

    # --- Macro / weighted averages ---
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0)
    results['precision_macro'] = float(prec_m)
    results['recall_macro'] = float(rec_m)
    results['f1_macro'] = float(f1_m)
    results['precision_weighted'] = float(prec_w)
    results['recall_weighted'] = float(rec_w)
    results['f1_weighted'] = float(f1_w)

    # --- Log-loss (cross-entropy on predicted probabilities) ---
    try:
        results['log_loss'] = float(log_loss(y_true, y_probs, labels=[0, 1, 2]))
    except Exception:
        results['log_loss'] = None

    # --- ROC-AUC per class (One-vs-Rest) ---
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    for i in range(NUM_CLASSES):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            results[f'roc_auc_{CLASS_NAMES[i]}'] = float(auc(fpr, tpr))
        except Exception:
            results[f'roc_auc_{CLASS_NAMES[i]}'] = None

    # --- Bootstrap confidence intervals for key metrics ---
    print("  Computing bootstrap confidence intervals ...")
    for metric_name, metric_fn in [
        ('accuracy', accuracy_score),
        ('cohen_kappa', cohen_kappa_score),
        ('f1_macro', lambda yt, yp: float(precision_recall_fscore_support(
            yt, yp, average='macro', zero_division=0)[2])),
    ]:
        bs = bootstrap_metric(y_true, y_pred, metric_fn)
        results[f'{metric_name}_bs_mean'] = bs['mean']
        results[f'{metric_name}_bs_std'] = bs['std']
        results[f'{metric_name}_ci95_lower'] = bs['ci_lower']
        results[f'{metric_name}_ci95_upper'] = bs['ci_upper']

    # --- Confusion matrix (raw counts) ---
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    results['confusion_matrix'] = cm.tolist()

    return results


def compute_per_file_metrics(file_results: list[dict]) -> dict:
    """
    Compute per-file accuracy/F1 and then report mean, std, variance,
    min, max across files — shows inter-recording variability.
    """
    per_file_acc = []
    per_file_f1_macro = []
    per_file_kappa = []
    per_file_records = []

    for fr in file_results:
        yt, yp = fr['y_true'], fr['y_pred']
        if len(yt) < 2:
            continue
        acc = accuracy_score(yt, yp)
        _, _, f1, _ = precision_recall_fscore_support(yt, yp, average='macro', zero_division=0)
        try:
            kappa = cohen_kappa_score(yt, yp)
        except Exception:
            kappa = np.nan
        per_file_acc.append(acc)
        per_file_f1_macro.append(f1)
        per_file_kappa.append(kappa)
        per_file_records.append({
            'filename': fr['filename'],
            'n_chunks': fr['n_chunks'],
            'accuracy': round(acc, 4),
            'f1_macro': round(float(f1), 4),
            'kappa': round(float(kappa), 4) if not np.isnan(kappa) else None,
        })

    per_file_acc = np.array(per_file_acc)
    per_file_f1_macro = np.array(per_file_f1_macro)
    per_file_kappa = np.array(per_file_kappa)

    def summarise(arr, name):
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return {}
        return {
            f'{name}_mean': float(np.mean(valid)),
            f'{name}_std': float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
            f'{name}_var': float(np.var(valid, ddof=1)) if len(valid) > 1 else 0.0,
            f'{name}_min': float(np.min(valid)),
            f'{name}_max': float(np.max(valid)),
            f'{name}_median': float(np.median(valid)),
        }

    summary = {}
    summary.update(summarise(per_file_acc, 'file_accuracy'))
    summary.update(summarise(per_file_f1_macro, 'file_f1_macro'))
    summary.update(summarise(per_file_kappa, 'file_kappa'))
    summary['per_file_details'] = per_file_records
    summary['n_files'] = len(per_file_records)

    return summary


def plot_confusion_matrix_eval(y_true: np.ndarray, y_pred: np.ndarray,
                               output_path: str) -> None:
    """Save a publication-quality normalised + raw confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    labels_list = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_list, yticklabels=labels_list, ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title('Confusion Matrix (counts)')

    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=labels_list, yticklabels=labels_list, ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_title('Confusion Matrix (row-normalised)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved confusion matrix: {output_path}")


def plot_roc_curves(y_true: np.ndarray, y_probs: np.ndarray,
                    output_path: str) -> None:
    """Plot One-vs-Rest ROC curves for each class."""
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    class_colors = {'exhale': 'green', 'inhale': 'red', 'silence': 'blue'}

    fig, ax = plt.subplots(figsize=(8, 7))
    for i in range(NUM_CLASSES):
        name = CLASS_NAMES[i]
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=class_colors[name], lw=2,
                    label=f'{name} (AUC = {roc_auc_val:.3f})')
        except Exception:
            pass

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves (One-vs-Rest)')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved ROC curves: {output_path}")


def plot_per_file_metrics(per_file_details: list[dict], output_path: str) -> None:
    """Bar chart of per-file accuracy and F1-macro with mean±std reference."""
    if not per_file_details:
        return
    names = [d['filename'] for d in per_file_details]
    accs = [d['accuracy'] for d in per_file_details]
    f1s = [d['f1_macro'] for d in per_file_details]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.8), 5))
    ax.bar(x - width / 2, accs, width, label='Accuracy', color='steelblue')
    ax.bar(x + width / 2, f1s, width, label='F1-macro', color='coral')

    mean_acc = np.mean(accs)
    mean_f1 = np.mean(f1s)
    ax.axhline(mean_acc, color='steelblue', ls='--', lw=1, alpha=0.7)
    ax.axhline(mean_f1, color='coral', ls='--', lw=1, alpha=0.7)

    ax.set_ylabel('Score')
    ax.set_title('Per-file evaluation metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved per-file metrics chart: {output_path}")


def print_evaluation_report(overall: dict, per_file: dict) -> str:
    """Build, print, and return a full evaluation report as text."""
    lines = []
    def p(text=''):
        lines.append(text)

    sep = '=' * 72
    p(f"\n{sep}")
    p("  EVALUATION REPORT  (chunk-level, streaming simulation)")
    p(sep)

    p(f"\n{'Metric':<35} {'Value':>12}")
    p('-' * 50)
    p(f"{'Accuracy':<35} {overall['accuracy']:>12.4f}")
    p(f"{'  95% CI':<35} [{overall.get('accuracy_ci95_lower', 0):.4f}, "
      f"{overall.get('accuracy_ci95_upper', 0):.4f}]")
    p(f"{'  Bootstrap std':<35} {overall.get('accuracy_bs_std', 0):>12.4f}")
    p(f"{'Cohen\'s Kappa':<35} {overall['cohen_kappa']:>12.4f}")
    p(f"{'  95% CI':<35} [{overall.get('cohen_kappa_ci95_lower', 0):.4f}, "
      f"{overall.get('cohen_kappa_ci95_upper', 0):.4f}]")
    p(f"{'Matthews Corr. Coeff. (MCC)':<35} {overall['matthews_corrcoef']:>12.4f}")
    p(f"{'Log-loss':<35} {overall.get('log_loss', 'N/A'):>12}")

    p(f"\n{'--- Macro-averaged ---':}")
    p(f"{'Precision (macro)':<35} {overall['precision_macro']:>12.4f}")
    p(f"{'Recall    (macro)':<35} {overall['recall_macro']:>12.4f}")
    p(f"{'F1-score  (macro)':<35} {overall['f1_macro']:>12.4f}")
    p(f"{'  95% CI':<35} [{overall.get('f1_macro_ci95_lower', 0):.4f}, "
      f"{overall.get('f1_macro_ci95_upper', 0):.4f}]")

    p(f"\n{'--- Weighted-averaged ---':}")
    p(f"{'Precision (weighted)':<35} {overall['precision_weighted']:>12.4f}")
    p(f"{'Recall    (weighted)':<35} {overall['recall_weighted']:>12.4f}")
    p(f"{'F1-score  (weighted)':<35} {overall['f1_weighted']:>12.4f}")

    p(f"\n{'--- Per-class ---':}")
    p(f"{'Class':<12} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Support':>8} {'AUC':>8}")
    p('-' * 56)
    for i in range(NUM_CLASSES):
        name = CLASS_NAMES[i]
        auc_val = overall.get(f'roc_auc_{name}', None)
        auc_str = f"{auc_val:.4f}" if auc_val is not None else 'N/A'
        p(f"{name:<12} {overall[f'precision_{name}']:>8.4f} "
          f"{overall[f'recall_{name}']:>8.4f} "
          f"{overall[f'f1_{name}']:>8.4f} "
          f"{overall[f'support_{name}']:>8d} "
          f"{auc_str:>8}")

    p(f"\n{'--- Confusion Matrix ---':}")
    cm = np.array(overall['confusion_matrix'])
    labels_list = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    header = f"{'':>12}" + ''.join(f"{l:>10}" for l in labels_list)
    p(header)
    for i, row in enumerate(cm):
        row_str = ''.join(f"{v:>10d}" for v in row)
        p(f"{labels_list[i]:>12}{row_str}")

    # Per-file variability
    p(f"\n{'--- Inter-recording variability ---':}")
    p(f"{'Files evaluated':<35} {per_file.get('n_files', 0):>12}")
    for metric in ['file_accuracy', 'file_f1_macro', 'file_kappa']:
        if f'{metric}_mean' not in per_file:
            continue
        label = metric.replace('file_', '').replace('_', ' ').title()
        p(f"{label + ' mean':<35} {per_file[f'{metric}_mean']:>12.4f}")
        p(f"{label + ' std':<35} {per_file[f'{metric}_std']:>12.4f}")
        p(f"{label + ' variance':<35} {per_file[f'{metric}_var']:>12.6f}")
        p(f"{label + ' min / max':<35} {per_file[f'{metric}_min']:.4f} / {per_file[f'{metric}_max']:.4f}")
        p(f"{label + ' median':<35} {per_file[f'{metric}_median']:>12.4f}")

    # Per-file detail table
    details = per_file.get('per_file_details', [])
    if details:
        p(f"\n{'--- Per-file detail ---':}")
        p(f"{'Filename':<30} {'Chunks':>8} {'Acc':>8} {'F1-m':>8} {'Kappa':>8}")
        p('-' * 66)
        for d in details:
            kappa_str = f"{d['kappa']:.4f}" if d['kappa'] is not None else 'N/A'
            p(f"{d['filename']:<30} {d['n_chunks']:>8} {d['accuracy']:>8.4f} "
              f"{d['f1_macro']:>8.4f} {kappa_str:>8}")

    p(sep)

    report_text = '\n'.join(lines)
    print(report_text)
    return report_text


def save_results_json(overall: dict, per_file: dict, output_path: str) -> None:
    """Persist all results as a JSON file for downstream tooling / LaTeX generation."""
    payload = {
        'overall_metrics': {k: v for k, v in overall.items()},
        'per_file_metrics': per_file,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved results JSON: {output_path}")


def save_per_file_csv(per_file_details: list[dict], output_path: str) -> None:
    """Save per-file metrics as CSV for easy import into spreadsheets / LaTeX."""
    if not per_file_details:
        return
    fieldnames = ['filename', 'n_chunks', 'accuracy', 'f1_macro', 'kappa']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv_module.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_file_details)
    print(f"Saved per-file CSV: {output_path}")


# ========================================================================================
#  MAIN
# ========================================================================================

def run():
    config = Config.from_yaml(CONFIG_PATH)
    wav_dir = WAV_DIR_OVERRIDE or config.data.data_dir
    label_dir = LABEL_DIR_OVERRIDE or config.data.label_dir

    mel_transform = MelSpectrogramTransform(config.data)
    classifier = BreathPhaseClassifier(MODEL_PATH, config.model, config.data)

    wav_files = sorted([f for f in os.listdir(wav_dir) if f.lower().endswith('.wav')])

    # Filter files by regex pattern if specified
    if FILENAME_REGEX_FILTER:
        pattern = re.compile(FILENAME_REGEX_FILTER, re.IGNORECASE)
        wav_files = [f for f in wav_files if pattern.search(f)]
        print(f"Filtered by regex '{FILENAME_REGEX_FILTER}': {len(wav_files)} matching files")

    if LIMIT_WAV:
        wav_files = wav_files[:LIMIT_WAV]

    print(f"Found {len(wav_files)} wav files in {wav_dir}")

    # --- Process each file and collect results ---
    file_results: list[dict] = []

    for i, wav_name in enumerate(wav_files, start=1):
        base = os.path.splitext(wav_name)[0]
        wav_path = os.path.join(wav_dir, wav_name)
        csv_path = os.path.join(label_dir, f"{base}.csv")
        if not os.path.exists(csv_path):
            print(f"[Skip] Missing label file for {wav_name}: {csv_path}")
            continue
        print(f"[{i}/{len(wav_files)}] Processing {wav_name}")
        try:
            result = process_file(wav_path, csv_path, classifier, mel_transform,
                                  config, BUFFER_SECONDS, OUTPUT_DIR)
            if result is not None and result['n_chunks'] > 0:
                file_results.append(result)
        except Exception as e:
            print(f"Error processing {wav_name}: {e}")

    if not file_results:
        print("No evaluation results collected. Exiting.")
        return

    # --- Aggregate predictions across all files ---
    all_y_true = np.concatenate([r['y_true'] for r in file_results])
    all_y_pred = np.concatenate([r['y_pred'] for r in file_results])
    all_y_probs = np.concatenate([r['y_probs'] for r in file_results])

    print(f"\nTotal evaluation chunks: {len(all_y_true)}")

    # --- Compute metrics ---
    print("Computing overall metrics ...")
    overall_metrics = compute_overall_metrics(all_y_true, all_y_pred, all_y_probs)

    print("Computing per-file metrics ...")
    per_file_metrics = compute_per_file_metrics(file_results)

    # --- Print report ---
    report_text = print_evaluation_report(overall_metrics, per_file_metrics)

    # --- Generate evaluation plots ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plot_confusion_matrix_eval(
        all_y_true, all_y_pred,
        os.path.join(OUTPUT_DIR, 'eval_confusion_matrix.png'))

    plot_roc_curves(
        all_y_true, all_y_probs,
        os.path.join(OUTPUT_DIR, 'eval_roc_curves.png'))

    plot_per_file_metrics(
        per_file_metrics.get('per_file_details', []),
        os.path.join(OUTPUT_DIR, 'eval_per_file_metrics.png'))

    # --- Save machine-readable results ---
    save_results_json(
        overall_metrics, per_file_metrics,
        os.path.join(OUTPUT_DIR, 'eval_results.json'))

    save_per_file_csv(
        per_file_metrics.get('per_file_details', []),
        os.path.join(OUTPUT_DIR, 'eval_per_file.csv'))

    # --- Full sklearn classification report ---
    sklearn_header = "\n" + "=" * 72 + "\n  FULL CLASSIFICATION REPORT (sklearn)\n" + "=" * 72
    sklearn_report = classification_report(
        all_y_true, all_y_pred,
        target_names=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
        digits=4, zero_division=0)
    print(sklearn_header)
    print(sklearn_report)

    # --- Save complete text report to file ---
    full_report = report_text + "\n" + sklearn_header + "\n" + sklearn_report
    report_path = os.path.join(OUTPUT_DIR, 'eval_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(full_report)
    print(f"Saved full text report: {report_path}")

    print("Done.")


if __name__ == '__main__':
    run()

