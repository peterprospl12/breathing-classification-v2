"""
Multi-model evaluation script for breathing phase classification.

Compares CNN, Feed-Forward, and Transformer models on evaluation data.
Computes comprehensive metrics and generates comparison plots.

Metrics computed:
- Global: Accuracy, Cohen's Kappa, MCC, Log Loss
- Per-class: Precision, Recall, F1-Score, Support, ROC AUC
- Averaged: Macro & Weighted Precision, Recall, F1-Score
- Bootstrap: Mean, Std, 95% CI for Accuracy, Cohen's Kappa, F1 Macro
- Confusion Matrix
- Per-file statistics: Mean, Std, Var, Min, Max, Median
"""

import os
import csv
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime
from scipy.ndimage import median_filter

from sklearn.metrics import (
    accuracy_score, confusion_matrix, cohen_kappa_score,
    matthews_corrcoef, log_loss, precision_recall_fscore_support,
    roc_auc_score, f1_score,
)

from torch.utils.data import DataLoader

from breathing_model.model.transformer.utils import BreathType, load_yaml
from breathing_model.model.transformer.dataset import BreathDataset, collate_fn
from breathing_model.model.transformer.model import BreathPhaseTransformerSeq
from breathing_model.model.cnn.model import BreathPhaseCNN
from breathing_model.model.feed_forward.model import BreathPhaseFeedForward

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

WAV_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'data', 'eval3', 'raw'))
LABEL_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'data', 'eval3', 'label'))
OUTPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, 'evaluation_output_swa_model_nasze_dane'))
PLOTS_DIR = os.path.normpath(os.path.join(OUTPUT_DIR, 'plots'))
METRICS_FILE = os.path.normpath(os.path.join(OUTPUT_DIR, 'evaluation_metrics.txt'))

LIMIT_WAV = None

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42

WINDOW_TRANS = 0.20    # Aggregation window (seconds) for metrics and plot coloring
WINDOW_INFERENCE = 10  # Inference chunk size (seconds) for chunked model execution
INFERENCE_OVERLAP = 0.5  # Overlap ratio between inference chunks (0.0 = no overlap, 0.5 = 50%)
MEDIAN_FILTER_SIZE = 5   # Median filter kernel size for temporal smoothing of predictions (0 = disabled)

MODELS_CONFIG = {
    'CNN': {
        'config_path': os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'model', 'cnn', 'config.yaml')),
        'model_path': os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'model', 'cnn', 'checkpoints', 'best_model_epoch_21.pth')),
    },
    'Feed-Forward': {
        'config_path': os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'model', 'feed_forward', 'config.yaml')),
        'model_path': os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'model', 'feed_forward', 'checkpoints', 'best_model_epoch_14.pth')),
    },
    'Transformer': {
        'config_path': os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'model', 'transformer', 'config.yaml')),
        'model_path': os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'model', 'transformer', 'checkpoints2', 'swa_model.pth')),
    },
}

NUM_CLASSES = 3
CLASS_NAMES = ['Exhale (0)', 'Inhale (1)', 'Silence (2)']


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_audio(wav_path: str, target_sr: int) -> np.ndarray:
    waveform, sr = torchaudio.load(wav_path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0).numpy().astype(np.float32)


def parse_label_csv(csv_path: str) -> list[dict]:
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
    if label == 'exhale':
        return BreathType.EXHALE
    if label == 'inhale':
        return BreathType.INHALE
    return BreathType.SILENCE


def get_ground_truth_for_chunk(labels: list[dict], start: int, end: int) -> int:
    """Majority voting for ground truth over a sample range."""
    counts = {0: 0, 1: 0, 2: 0}
    counts[2] = end - start  # Default silence

    for lab in labels:
        cls_map = {'exhale': 0, 'inhale': 1, 'silence': 2}
        cls_id = cls_map.get(lab['class'], 2)
        overlap_start = max(lab['start'], start)
        overlap_end = min(lab['end'], end)
        overlap = overlap_end - overlap_start
        if overlap > 0:
            counts[cls_id] += overlap
            counts[2] -= overlap

    return max(counts, key=counts.get)


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(model_name: str, config: dict, model_path: str, device: torch.device) -> torch.nn.Module:
    model_cfg = config['model']

    if model_name == 'CNN':
        model = BreathPhaseCNN(
            n_mels=model_cfg['n_mels'],
            num_classes=model_cfg['num_classes'],
            dropout=model_cfg['dropout'],
        )
    elif model_name == 'Feed-Forward':
        model = BreathPhaseFeedForward(
            n_mels=model_cfg['n_mels'],
            context_frames=model_cfg['context_frames'],
            hidden_dim=model_cfg['hidden_dim'],
            num_classes=model_cfg['num_classes'],
            dropout=model_cfg['dropout'],
        )
    elif model_name == 'Transformer':
        model = BreathPhaseTransformerSeq(
            n_mels=model_cfg['n_mels'],
            d_model=model_cfg['d_model'],
            nhead=model_cfg['nhead'],
            num_layers=model_cfg['num_layers'],
            num_classes=model_cfg['num_classes'],
            dim_feedforward=model_cfg.get('dim_feedforward', 1024),
            dropout=model_cfg.get('dropout', 0.15),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# ============================================================
# METRICS COMPUTATION
# ============================================================

def compute_bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=1000, seed=42, ci=0.95):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    scores = []
    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, n)
        try:
            score = metric_fn(y_true[indices], y_pred[indices])
            scores.append(score)
        except Exception:
            continue

    scores = np.array(scores)
    alpha = 1 - ci
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    return scores.mean(), scores.std(), lower, upper


def compute_all_metrics(y_true, y_pred, y_probs, per_file_results: list[dict]) -> dict:
    metrics = {}

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # --- Global metrics ---
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['total_frames'] = len(y_true)

    try:
        metrics['log_loss'] = log_loss(y_true, y_probs, labels=[0, 1, 2])
    except Exception:
        metrics['log_loss'] = float('nan')

    # --- Per-class metrics ---
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    metrics['per_class'] = {}
    for i, name in enumerate(CLASS_NAMES):
        metrics['per_class'][name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': int(support[i]),
        }

    # ROC AUC per class (One-vs-Rest)
    for i, name in enumerate(CLASS_NAMES):
        try:
            y_true_binary = (y_true == i).astype(int)
            metrics['per_class'][name]['roc_auc'] = roc_auc_score(y_true_binary, y_probs[:, i])
        except Exception:
            metrics['per_class'][name]['roc_auc'] = float('nan')

    # --- Macro and Weighted averages ---
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics['macro'] = {'precision': precision_macro, 'recall': recall_macro, 'f1': f1_macro}
    metrics['weighted'] = {'precision': precision_weighted, 'recall': recall_weighted, 'f1': f1_weighted}

    # --- Bootstrap analysis ---
    def accuracy_fn(yt, yp):
        return accuracy_score(yt, yp)

    def kappa_fn(yt, yp):
        return cohen_kappa_score(yt, yp)

    def f1_macro_fn(yt, yp):
        return f1_score(yt, yp, average='macro', zero_division=0)

    metrics['bootstrap'] = {}
    for name, fn in [('Accuracy', accuracy_fn), ('Cohen\'s Kappa', kappa_fn), ('F1 Macro', f1_macro_fn)]:
        mean, std, lower, upper = compute_bootstrap_ci(y_true, y_pred, fn, BOOTSTRAP_N, BOOTSTRAP_SEED)
        metrics['bootstrap'][name] = {'mean': mean, 'std': std, 'ci_lower': lower, 'ci_upper': upper}

    # --- Confusion matrix ---
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # --- Per-file statistics ---
    file_accuracies = []
    file_f1_macros = []
    file_kappas = []

    for fr in per_file_results:
        if len(fr['y_true']) == 0:
            continue
        yt = np.array(fr['y_true'])
        yp = np.array(fr['y_pred'])
        file_accuracies.append(accuracy_score(yt, yp))
        file_f1_macros.append(f1_score(yt, yp, average='macro', zero_division=0))
        try:
            file_kappas.append(cohen_kappa_score(yt, yp))
        except Exception:
            file_kappas.append(0.0)

    def compute_stats(values):
        arr = np.array(values)
        return {
            'mean': arr.mean(),
            'std': arr.std(),
            'var': arr.var(),
            'min': arr.min(),
            'max': arr.max(),
            'median': np.median(arr),
        }

    metrics['per_file_stats'] = {}
    if file_accuracies:
        metrics['per_file_stats']['Accuracy'] = compute_stats(file_accuracies)
    if file_f1_macros:
        metrics['per_file_stats']['F1 Macro'] = compute_stats(file_f1_macros)
    if file_kappas:
        metrics['per_file_stats']['Cohen\'s Kappa'] = compute_stats(file_kappas)

    return metrics


# ============================================================
# METRICS FORMATTING & WRITING
# ============================================================

def format_metrics_report(all_model_metrics: dict[str, dict]) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("MULTI-MODEL EVALUATION REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Evaluation data: {WAV_DIR}")
    lines.append(f"Evaluation mode: chunked inference ({WINDOW_INFERENCE}s), aggregated ({WINDOW_TRANS}s windows)")
    lines.append(f"Bootstrap samples: {BOOTSTRAP_N}")
    lines.append("=" * 80)

    for model_name, metrics in all_model_metrics.items():
        lines.append("")
        lines.append("*" * 80)
        lines.append(f"  MODEL: {model_name}")
        lines.append(f"  Checkpoint: {MODELS_CONFIG[model_name]['model_path']}")
        lines.append(f"  Total windows evaluated: {metrics['total_frames']}")
        lines.append("*" * 80)

        # Global metrics
        lines.append("")
        lines.append("-" * 60)
        lines.append("GLOBAL METRICS")
        lines.append("-" * 60)
        lines.append(f"  Accuracy:                    {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
        lines.append(f"  Cohen's Kappa:               {metrics['cohen_kappa']:.4f}")
        lines.append(f"  Matthews Corr. Coeff. (MCC): {metrics['mcc']:.4f}")
        log_loss_val = metrics['log_loss']
        log_loss_str = f"{log_loss_val:.4f}" if not np.isnan(log_loss_val) else "N/A"
        lines.append(f"  Log Loss (Cross-Entropy):    {log_loss_str}")

        # Per-class metrics
        lines.append("")
        lines.append("-" * 60)
        lines.append("PER-CLASS METRICS")
        lines.append("-" * 60)
        header = f"  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10} {'ROC AUC':>10}"
        lines.append(header)
        lines.append("  " + "-" * 65)

        for cls_name, cls_metrics in metrics['per_class'].items():
            roc_val = cls_metrics['roc_auc']
            roc_str = f"{roc_val:.4f}" if not np.isnan(roc_val) else "N/A"
            lines.append(
                f"  {cls_name:<15} {cls_metrics['precision']:>10.4f} {cls_metrics['recall']:>10.4f} "
                f"{cls_metrics['f1']:>10.4f} {cls_metrics['support']:>10d} {roc_str:>10}"
            )

        # Averaged metrics
        lines.append("")
        lines.append("-" * 60)
        lines.append("AVERAGED METRICS")
        lines.append("-" * 60)
        macro = metrics['macro']
        weighted = metrics['weighted']
        lines.append(f"  Macro-averaged:")
        lines.append(f"    Precision: {macro['precision']:.4f}  Recall: {macro['recall']:.4f}  F1-Score: {macro['f1']:.4f}")
        lines.append(f"  Weighted-averaged:")
        lines.append(f"    Precision: {weighted['precision']:.4f}  Recall: {weighted['recall']:.4f}  F1-Score: {weighted['f1']:.4f}")

        # Bootstrap
        lines.append("")
        lines.append("-" * 60)
        lines.append(f"BOOTSTRAP ANALYSIS (N={BOOTSTRAP_N}, 95% CI)")
        lines.append("-" * 60)
        for metric_name, bs in metrics['bootstrap'].items():
            lines.append(f"  {metric_name}:")
            lines.append(f"    Mean: {bs['mean']:.4f}  Std: {bs['std']:.4f}")
            lines.append(f"    95% CI: [{bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}]")

        # Confusion matrix
        lines.append("")
        lines.append("-" * 60)
        lines.append("CONFUSION MATRIX")
        lines.append("-" * 60)
        cm = metrics['confusion_matrix']
        lines.append(f"  {'':>15} {'Pred Exhale':>12} {'Pred Inhale':>12} {'Pred Silence':>13}")
        lines.append("  " + "-" * 52)
        row_names = ['True Exhale', 'True Inhale', 'True Silence']
        for i, rn in enumerate(row_names):
            lines.append(f"  {rn:>15} {cm[i, 0]:>12d} {cm[i, 1]:>12d} {cm[i, 2]:>13d}")

        # Per-file statistics
        lines.append("")
        lines.append("-" * 60)
        lines.append("PER-FILE STATISTICS")
        lines.append("-" * 60)
        for stat_name, stats in metrics.get('per_file_stats', {}).items():
            lines.append(f"  {stat_name}:")
            lines.append(f"    Mean: {stats['mean']:.4f}  Std: {stats['std']:.4f}  Var: {stats['var']:.6f}")
            lines.append(f"    Min: {stats['min']:.4f}  Max: {stats['max']:.4f}  Median: {stats['median']:.4f}")

        lines.append("")

    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load all model configs
    configs = {}
    for model_name, model_info in MODELS_CONFIG.items():
        configs[model_name] = load_yaml(model_info['config_path'])

    # Use transformer config for audio/data processing (all models share the same data params)
    data_config = configs['Transformer']['data']
    sample_rate = data_config['sample_rate']
    hop_length = data_config['hop_length']

    samples_trans = int(sample_rate * WINDOW_TRANS)
    frames_per_inference = int(sample_rate * WINDOW_INFERENCE / hop_length)

    # Create dataset and data loader (consistent mel computation via BreathDataset)
    test_dataset = BreathDataset(
        data_dir=WAV_DIR,
        label_dir=LABEL_DIR,
        sample_rate=sample_rate,
        n_mels=data_config['n_mels'],
        n_fft=data_config['n_fft'],
        hop_length=hop_length,
        augment=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )

    # Load all models
    models = {}
    for model_name, model_info in MODELS_CONFIG.items():
        print(f"Loading {model_name} model from {model_info['model_path']}...")
        models[model_name] = load_model(model_name, configs[model_name], model_info['model_path'], device)
        print(f"  {model_name} loaded successfully.")

    num_files = min(LIMIT_WAV, len(test_dataset)) if LIMIT_WAV else len(test_dataset)
    print(f"\nFound {len(test_dataset)} files in dataset (processing {num_files})")
    print(f"Inference chunk: {WINDOW_INFERENCE}s ({frames_per_inference} frames)")
    print(f"Aggregation window: {WINDOW_TRANS}s ({samples_trans} samples)")

    # Aggregate results per model
    all_results = {name: {'y_true': [], 'y_pred': [], 'y_probs': [], 'per_file': []}
                   for name in models}

    with torch.no_grad():
        for i, (spectrogram, _labels_tensor, padding_mask) in enumerate(test_loader):
            if LIMIT_WAV and i >= LIMIT_WAV:
                break

            wav_filename = test_dataset.wav_files[i]
            base_name = os.path.splitext(wav_filename)[0]
            wav_path = os.path.join(WAV_DIR, wav_filename)
            csv_path = os.path.join(LABEL_DIR, f"{base_name}.csv")

            print(f"[{i + 1}/{num_files}] Processing {wav_filename}...")

            total_length = spectrogram.shape[-1]
            spectrogram = spectrogram.to(device)
            valid_mask = ~padding_mask.to(device)

            # Load raw audio for plotting and CSV labels for ground truth
            audio = load_audio(wav_path, sample_rate)
            n_samples = len(audio)
            csv_labels = parse_label_csv(csv_path)

            # --- Chunked inference with overlap for each model ---
            model_raw_preds = {}
            model_raw_probs = {}

            overlap_frames = int(frames_per_inference * INFERENCE_OVERLAP)
            stride_frames = max(frames_per_inference - overlap_frames, 1)

            for model_name, model in models.items():
                # Accumulate logits with overlap averaging
                logits_sum = torch.zeros((1, total_length, NUM_CLASSES), device=device)
                logits_count = torch.zeros((1, total_length, 1), device=device)

                for start in range(0, total_length, stride_frames):
                    end = min(start + frames_per_inference, total_length)
                    chunk = spectrogram[..., start:end]
                    chunk_len = end - start
                    chunk_mask = torch.zeros((1, chunk_len), dtype=torch.bool).to(device)
                    output = model(chunk, src_key_padding_mask=chunk_mask)
                    logits_sum[:, start:end, :] += output
                    logits_count[:, start:end, :] += 1

                # Average overlapping regions
                logits_count = logits_count.clamp(min=1)
                full_logits = logits_sum / logits_count  # [1, T, C]
                full_probs = torch.softmax(full_logits, dim=-1)   # [1, T, C]

                # Extract valid (non-padded) predictions
                raw_preds = torch.argmax(full_logits, dim=-1)[valid_mask].cpu().numpy()
                raw_probs = full_probs[0][valid_mask[0]].cpu().numpy()  # [T_valid, C]

                # Apply temporal median filter to smooth predictions
                if MEDIAN_FILTER_SIZE > 1 and len(raw_preds) > MEDIAN_FILTER_SIZE:
                    raw_preds = median_filter(raw_preds, size=MEDIAN_FILTER_SIZE).astype(raw_preds.dtype)

                model_raw_preds[model_name] = raw_preds
                model_raw_probs[model_name] = raw_probs

            # --- Windowed aggregation for metrics (like _two_transformers) ---
            for model_name in models:
                raw_preds = model_raw_preds[model_name]
                raw_probs = model_raw_probs[model_name]

                file_y_true = []
                file_y_pred = []
                file_y_probs = []

                for j in range(0, n_samples, samples_trans):
                    end_sample = min(j + samples_trans, n_samples)
                    if (end_sample - j) < samples_trans:
                        continue

                    gt = get_ground_truth_for_chunk(csv_labels, j, end_sample)

                    start_frame = int(j / hop_length)
                    end_frame = int(end_sample / hop_length)
                    chunk_preds = raw_preds[start_frame:end_frame]
                    chunk_probs = raw_probs[start_frame:end_frame]

                    if len(chunk_preds) > 0:
                        pred = int(np.bincount(chunk_preds, minlength=NUM_CLASSES).argmax())
                        avg_probs = chunk_probs.mean(axis=0)
                    else:
                        pred = 2  # silence
                        avg_probs = np.array([0.0, 0.0, 1.0])

                    file_y_true.append(gt)
                    file_y_pred.append(pred)
                    file_y_probs.append(avg_probs.tolist())

                all_results[model_name]['y_true'].extend(file_y_true)
                all_results[model_name]['y_pred'].extend(file_y_pred)
                all_results[model_name]['y_probs'].extend(file_y_probs)
                all_results[model_name]['per_file'].append({
                    'file': wav_filename,
                    'y_true': file_y_true,
                    'y_pred': file_y_pred,
                })

            # --- Generate comparison plot (4 subplots) ---
            gt_segments = []
            for lab in csv_labels:
                s = max(0, lab['start'])
                e = min(n_samples, lab['end'])
                if e <= s:
                    continue
                gt_segments.append({'start': s, 'end': e, 'cls': label_to_breath_type(lab['class'])})

            time_axis = np.arange(n_samples) / sample_rate
            model_names = list(models.keys())

            fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)
            fig.suptitle(f"Model Comparison: {base_name}", fontsize=14, fontweight='bold')

            # 1) Ground truth
            axes[0].set_title('Ground Truth')
            axes[0].set_ylabel('Amplitude')
            for seg in gt_segments:
                axes[0].plot(time_axis[seg['start']:seg['end']],
                             audio[seg['start']:seg['end']],
                             color=seg['cls'].get_color())

            # 2-4) Model predictions (colored by windowed majority voting)
            for idx, model_name in enumerate(model_names):
                ax = axes[idx + 1]
                ax.set_title(f'{model_name} Predictions')
                ax.set_ylabel('Amplitude')
                preds = model_raw_preds[model_name]

                for j in range(0, n_samples, samples_trans):
                    end_sample = min(j + samples_trans, n_samples)
                    if (end_sample - j) < samples_trans:
                        continue

                    start_frame = int(j / hop_length)
                    end_frame = int(end_sample / hop_length)
                    chunk_preds = preds[start_frame:end_frame]

                    if len(chunk_preds) > 0:
                        p = int(np.bincount(chunk_preds, minlength=NUM_CLASSES).argmax())
                    else:
                        p = 2

                    try:
                        btype = BreathType(p)
                    except ValueError:
                        btype = BreathType.SILENCE

                    ax.plot(time_axis[j:end_sample],
                            audio[j:end_sample],
                            color=btype.get_color())

            axes[-1].set_xlabel('Time [s]')

            custom_lines = [
                Line2D([0], [0], color=BreathType.EXHALE.get_color(), lw=2),
                Line2D([0], [0], color=BreathType.INHALE.get_color(), lw=2),
                Line2D([0], [0], color=BreathType.SILENCE.get_color(), lw=2),
            ]
            fig.legend(custom_lines, ['Exhale', 'Inhale', 'Silence'], loc='lower center', ncol=3)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.08)

            os.makedirs(PLOTS_DIR, exist_ok=True)
            out_path = os.path.join(PLOTS_DIR, f"{base_name}_comparison.png")
            plt.savefig(out_path, dpi=150)
            plt.close(fig)

    # Compute metrics for each model
    print("\nComputing metrics...")
    all_model_metrics = {}
    for model_name in models:
        res = all_results[model_name]
        if len(res['y_true']) == 0:
            print(f"No data for {model_name}, skipping metrics.")
            continue
        print(f"  Computing metrics for {model_name}...")
        all_model_metrics[model_name] = compute_all_metrics(
            res['y_true'], res['y_pred'], res['y_probs'], res['per_file']
        )

    # Write metrics report to file
    report = format_metrics_report(all_model_metrics)
    with open(METRICS_FILE, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nMetrics report saved to: {METRICS_FILE}")
    print(f"Plots saved to: {PLOTS_DIR}")
    print("Done.")


if __name__ == '__main__':
    run()
