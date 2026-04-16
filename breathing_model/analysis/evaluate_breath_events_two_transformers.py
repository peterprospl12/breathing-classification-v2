"""
Count-only breath evaluation for two Transformer models (3-class and 2-class).

Rules:
- One folder with WAV/CSV labels.
- Window aggregation at WINDOW_TRANS seconds.
- Prediction smoothing: runs shorter than MIN_RUN_LEN are treated as noise.
- Breath counting = number of runs (continuous segments) for selected classes.
- Detection summary is count-based:
    detected = min(gt_count, pred_count)
    missed   = max(0, gt_count - pred_count)
    false    = max(0, pred_count - gt_count)
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import datetime

import numpy as np
import torch
from scipy.ndimage import median_filter
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from breathing_model.model.exhale_only_detection.model import BreathPhaseTransformerSeq as TwoClassTransformer
from breathing_model.model.exhale_only_detection.utils import Config as TwoClassConfig
from breathing_model.model.transformer.dataset import BreathDataset, collate_fn
from breathing_model.model.transformer.model import BreathPhaseTransformerSeq as ThreeClassTransformer
from breathing_model.model.transformer.utils import Config as ThreeClassConfig

# One dataset only
WAV_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "eval_unseen_people", "raw"))
LABEL_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "eval_unseen_people", "label"))

THREE_CLASS_CONFIG_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "model", "transformer", "config.yaml"))
THREE_CLASS_MODEL_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "model", "transformer", "best_models", "best_model_epoch_31.pth"))
TWO_CLASS_CONFIG_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "model", "exhale_only_detection", "config.yaml"))
TWO_CLASS_MODEL_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "model", "exhale_only_detection", "best_models", "best_model_epoch_21.pth"))

OUTPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "event_level_output_two_transformers"))
REPORT_PATH = os.path.join(OUTPUT_DIR, "event_level_metrics.txt")

WINDOW_TRANS = 0.20
WINDOW_INFERENCE = 10
INFERENCE_OVERLAP = 0.0
MEDIAN_FILTER_SIZE = 3
MIN_RUN_LEN = 2

LIMIT_WAV_ENV = os.environ.get("LIMIT_WAV", "")
LIMIT_WAV = int(LIMIT_WAV_ENV) if LIMIT_WAV_ENV.strip() else None

NUM_WORKERS = 0 if os.name == "nt" else min(4, os.cpu_count() or 1)


def parse_label_csv(csv_path: str) -> list[dict]:
    labels = []
    if not os.path.exists(csv_path):
        return labels
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        _ = next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                labels.append({"class": row[0].strip().lower(), "start": int(row[1]), "end": int(row[2])})
            except ValueError:
                continue
    return labels


def get_gt_three_class(labels: list[dict], start: int, end: int) -> int:
    counts = {0: 0, 1: 0, 2: 0}
    counts[2] = end - start
    cls_map = {"exhale": 0, "inhale": 1, "silence": 2}
    for lab in labels:
        cls_id = cls_map.get(lab["class"], 2)
        overlap_start = max(lab["start"], start)
        overlap_end = min(lab["end"], end)
        overlap = overlap_end - overlap_start
        if overlap > 0:
            counts[cls_id] += overlap
            counts[2] -= overlap
    return max(counts, key=counts.get)


def get_gt_two_class(labels: list[dict], start: int, end: int) -> int:
    counts = {0: 0, 1: 0}
    counts[1] = end - start
    for lab in labels:
        cls_id = 0 if lab["class"] == "exhale" else 1
        overlap_start = max(lab["start"], start)
        overlap_end = min(lab["end"], end)
        overlap = overlap_end - overlap_start
        if overlap > 0:
            counts[cls_id] += overlap
            counts[1] -= overlap
    return max(counts, key=counts.get)


def smooth_short_runs(sequence: list[int], min_run_len: int = 2) -> list[int]:
    if not sequence:
        return sequence

    arr = list(sequence)
    n = len(arr)
    i = 0
    while i < n:
        j = i + 1
        while j < n and arr[j] == arr[i]:
            j += 1

        run_len = j - i
        if run_len < min_run_len:
            prev_cls = arr[i - 1] if i > 0 else None
            next_cls = arr[j] if j < n else None
            if prev_cls is not None and next_cls is not None and prev_cls == next_cls:
                fill_cls = prev_cls
            elif prev_cls is not None:
                fill_cls = prev_cls
            elif next_cls is not None:
                fill_cls = next_cls
            else:
                fill_cls = arr[i]
            for k in range(i, j):
                arr[k] = fill_cls

        i = j

    return arr


def sequence_to_events(sequence: list[int]) -> list[dict]:
    events = []
    if not sequence:
        return events
    start = 0
    cls = sequence[0]
    for idx in range(1, len(sequence)):
        if sequence[idx] != cls:
            events.append({"class": cls, "start": start, "end": idx})
            start = idx
            cls = sequence[idx]
    events.append({"class": cls, "start": start, "end": len(sequence)})
    return events


def class_mode(values: list[int], num_classes: int) -> int:
    if not values:
        return 0
    return int(np.bincount(np.array(values), minlength=num_classes).argmax())


def infer_raw_predictions(
    model: torch.nn.Module,
    spectrogram: torch.Tensor,
    valid_mask: torch.Tensor,
    frames_per_inference: int,
    num_classes: int,
    device: torch.device,
) -> list[int]:
    total_length = spectrogram.shape[-1]
    overlap_frames = int(frames_per_inference * INFERENCE_OVERLAP)
    stride_frames = max(frames_per_inference - overlap_frames, 1)

    logits_sum = torch.zeros((1, total_length, num_classes), device=device)
    logits_count = torch.zeros((1, total_length, 1), device=device)

    for start in range(0, total_length, stride_frames):
        end = min(start + frames_per_inference, total_length)
        chunk = spectrogram[..., start:end]
        chunk_len = end - start
        chunk_mask = torch.zeros((1, chunk_len), dtype=torch.bool, device=device)
        output = model(chunk, src_key_padding_mask=chunk_mask)
        logits_sum[:, start:end, :] += output
        logits_count[:, start:end, :] += 1

    full_logits = logits_sum / logits_count.clamp(min=1)
    raw_preds = torch.argmax(full_logits, dim=-1)[valid_mask].cpu().numpy().astype(np.int64)

    if MEDIAN_FILTER_SIZE > 1 and len(raw_preds) > MEDIAN_FILTER_SIZE:
        raw_preds = median_filter(raw_preds, size=MEDIAN_FILTER_SIZE).astype(np.int64)

    return raw_preds.tolist()


def aggregate_to_windows(
    raw_preds: list[int],
    csv_labels: list[dict],
    n_samples: int,
    samples_trans: int,
    hop_length: int,
    num_classes: int,
    gt_mode: str,
) -> tuple[list[int], list[int]]:
    gt_seq = []
    pred_seq = []

    for sample_start in range(0, n_samples, samples_trans):
        sample_end = min(sample_start + samples_trans, n_samples)
        if (sample_end - sample_start) < samples_trans:
            continue

        if gt_mode == "two_class":
            gt = get_gt_two_class(csv_labels, sample_start, sample_end)
            default_pred = 1
        else:
            gt = get_gt_three_class(csv_labels, sample_start, sample_end)
            default_pred = 2

        start_frame = int(sample_start / hop_length)
        end_frame = int(sample_end / hop_length)
        chunk_preds = raw_preds[start_frame:end_frame]
        pred = class_mode(chunk_preds, num_classes) if chunk_preds else default_pred

        gt_seq.append(gt)
        pred_seq.append(pred)

    return gt_seq, pred_seq


def count_runs_for_class(sequence: list[int], class_id: int, min_len: int) -> int:
    count = 0
    for ev in sequence_to_events(sequence):
        if ev["class"] == class_id and (ev["end"] - ev["start"]) >= min_len:
            count += 1
    return count


def empty_count_stats() -> dict:
    return {
        "files": 0,
        "exhale_gt": 0,
        "exhale_pred": 0,
        "inhale_gt": 0,
        "inhale_pred": 0,
        "bpm_error_list": [],
        "file_rows": [],
    }


def compute_bpm(exhale_count: int, duration_sec: float) -> float:
    if duration_sec <= 0:
        return float("nan")
    return (float(exhale_count) / duration_sec) * 60.0


def compute_bpm_error_stats(bpm_errors: list[float]) -> dict:
    arr = np.array([x for x in bpm_errors if np.isfinite(x)], dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "median_abs": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "within_1": float("nan"),
            "within_2": float("nan"),
        }

    abs_arr = np.abs(arr)
    return {
        "count": int(arr.size),
        "mae": float(abs_arr.mean()),
        "rmse": float(np.sqrt(np.mean(arr ** 2))),
        "bias": float(arr.mean()),
        "median_abs": float(np.median(abs_arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "within_1": float((abs_arr <= 1.0).mean() * 100.0),
        "within_2": float((abs_arr <= 2.0).mean() * 100.0),
    }


def update_counts(
    stats: dict,
    file_name: str,
    gt_seq: list[int],
    pred_seq: list[int],
    is_three_class: bool,
    duration_sec: float,
) -> None:
    exhale_gt = count_runs_for_class(gt_seq, class_id=0, min_len=MIN_RUN_LEN)
    exhale_pred = count_runs_for_class(pred_seq, class_id=0, min_len=MIN_RUN_LEN)

    inhale_gt = count_runs_for_class(gt_seq, class_id=1, min_len=MIN_RUN_LEN) if is_three_class else 0
    inhale_pred = count_runs_for_class(pred_seq, class_id=1, min_len=MIN_RUN_LEN) if is_three_class else 0

    stats["files"] += 1
    stats["exhale_gt"] += exhale_gt
    stats["exhale_pred"] += exhale_pred
    stats["inhale_gt"] += inhale_gt
    stats["inhale_pred"] += inhale_pred

    bpm_gt = compute_bpm(exhale_gt, duration_sec)
    bpm_pred = compute_bpm(exhale_pred, duration_sec)
    bpm_err = bpm_pred - bpm_gt if np.isfinite(bpm_gt) and np.isfinite(bpm_pred) else float("nan")
    stats["bpm_error_list"].append(bpm_err)

    stats["file_rows"].append(
        {
            "file": file_name,
            "exhale_gt": exhale_gt,
            "exhale_pred": exhale_pred,
            "inhale_gt": inhale_gt,
            "inhale_pred": inhale_pred,
            "bpm_gt": bpm_gt,
            "bpm_pred": bpm_pred,
            "bpm_err": bpm_err,
        }
    )


def add_count_summary(lines: list[str], title: str, stats: dict, include_inhale: bool) -> None:
    lines.append("-" * 80)
    lines.append(title)
    lines.append("-" * 80)
    lines.append(f"Files: {stats['files']}")

    ex_gt = stats["exhale_gt"]
    ex_pr = stats["exhale_pred"]
    ex_detected = min(ex_gt, ex_pr)
    ex_missed = max(0, ex_gt - ex_pr)
    ex_false = max(0, ex_pr - ex_gt)

    lines.append("Exhale counts:")
    lines.append(f"  GT total:       {ex_gt}")
    lines.append(f"  Pred total:     {ex_pr}")
    lines.append(f"  Detected(min):  {ex_detected}")
    lines.append(f"  Missed:         {ex_missed}")
    lines.append(f"  False extra:    {ex_false}")

    bpm_stats = compute_bpm_error_stats(stats["bpm_error_list"])
    lines.append("BPM (exhale-based) error stats [Pred - GT]:")
    lines.append(f"  Samples:        {bpm_stats['count']}")
    lines.append(f"  MAE:            {bpm_stats['mae']:.3f}")
    lines.append(f"  RMSE:           {bpm_stats['rmse']:.3f}")
    lines.append(f"  Bias:           {bpm_stats['bias']:.3f}")
    lines.append(f"  Median |err|:   {bpm_stats['median_abs']:.3f}")
    lines.append(f"  Error min/max:  {bpm_stats['min']:.3f} / {bpm_stats['max']:.3f}")
    lines.append(f"  |err| <= 1 BPM: {bpm_stats['within_1']:.1f}%")
    lines.append(f"  |err| <= 2 BPM: {bpm_stats['within_2']:.1f}%")

    if include_inhale:
        in_gt = stats["inhale_gt"]
        in_pr = stats["inhale_pred"]
        in_detected = min(in_gt, in_pr)
        in_missed = max(0, in_gt - in_pr)
        in_false = max(0, in_pr - in_gt)

        lines.append("Inhale counts:")
        lines.append(f"  GT total:       {in_gt}")
        lines.append(f"  Pred total:     {in_pr}")
        lines.append(f"  Detected(min):  {in_detected}")
        lines.append(f"  Missed:         {in_missed}")
        lines.append(f"  False extra:    {in_false}")

    lines.append("Per-file counts:")
    header = "  file | ex_gt ex_pr | bpm_gt bpm_pr bpm_err"
    if include_inhale:
        header += " | in_gt in_pr"
    lines.append(header)
    for row in stats["file_rows"]:
        line = (
            f"  {row['file']} | {row['exhale_gt']:>5} {row['exhale_pred']:>5} | "
            f"{row['bpm_gt']:>6.2f} {row['bpm_pred']:>6.2f} {row['bpm_err']:>7.2f}"
        )
        if include_inhale:
            line += f" | {row['inhale_gt']:>5} {row['inhale_pred']:>5}"
        lines.append(line)
    lines.append("")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(WAV_DIR):
        raise FileNotFoundError(f"Brak WAV_DIR: {WAV_DIR}")
    if not os.path.exists(LABEL_DIR):
        raise FileNotFoundError(f"Brak LABEL_DIR: {LABEL_DIR}")

    three_cfg = ThreeClassConfig.from_yaml(THREE_CLASS_CONFIG_PATH)
    two_cfg = TwoClassConfig.from_yaml(TWO_CLASS_CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading 3-class Transformer...")
    three_model = ThreeClassTransformer(
        n_mels=three_cfg.model.n_mels,
        d_model=three_cfg.model.d_model,
        nhead=three_cfg.model.nhead,
        num_layers=three_cfg.model.num_layers,
        num_classes=three_cfg.model.num_classes,
    ).to(device)
    ckpt_three = torch.load(THREE_CLASS_MODEL_PATH, map_location=device)
    state_three = ckpt_three["model_state_dict"] if "model_state_dict" in ckpt_three else ckpt_three
    if "pos_encoder.pe" in state_three:
        del state_three["pos_encoder.pe"]
    three_model.load_state_dict(state_three, strict=False)
    three_model.eval()

    print("Loading 2-class Transformer...")
    two_model = TwoClassTransformer(
        n_mels=two_cfg.model.n_mels,
        d_model=two_cfg.model.d_model,
        nhead=two_cfg.model.nhead,
        num_layers=two_cfg.model.num_layers,
        num_classes=two_cfg.model.num_classes,
    ).to(device)
    ckpt_two = torch.load(TWO_CLASS_MODEL_PATH, map_location=device)
    state_two = ckpt_two["model_state_dict"] if "model_state_dict" in ckpt_two else ckpt_two
    if "pos_encoder.pe" in state_two:
        del state_two["pos_encoder.pe"]
    two_model.load_state_dict(state_two, strict=False)
    two_model.eval()

    dataset = BreathDataset(
        data_dir=WAV_DIR,
        label_dir=LABEL_DIR,
        sample_rate=three_cfg.data.sample_rate,
        n_mels=three_cfg.data.n_mels,
        n_fft=three_cfg.data.n_fft,
        hop_length=three_cfg.data.hop_length,
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    sample_rate = three_cfg.data.sample_rate
    hop_length = three_cfg.data.hop_length
    samples_trans = int(sample_rate * WINDOW_TRANS)
    frames_per_inference = int(sample_rate * WINDOW_INFERENCE / hop_length)

    stats_three = empty_count_stats()
    stats_two = empty_count_stats()

    num_files = min(LIMIT_WAV, len(dataset)) if LIMIT_WAV else len(dataset)
    print(f"Found {len(dataset)} files (processing {num_files})")

    with torch.inference_mode():
        for i, (spectrogram, _labels, padding_mask) in enumerate(loader):
            if LIMIT_WAV and i >= LIMIT_WAV:
                break

            wav_filename = dataset.wav_files[i]
            base_name = os.path.splitext(wav_filename)[0]
            csv_path = os.path.join(LABEL_DIR, f"{base_name}.csv")

            print(f"[{i + 1}/{num_files}] Processing {wav_filename}...")

            spectrogram = spectrogram.to(device)
            valid_mask = ~padding_mask.to(device)
            valid_frames = int(valid_mask.sum().item())
            n_samples = valid_frames * hop_length
            duration_sec = n_samples / float(sample_rate)
            csv_labels = parse_label_csv(csv_path)

            raw_preds_three = infer_raw_predictions(
                model=three_model,
                spectrogram=spectrogram,
                valid_mask=valid_mask,
                frames_per_inference=frames_per_inference,
                num_classes=3,
                device=device,
            )
            raw_preds_two = infer_raw_predictions(
                model=two_model,
                spectrogram=spectrogram,
                valid_mask=valid_mask,
                frames_per_inference=frames_per_inference,
                num_classes=2,
                device=device,
            )

            gt_three, pred_three = aggregate_to_windows(
                raw_preds=raw_preds_three,
                csv_labels=csv_labels,
                n_samples=n_samples,
                samples_trans=samples_trans,
                hop_length=hop_length,
                num_classes=3,
                gt_mode="three_class",
            )
            gt_two, pred_two = aggregate_to_windows(
                raw_preds=raw_preds_two,
                csv_labels=csv_labels,
                n_samples=n_samples,
                samples_trans=samples_trans,
                hop_length=hop_length,
                num_classes=2,
                gt_mode="two_class",
            )

            pred_three = smooth_short_runs(pred_three, MIN_RUN_LEN)
            pred_two = smooth_short_runs(pred_two, MIN_RUN_LEN)

            update_counts(
                stats_three,
                wav_filename,
                gt_three,
                pred_three,
                is_three_class=True,
                duration_sec=duration_sec,
            )
            update_counts(
                stats_two,
                wav_filename,
                gt_two,
                pred_two,
                is_three_class=False,
                duration_sec=duration_sec,
            )

    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("BREATH COUNTS EVALUATION (TWO TRANSFORMERS)")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Data: {WAV_DIR}")
    lines.append(
        f"WINDOW_TRANS={WINDOW_TRANS}s | WINDOW_INFERENCE={WINDOW_INFERENCE}s | "
        f"INFERENCE_OVERLAP={INFERENCE_OVERLAP} | MIN_RUN_LEN={MIN_RUN_LEN}"
    )
    lines.append("Rule: run length >= MIN_RUN_LEN counts as one breath event.")
    lines.append("Detected=min(GT,Pred), Missed=max(0,GT-Pred), False=max(0,Pred-GT)")
    lines.append("=" * 90)
    lines.append("")

    add_count_summary(lines, "MODEL: Transformer-3Class", stats_three, include_inhale=True)
    add_count_summary(lines, "MODEL: Transformer-2Class", stats_two, include_inhale=False)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\nDone.")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()

