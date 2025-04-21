import os
import csv
import torch
import torchaudio
import torch.nn as nn
import argparse
import numpy as np

# Global constants – must match those used during training
SAMPLE_RATE = 44100
SNIPPET_DURATION = 0.5  # seconds
SNIPPET_SAMPLES = int(SAMPLE_RATE * SNIPPET_DURATION)  # e.g., 22050 samples

# Mapping of phase codes to human‑readable labels.
PHASE_LABELS = {0: "exhale", 1: "inhale", 2: "silence", -1: "unknown"}


# -------------------------------
# Model definition – must match training configuration
# -------------------------------
class BreathClassifier(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2, num_classes=3):
        """
        Initializes the BreathClassifier model.

        Args:
            hidden_size (int): Number of hidden units in the LSTM.
            num_layers (int): Number of LSTM layers.
            num_classes (int): Number of output classes.
        """
        super(BreathClassifier, self).__init__()

        # MelSpectrogram transform: 64 Mel bins, FFT size 1024, hop length 512.
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )

        # LSTM network: expects input of shape (batch, time, features)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, waveform):
        """
        Forward pass of the model.

        Args:
            waveform (Tensor): Input tensor of shape (batch, SNIPPET_SAMPLES).

        Returns:
            logits (Tensor): Raw class scores of shape (batch, num_classes).
        """
        # Compute the Mel spectrogram (batch, n_mels, time)
        mel_spec = self.mel_transform(waveform)
        # Log-scale the spectrogram
        log_mel_spec = torch.log(mel_spec + 1e-9)
        # Transpose to shape (batch, time, n_mels)
        log_mel_spec = log_mel_spec.transpose(1, 2)
        # Pass through LSTM
        lstm_out, _ = self.lstm(log_mel_spec)
        # Use the last time step for classification
        last_time_step = lstm_out[:, -1, :]
        out = self.dropout(last_time_step)
        logits = self.fc(out)
        return logits


# -------------------------------
# Utility function to split audio into fixed-length snippets
# -------------------------------
def split_audio_into_snippets(waveform, snippet_samples):
    """
    Splits a 1D waveform tensor into non-overlapping snippets of length snippet_samples.
    If the waveform length is not an exact multiple of snippet_samples, the last snippet is padded with zeros.

    Args:
        waveform (Tensor): 1D tensor of audio samples.
        snippet_samples (int): Number of samples per snippet.

    Returns:
        snippets (Tensor): Tensor of shape (num_snippets, snippet_samples).
    """
    total_samples = waveform.shape[0]
    num_snippets = int(np.ceil(total_samples / snippet_samples))

    # Pad the waveform if necessary
    pad_length = num_snippets * snippet_samples - total_samples
    if pad_length > 0:
        waveform = torch.cat([waveform, torch.zeros(pad_length)])

    # Reshape into snippets
    snippets = waveform.unfold(0, snippet_samples, snippet_samples)
    # snippets shape: (num_snippets, snippet_samples)
    return snippets


# -------------------------------
# Utility function to assign ground truth label to a snippet based on its center sample
# -------------------------------
def get_ground_truth_for_snippet(center_sample, ground_truth_intervals):
    """
    Given a sample index (center of snippet), determine the ground truth label.

    Args:
        center_sample (int): The center sample index of the snippet.
        ground_truth_intervals (list): List of tuples (phase_code, start_sample, end_sample)
                                       loaded from the CSV.

    Returns:
        phase_code (int): The ground truth phase code if found; otherwise, -1.
    """
    for phase_code, start_sample, end_sample in ground_truth_intervals:
        if start_sample <= center_sample <= end_sample:
            return phase_code
    return -1  # Not found


# -------------------------------
# Main evaluation function that compares model predictions with ground truth from CSV
# -------------------------------
def evaluate_from_csv(model, audio_path, csv_path, device):
    """
    Loads an audio file and its corresponding ground-truth CSV file,
    splits the entire audio evenly into 0.5-second snippets, obtains predictions,
    and then compares each snippet's prediction with the ground truth determined from CSV.

    Args:
        model (nn.Module): The pre-trained BreathClassifier.
        audio_path (str): Path to the audio (.wav) file.
        csv_path (str): Path to the CSV file with ground-truth labels.
        device (torch.device): Device to run model on.

    Returns:
        overall_accuracy (float): Percentage of correct predictions over snippets with valid ground truth.
    """
    # Load the audio file
    waveform, sr = torchaudio.load(audio_path)

    # Convert to mono if necessary
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)  # shape: (num_samples,)

    # Resample if needed
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

    total_samples = waveform.shape[0]
    # Split the entire audio into fixed 0.5-second snippets
    snippets = split_audio_into_snippets(waveform, SNIPPET_SAMPLES)
    num_snippets = snippets.shape[0]

    # Load the CSV file (ground truth intervals)
    ground_truth = []  # list of tuples: (phase_code, start_sample, end_sample)
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        # Check if first row is header and skip it
        if rows and rows[0] == ["phase_code", "start_sample", "end_sample"]:
            rows = rows[1:]
        for row in rows:
            if len(row) != 3:
                continue
            phase_code, start_sample, end_sample = int(
                row[0]), int(row[1]), int(row[2])

            # Adjust start and end samples based on resampling
            if sr != SAMPLE_RATE:
                scale_factor = SAMPLE_RATE / sr
                start_sample = int(start_sample * scale_factor)
                end_sample = int(end_sample * scale_factor)

            ground_truth.append((phase_code, start_sample, end_sample))

    correct = 0
    total_valid = 0
    results = []  # to store per-snippet results

    model.eval()
    with torch.no_grad():
        # Normalize all snippets (zero-mean, unit variance)
        snippets = snippets - snippets.mean(dim=1, keepdim=True)
        std = snippets.std(dim=1, keepdim=True)
        snippets = snippets / (std + 1e-9)

        inputs = snippets.to(device)  # shape: (num_snippets, SNIPPET_SAMPLES)
        logits = model(inputs)
        _, preds = torch.max(logits, 1)

    # For each snippet, compute start/end sample and determine ground truth from CSV based on the snippet's center.
    for i in range(num_snippets):
        start_sample = i * SNIPPET_SAMPLES
        end_sample = start_sample + SNIPPET_SAMPLES - 1
        center_sample = (start_sample + end_sample) // 2
        gt_code = get_ground_truth_for_snippet(center_sample, ground_truth)
        pred_code = int(preds[i].item())
        is_correct = (pred_code == gt_code) if gt_code != -1 else None
        if gt_code != -1:
            total_valid += 1
            if is_correct:
                correct += 1
        results.append(
            (pred_code, gt_code, start_sample, end_sample, is_correct))

    accuracy = (correct / total_valid) * 100 if total_valid > 0 else 0.0

    # Print per-snippet results
    print("Snippet-wise results (for fixed 0.5s segments):")
    print("Snippet, Predicted, GroundTruth, StartSample, EndSample, Correct")
    for idx, (pred_code, gt_code, start, end, is_corr) in enumerate(results):
        print(f"{idx + 1}, {PHASE_LABELS[pred_code]} ({pred_code}), {PHASE_LABELS[gt_code]} ({gt_code}), "
              f"{start}, {end}, {is_corr}")

    print(
        f"\nOverall accuracy (over snippets with valid ground truth): {accuracy:.2f}% ({correct}/{total_valid})")
    return accuracy


# -------------------------------
# Main command-line interface
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate fixed 0.5s breath phase predictions against ground truth CSV.")
    parser.add_argument("--audio", type=str, required=True,
                        help="Path to the audio file (.wav)")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to the CSV file with ground-truth labels")
    parser.add_argument("--model", type=str, default="breath_classifier.pth",
                        help="Path to the saved model file")
    args = parser.parse_args()

    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model and load the saved weights
    model = BreathClassifier(
        hidden_size=128, num_layers=2, num_classes=3).to(device)
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file '{args.model}' not found.")
    model.load_state_dict(torch.load(args.model, map_location=device))

    # Evaluate the model predictions against the ground truth using fixed 0.5s snippets
    evaluate_from_csv(model, args.audio, args.csv, device)
