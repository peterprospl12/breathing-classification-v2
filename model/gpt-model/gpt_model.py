import os
import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
import numpy as np

# -------------------------------
# Global constants and parameters
# -------------------------------
DATA_SEQ_FOLDER = "../scripts/data-seq"  # Folder where generated sequences (.wav and .csv) are stored
SAMPLE_RATE = 44100  # All audio is in 44.1 kHz
SNIPPET_DURATION = 0.5  # Duration of snippet to classify (seconds)
SNIPPET_SAMPLES = int(SAMPLE_RATE * SNIPPET_DURATION)  # 22050 samples

# Training hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# Custom Dataset
# -------------------------------
class BreathDataset(Dataset):
    """
    A PyTorch Dataset that loads the generated 30-second sequences and their label CSV files.
    For each labeled phase (with start and end sample indices), it extracts a fixed-length snippet.

    If the phase segment is longer than SNIPPET_SAMPLES, a random contiguous snippet is cropped.
    If it is shorter, the snippet is padded with silence (zeros) to reach SNIPPET_SAMPLES.
    """

    def __init__(self, data_folder):
        """
        Args:
            data_folder (str): Path to the folder containing the generated .wav and .csv files.
        """
        self.data_folder = data_folder
        # List to store tuples: (wav_path, phase_code, start_sample, end_sample)
        self.segments = []
        # Cache to hold loaded audio so that we don't re-read the same file multiple times
        self.audio_cache = {}
        # Assume files are named ours{i}.wav and ours{i}.csv
        for filename in os.listdir(data_folder):
            if filename.endswith('.csv'):
                base = filename[:-4]  # e.g., "ours0"
                csv_path = os.path.join(data_folder, filename)
                wav_path = os.path.join(data_folder, base + ".wav")
                # Check that both files exist
                if not os.path.exists(wav_path):
                    continue
                # Read CSV file (skip header if present)
                with open(csv_path, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    # If header row is detected, remove it
                    if rows and rows[0] == ["phase_code", "start_sample", "end_sample"]:
                        rows = rows[1:]
                    for row in rows:
                        if len(row) != 3:
                            continue
                        phase_code, start_sample, end_sample = int(row[0]), int(row[1]), int(row[2])
                        self.segments.append((wav_path, phase_code, start_sample, end_sample))

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        """
        Returns:
            snippet: Tensor of shape (SNIPPET_SAMPLES,) containing the audio snippet (normalized)
            label: int (0: exhale, 1: inhale, 2: silence)
        """
        wav_path, phase_code, seg_start, seg_end = self.segments[idx]

        # Load audio for this wav file if not already in cache.
        if wav_path not in self.audio_cache:
            # torchaudio.load returns waveform of shape (channels, samples)
            waveform, sr = torchaudio.load(wav_path)
            # Since our audio is mono, squeeze channel dimension
            waveform = waveform.squeeze(0)  # shape: (num_samples,)
            self.audio_cache[wav_path] = waveform
        else:
            waveform = self.audio_cache[wav_path]

        # Extract the segment from the full sequence
        # Ensure that indices are within bounds
        seg_start = max(0, seg_start)
        seg_end = min(waveform.size(0) - 1, seg_end)
        segment = waveform[seg_start:seg_end + 1]  # shape: (segment_length,)
        segment_length = segment.size(0)

        # Decide how to get a fixed-length snippet (SNIPPET_SAMPLES)
        if segment_length >= SNIPPET_SAMPLES:
            # Randomly crop a window of SNIPPET_SAMPLES from the segment
            max_start = segment_length - SNIPPET_SAMPLES
            crop_start = random.randint(0, max_start)
            snippet = segment[crop_start: crop_start + SNIPPET_SAMPLES]
        else:
            # If segment is too short, pad with zeros (silence) at the end
            pad_length = SNIPPET_SAMPLES - segment_length
            snippet = torch.cat([segment, torch.zeros(pad_length)])

        # Normalize the snippet: zero-mean and unit variance.
        snippet = snippet - snippet.mean()
        std = snippet.std()
        snippet = snippet / (std + 1e-9)

        return snippet, phase_code


# -------------------------------
# Model Definition
# -------------------------------
class BreathClassifier(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2, num_classes=3):
        """
        Initializes the BreathClassifier model.

        Args:
            hidden_size (int): Number of hidden units in the LSTM.
            num_layers (int): Number of LSTM layers.
            num_classes (int): Number of output classes (0: exhale, 1: inhale, 2: silence).
        """
        super(BreathClassifier, self).__init__()

        # MelSpectrogram transform:
        # Use 64 Mel bins, FFT size of 1024 and hop length of 512.
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )

        # LSTM network to model the temporal evolution of the Mel spectrogram.
        # The LSTM expects input shape (batch, time, features).
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Dropout layer for regularization.
        self.dropout = nn.Dropout(0.3)

        # Fully connected layer for classification.
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, waveform):
        """
        Forward pass of the model.

        Args:
            waveform (Tensor): Input waveform of shape (batch, SNIPPET_SAMPLES).

        Returns:
            logits (Tensor): Raw class scores of shape (batch, num_classes).
        """
        # Note: We already normalized the waveform in the Dataset.

        # Compute the Mel spectrogram: shape (batch, n_mels, time)
        mel_spec = self.mel_transform(waveform)

        # Apply logarithmic scaling.
        log_mel_spec = torch.log(mel_spec + 1e-9)

        # Transpose to shape (batch, time, n_mels)
        log_mel_spec = log_mel_spec.transpose(1, 2)

        # LSTM: output shape (batch, time, hidden_size)
        lstm_out, _ = self.lstm(log_mel_spec)

        # Use the output from the last time step as the snippet representation.
        last_time_step = lstm_out[:, -1, :]

        # Apply dropout and fully-connected layer.
        out = self.dropout(last_time_step)
        logits = self.fc(out)
        return logits


# -------------------------------
# Training loop and metrics
# -------------------------------
def train_model(model, dataloader, optimizer, criterion, num_epochs):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(DEVICE)  # shape: (batch, SNIPPET_SAMPLES)
            labels = labels.to(DEVICE)  # shape: (batch,)

            optimizer.zero_grad()
            outputs = model(inputs)  # shape: (batch, num_classes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # Compute accuracy for the batch
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100

        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")


def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total * 100
    print(f"Evaluation | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


# -------------------------------
# Main training script
# -------------------------------
def main():
    # Create the dataset and dataloader
    dataset = BreathDataset(DATA_SEQ_FOLDER)

    # Shuffle and split into training and (optionally) validation sets.
    # Here, we use 80% for training and 20% for evaluation.
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    split = int(0.8 * dataset_size)
    train_indices, eval_indices = indices[:split], indices[split:]

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    eval_subset = torch.utils.data.Subset(dataset, eval_indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_subset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the model, criterion, and optimizer.
    model = BreathClassifier(hidden_size=128, num_layers=2, num_classes=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    train_model(model, train_loader, optimizer, criterion, NUM_EPOCHS)

    print("Evaluating model on validation set:")
    evaluate_model(model, eval_loader, criterion)

    # Optionally, save the trained model.
    torch.save(model.state_dict(), "breath_classifier.pth")
    print("Model saved as breath_classifier.pth")


if __name__ == "__main__":
    main()
