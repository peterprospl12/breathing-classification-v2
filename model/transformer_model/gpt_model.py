import os
import csv
import random
import torch
import torchaudio
from torch import optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torchaudio

# -------------------------------
# Global constants and parameters
# -------------------------------
DATA_SEQ_FOLDER = "../../scripts/data-seq"  # Folder with generated sequences (.wav and .csv)
SAMPLE_RATE = 44100  # All audio is in 44.1 kHz
SNIPPET_DURATION = 10
SNIPPET_SAMPLES = int(SAMPLE_RATE * SNIPPET_DURATION)  # For 30 s, e.g., 30 * 44100 = 1,323,000 samples
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
NUM_EPOCHS = 40
LEARNING_RATE = 1e-3
class BreathDataset(Dataset):
    """
    Dataset that loads x-second sequences and their corresponding labels.
    For each segment (specified by CSV), it extracts exactly SNIPPET_SAMPLES samples:
      - If the segment is longer – randomly selects a x s fragment.
      - If it is shorter – pads with zeros (silence).
    Finally, normalization (zero-mean and unit variance) is applied.
    """
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.segments = []  # List of tuples: (wav_path, phase_code, start_sample, end_sample)
        self.audio_cache = {}
        # Assume files are named e.g., ours0.wav and ours0.csv
        for filename in os.listdir(data_folder):
            if filename.endswith('.csv'):
                base = filename[:-4]
                csv_path = os.path.join(data_folder, filename)
                wav_path = os.path.join(data_folder, base + ".wav")
                if not os.path.exists(wav_path):
                    continue
                with open(csv_path, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
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
        wav_path, phase_code, seg_start, seg_end = self.segments[idx]
        if wav_path not in self.audio_cache:
            waveform, sr = torchaudio.load(wav_path)
            waveform = waveform.squeeze(0)  # Assume audio is mono.
            self.audio_cache[wav_path] = waveform
        else:
            waveform = self.audio_cache[wav_path]

        # Ensure indices are correct:
        seg_start = max(0, seg_start)
        seg_end = min(waveform.size(0) - 1, seg_end)
        segment = waveform[seg_start:seg_end + 1]  # shape: (segment_length,)
        segment_length = segment.size(0)

        # Extract or pad to a fixed length (30 seconds)
        if segment_length >= SNIPPET_SAMPLES:
            max_start = segment_length - SNIPPET_SAMPLES
            crop_start = random.randint(0, max_start)
            snippet = segment[crop_start: crop_start + SNIPPET_SAMPLES]
        else:
            pad_length = SNIPPET_SAMPLES - segment_length
            snippet = torch.cat([segment, torch.zeros(pad_length)])

        # Data augmentation: add noise
        noise = torch.randn_like(snippet) * 0.005
        snippet = snippet + noise

        # Data augmentation: pitch shift
        pitch_shift = random.uniform(-2, 2)
        snippet = torchaudio.transforms.Resample(orig_freq=SAMPLE_RATE, new_freq=SAMPLE_RATE * (2 ** (pitch_shift / 12)))(snippet)

        # Normalization (zero-mean, unit variance)
        snippet = snippet - snippet.mean()
        snippet = snippet / (snippet.std() + 1e-9)
        return snippet, phase_code



# -------------------------------
# Model Definition
# -------------------------------
class BreathClassifier(nn.Module):
    def __init__(self, hidden_size=256, num_layers=3, num_classes=3):
        super(BreathClassifier, self).__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, waveform):
        """
        waveform: Tensor of shape (batch, SNIPPET_SAMPLES) – here SNIPPET_SAMPLES = 30 * 44100
        """
        # Compute Mel spectrogram
        mel_spec = self.mel_transform(waveform)  # (batch, 64, time)
        log_mel_spec = torch.log(mel_spec + 1e-9)
        log_mel_spec = log_mel_spec.transpose(1, 2)  # (batch, time, 64)

        # Process through LSTM
        lstm_out, _ = self.lstm(log_mel_spec)  # (batch, time, hidden_size)

        # Calculate how many frames correspond to 0.5 s.
        num_frames = int(0.5 * SAMPLE_RATE / self.mel_transform.hop_length)
        # Ensure we do not exceed the sequence length:
        num_frames = min(num_frames, lstm_out.size(1))

        # Average the last num_frames frames – this represents the last 0.5 s
        last_time_window = lstm_out[:, -num_frames:, :]  # (batch, num_frames, hidden_size)
        aggregated = last_time_window.mean(dim=1)          # (batch, hidden_size)

        # Dropout and classification
        out = self.dropout(aggregated)
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
    model = BreathClassifier(hidden_size=256, num_layers=3, num_classes=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    train_model(model, train_loader, optimizer, criterion, NUM_EPOCHS)

    print("Evaluating model on validation set:")
    evaluate_model(model, eval_loader, criterion)

    # Save the trained model.
    torch.save(model.state_dict(), "breath_classifier6.pth")
    print("Model saved as breath_classifier6.pth")


if __name__ == "__main__":
    main()
