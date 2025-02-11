"""

Same script as in model-train-rnn.ipynb, but all segments are in one script, because while changing AudioClassifier, sometimes Jupyter cache the class and do not update it while executing cell again

"""

from model_classes import AudioDataset
import torch
import os
from scipy.io.wavfile import read
import csv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np
import librosa
import time
import torch.optim as optim
import torch.nn as nn
from model_classes import AudioClassifierLSTM as AudioClassifier

REFRESH_TIME = 0.25  # seconds
BATCH_SIZE = 16
N_MFCC = 20

# Directories with data
data_dir = '../../sequences'
model_file_name = 'model_lstm'

# Function to load labels from csv file to list of tuples (label, start_frame, end_frame)
def load_labels(csv_file_v):
    labels_v = []
    with open(csv_file_v, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            if row[0] == 'silence':
                labels_v.append((2, int(row[1]), int(row[2])))  # 2: silence
            elif row[0] == 'inhale':
                labels_v.append((1, int(row[1]), int(row[2])))  # 1: inhale
            elif row[0] == 'exhale':
                labels_v.append((0, int(row[1]), int(row[2])))  # 0: exhale
    return labels_v

# Function to get the label for a given part of recording (from start_frame to end_frame)
def get_label_for_time(labels_v, start_frame, end_frame):
    label_counts = [0, 0, 0]  # 0: exhale, 1: inhale, 2: silence

    for label_it, start, end in labels_v:
        if start < end_frame and end > start_frame:
            overlap_start = max(start, start_frame)
            overlap_end = min(end, end_frame)
            overlap_length = overlap_end - overlap_start
            label_counts[label_it] += overlap_length

    return label_counts.index(max(label_counts))

# Creating list of files
wav_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.wav')]
train_data = []

# Main loop to preprocess data into MFCCs
for wav_file in wav_files:
    csv_file = wav_file.replace('.wav', '.csv')

    # Ensure that there is a corresponding CSV file
    if not os.path.exists(csv_file):
        continue

    # Load audio and labels
    sr, y = read(wav_file)

    # Throw error if sampling rate is not 44100, recording is not in mono or dtype is not int16
    if sr != 44100:
        raise Exception("Sampling rate is not 44100. Make sure you have used right sequence creator.")
    if y.dtype != np.int16:
        raise Exception("Data type is not int16. Make sure you have used right sequence creator.")
    if y.ndim != 1:
        raise Exception("Audio is not mono. Make sure you have used right sequence creator.")

    # Load labels from CSV file
    labels = load_labels(csv_file)

    # Calculate chunk size
    chunk_size = int(sr * REFRESH_TIME)

    # List of MFCCs for every data sequence (it will be a list of lists of tuples (mfcc coefficients, label))
    mfcc_sequence = []

    # Iterate through every 0.25s audio chunk
    for i in range(0, len(y), chunk_size):
        # Get frame's samples
        frame = y[i:i + chunk_size]

        # Ensure that the frame has the right size
        if len(frame) == chunk_size:
            # Conversion to float32 from int16
            if frame.dtype != np.int16:
                raise Exception("Data type is not int16. Make sure you have used right sequence creator.")
            frames_float32 = frame.astype(np.float32) / np.iinfo(np.int16).max

            # Make sure that frame is mono, 44100 Hz and converted to float32
            if frames_float32.ndim != 1:
                raise Exception("Audio is not mono. Make sure you have used right sequence creator.")
            if frames_float32.dtype != np.float32:
                raise Exception("Data type is not float32. Make sure you have used right sequence creator.")
            if sr != 44100:
                raise Exception("Sampling rate is not 44100. Make sure you have used right sequence creator.")

            # Calculate MFCCs
            mfcc = librosa.feature.mfcc(
                y=frames_float32,
                sr=sr,
                n_mfcc=N_MFCC,
                n_mels=40,
                fmin=20,
                fmax=8000,
                lifter=22,
                norm="ortho"
            )

            delta_mfcc = librosa.feature.delta(mfcc)

            delta2_mfcc = librosa.feature.delta(mfcc, order=2)

            combined_features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])

            # Because function will return x times 13 MFCCs, we will calculate mean of them (size of mfcc above is [13, x])
            features = combined_features.mean(axis=1)

            # Get label for the frame
            label = get_label_for_time(labels, i, i + chunk_size)

            # Append MFCCs and label to the sequence (we append tuple of a ndarray of length 13 and a label)
            mfcc_sequence.append((features, label))

    train_data.append(mfcc_sequence)  # Append sequence to the list of sequences

# Ensure that all sequences have the same length
length = len(train_data[0])
for sequence in train_data:
    if len(sequence) != length:
        raise Exception("Sequences have different lengths")

# Split data into train and validation sets
train_data, val_data = train_test_split(train_data, test_size=0.2)

train_dataset = AudioDataset(train_data)
val_dataset = AudioDataset(val_data)

# DataLoader and collate function (collate function is used to pad sequences to the same length, but our sequences should have the same length)
def collate_fn(batch):
    sequences, labels_t = zip(*batch)
    lengths_t = [seq.size(0) for seq in sequences]
    max_length = max(lengths_t)
    padded_sequences = torch.zeros(len(sequences), max_length, N_MFCC * 3)
    padded_labels = torch.zeros(len(sequences), max_length, dtype=torch.long)
    for j, seq in enumerate(sequences):
        padded_sequences[j, :seq.size(0), :] = seq
        padded_labels[j, :len(labels_t[j])] = labels_t[j]
    return padded_sequences, padded_labels

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

NUM_EPOCHS = 100
PATIENCE_TIME = 10
LEARNING_RATE = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

model = AudioClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

best_val_accuracy = 0.0
val_loss_on_best_val_acc = 0.0
train_loss_on_best_val_acc = 0.0
train_acc_on_best_val_acc = 0.0

early_stopping_counter = 0

total_start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()

    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)  # Shape: [batch, time_steps, features]
        labels = labels.to(device)  # Shape: [batch, time_steps]

        optimizer.zero_grad()
        outputs, _ = model(inputs)  # outputs.shape: [batch, time_steps, num_classes]

        # Flattening to [batch * time_steps, num_classes]
        outputs_flat = outputs.view(-1, outputs.size(-1))
        labels_flat = labels.view(-1)  # [batch * time_steps]

        loss = criterion(outputs_flat, labels_flat)
        loss.backward()
        optimizer.step()

        # Calculate loss and accuracy
        _, predicted = torch.max(outputs_flat, 1)  # Get the predicted class (index of the maximum logit) for each audio segment
        train_correct += (predicted == labels_flat).sum().item()  # Count how many predictions match the true labels in this batch
        train_total += labels_flat.size(0)  # Update the total number of audio segments processed so far
        train_loss += loss.item()  # Accumulate the loss for this batch to calculate the average loss later

    avg_train_loss = train_loss / len(train_loader)
    train_acc = train_correct / train_total

    # Switch the model to evaluation mode (turns off dropout, batch norm, etc.)
    model.eval()

    # Initialize variables to track total correct predictions, total samples, and accumulated loss for validation
    val_loss, val_correct, val_total = 0.0, 0, 0

    # Disable gradient calculation for validation (saves memory and speeds up computation)
    with torch.no_grad():
        # Iterate through batches in the validation set
        for inputs, labels in val_loader:
            inputs = inputs.to(device)  # Shape: [batch, time_steps, features]
            labels = labels.to(device)  # Shape: [batch, time_steps]

            # Forward pass: compute model predictions
            outputs, _ = model(inputs)  # outputs.shape: [batch, time_steps, num_classes]

            # Flattening to [batch * time_steps, num_classes]
            outputs_flat = outputs.view(-1, outputs.size(-1))
            labels_flat = labels.view(-1)  # [batch * time_steps]

            # Calculate loss (how far the model's predictions are from the correct answers)
            loss = criterion(outputs_flat, labels_flat)
            val_loss += loss.item()  # Accumulate the loss to calculate the average loss later

            # Calculate accuracy for this batch
            _, predicted = torch.max(outputs_flat, 1)  # Get the predicted class (index of the maximum logit) for each audio segment
            val_correct += (predicted == labels_flat).sum().item()  # Count how many predictions match the true labels in this batch
            val_total += labels_flat.size(0)  # Update the total number of audio segments processed so far

    # Calculate the average validation loss and accuracy for the entire epoch
    avg_val_loss = val_loss / len(val_loader)  # Average loss = total loss / number of batches
    val_acc = val_correct / val_total  # Accuracy = correct predictions / total samples


    # Update and early stopping
    scheduler.step()

    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        train_acc_on_best_val_acc = train_acc
        val_loss_on_best_val_acc = avg_val_loss
        train_loss_on_best_val_acc = avg_train_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), f'{model_file_name}.pth')
    else:
        early_stopping_counter += 1

    epoch_time = time.time() - epoch_start_time
    print(f'Epoch {epoch+1}/{NUM_EPOCHS} [{epoch_time:.2f}s]')
    print(f' Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}')
    print(f' Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}\n')

    if early_stopping_counter >= PATIENCE_TIME:
        print("Early stopping!")
        break

# Save the final model after training completes

print(f"Final model saved to {model_file_name}.pth")

# Print metrics for the final model
print("\nFinal Model Metrics:")
print(f' Train Loss: {train_loss_on_best_val_acc:.4f}, Train Acc: {train_acc_on_best_val_acc:.4f}')
print(f' Val Loss: {val_loss_on_best_val_acc:.4f}, Val Acc: {best_val_accuracy:.4f}')

total_time = time.time() - total_start_time
print(f'Total training time: {total_time:.2f}s')