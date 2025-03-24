"""
This script preprocess the data-raw into MFCCs and labels. It also creates a DataLoader object for training and validation sets.

The script divides data-sequences into 0.25s chunks and calculates MFCCs for every chunk. Then, it assigns a label to every chunk based on the labels from the CSV file. If a chunk has both sample from two different classes, the label is assigned based on the majority of samples in the chunk.

The goal is to create a dataset, and final shape of output is a list of data-sequences. Every sequence is a list of tuples (MFCCs, label). The DataLoader object will be used to iterate through the dataset during training.

Most important parameters of this script is:
REFRESH_TIME - length of one classification window in seconds
BATCH_SIZE - batch size for DataLoader
data_dir - directory with training and validation data-raw (there must be data-sequences in directories, ideally created with create_sequences.py script)
"""
from model_classes import AudioDataset
import os
from scipy.io.wavfile import read
import csv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np
import librosa
import time
import torch
import torch.optim as optim
import torch.nn as nn
from model_classes import AudioClassifierLSTM as AudioClassifier

REFRESH_TIME = 0.25  # seconds
BATCH_SIZE = 16

# Directories with data-raw
data_dir = '../../data-sequences'
model_file_name = 'model_lstm.pth'

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

# Main loop to preprocess data-raw into MFCCs
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
    if y.ndim > 1 and y.shape[1] > 1:
        raise Exception("Audio is not mono. Make sure you have used right sequence creator.")

    # Load labels from CSV file
    labels = load_labels(csv_file)

    # Calculate chunk size
    chunk_size = int(sr * REFRESH_TIME)

    # List of MFCCs for every data-raw sequence (it will be a list of lists of tuples (mfcc coefficients, label))
    mfcc_sequence = []

    # Iterate through every 0.25s audio chunk
    for i in range(0, len(y), chunk_size):
        # Get frame's samples
        frame = y[i:i + chunk_size]

        # Ensure that the frame has the right size
        if len(frame) == chunk_size:
            # Make sure that frame is mono, 44100 Hz and in int16 format
            if frame.dtype != np.int16:
                raise Exception("Data type is not int16. Make sure you have used right sequence creator.")
            if sr != 44100:
                raise Exception("Sampling rate is not 44100. Make sure you have used right sequence creator.")
            if frame.ndim > 1 and frame.shape[1] > 1:
                raise Exception("Audio is not mono. Make sure you have used right sequence creator.")

            # Conversion to float32 from int16
            frames_float32 = frame.astype(np.float32) / np.iinfo(np.int16).max

            # Make sure that frame is mono, 44100 Hz and converted to float32
            if frames_float32.ndim > 1 and frames_float32.shape[1] > 1:
                raise Exception("Audio is not mono. Make sure you have used right sequence creator.")
            if frames_float32.dtype != np.float32:
                raise Exception("Data type is not float32. Make sure you have used right sequence creator.")
            if sr != 44100:
                raise Exception("Sampling rate is not 44100. Make sure you have used right sequence creator.")

            # # Perform Short-Time Fourier Transform (STFT) to get spectrogram
            # stft = librosa.stft(frames_float32, n_fft=512, hop_length=256)
            # spectrogram = np.abs(stft)
            #
            # # Convert spectrogram to decibels
            # log_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
            #
            # # Convert amplitude to dB
            # features = log_spectrogram.mean(axis=1)

            # Calculate mel-spectrogram with larger n_fft (e.g., 1024) and specify the number of mel bands (e.g., 128)
            mel_spec = librosa.feature.melspectrogram(
                y=frames_float32,
                sr=44100,
                n_fft=1024,  # Larger FFT window
                hop_length=512,
                n_mels=40  # Number of mel bands
            )

            # Convert amplitude to decibel scale
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

            # Extract features as the mean value for each mel band
            features = log_mel_spec.mean(axis=1)

            # Get label for the frame
            label = get_label_for_time(labels, i, i + chunk_size)

            # Append MFCCs and label to the sequence (we append tuple of a ndarray of length 20 and a label)
            mfcc_sequence.append((features, label))

    train_data.append(mfcc_sequence)  # Append sequence to the list of data-sequences

# Ensure that all data-sequences have the same length
length = len(train_data[0])
for sequence in train_data:
    if len(sequence) != length:
        raise Exception("Sequences have different lengths")

# Split data-raw into train and validation sets
train_data, val_data = train_test_split(train_data, test_size=0.2)

train_dataset = AudioDataset(train_data)
val_dataset = AudioDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

architecture = 'LSTM'

NUM_EPOCHS = 100
PATIENCE_TIME = 10
LEARNING_RATE = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

model = AudioClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
        torch.save(model.state_dict(), model_file_name)
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

print(model_file_name)

# Print metrics for the final model
print("\nFinal Model Metrics:")
print(f' Train Loss: {train_loss_on_best_val_acc:.4f}, Train Acc: {train_acc_on_best_val_acc:.4f}')
print(f' Val Loss: {val_loss_on_best_val_acc:.4f}, Val Acc: {best_val_accuracy:.4f}')

total_time = time.time() - total_start_time
print(f'Total training time: {total_time:.2f}s')