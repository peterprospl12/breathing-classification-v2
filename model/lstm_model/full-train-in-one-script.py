import os
from datetime import datetime

from scipy.io.wavfile import read
import csv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np
import librosa
import time
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from model_classes import AudioClassifierLSTM as AudioClassifier
import torch.nn as nn

REFRESH_TIME = 0.1  # seconds
BATCH_SIZE = 16

# Directories with data
data_dir = '../../train-sequences-test'

# Function to load labels from csv file
def load_labels(csv_file_v):
    labels_v = []
    with open(csv_file_v, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Pomijamy nagłówek
        for row in reader:
            if row[0] == 'silence':
                labels_v.append((2, int(row[1]), int(row[2])))
            elif row[0] == 'inhale':
                labels_v.append((1, int(row[1]), int(row[2])))
            elif row[0] == 'exhale':
                labels_v.append((0, int(row[1]), int(row[2])))
    return labels_v

# Function to get the label for a given time
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
    if not os.path.exists(csv_file):
        continue

    # Load audio and labels
    sr, y = read(wav_file)

    if sr != 44100:
        # raise Exception("Sampling rate is not 44100 its {}".format(sr))
        print("Sampling rate is not 44100 its {}".format(sr))

    labels = load_labels(csv_file)

    # Calculate chunk size
    chunk_size = int(sr * REFRESH_TIME)

    # List of MFCCs for every data sequence (it will be a list of lists of tuples (mfcc coefficients, label))
    mfcc_sequence = []

    # Iterate through every 0.25s audio chunk
    for i in range(0, len(y), chunk_size):
        frame = y[i:i + chunk_size]
        if len(frame) == chunk_size:
            frame = frame.astype(np.float32)
            frame /= np.iinfo(np.int16).max
            mfcc = librosa.feature.mfcc(y=frame, sr=sr)
            mfcc_mean = mfcc.mean(axis=1)
            label = get_label_for_time(labels, i, i + chunk_size)
            mfcc_sequence.append((mfcc_mean, label))

    if mfcc_sequence:
        train_data.append(mfcc_sequence)

# Check length of every sequence

lengths = [len(seq) for seq in train_data]
print("Min length: ", min(lengths))
print("Max length: ", max(lengths))
print(lengths)

# Split data into train and validation sets
train_data, val_data = train_test_split(train_data, test_size=0.2)

# DataLoader and collate function
from model_classes import AudioDataset
import torch

train_dataset = AudioDataset(train_data)
val_dataset = AudioDataset(val_data)

def collate_fn(batch):
    sequences, labels_t = zip(*batch)
    lengths_t = [seq.size(0) for seq in sequences]
    max_length = max(lengths_t)
    padded_sequences = torch.zeros(len(sequences), max_length, 20)
    padded_labels = torch.zeros(len(sequences), max_length, dtype=torch.long)
    for j, seq in enumerate(sequences):
        padded_sequences[j, :seq.size(0), :] = seq
        padded_labels[j, :len(labels_t[j])] = labels_t[j]
    return padded_sequences, padded_labels

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

REFRESH_TIME = 0.25  # Refresh time in seconds in future realtime
NUM_EPOCHS = 100  # Number of epochs (the more epoch the better model, but it takes more time)
PATIENCE_TIME = 10  # Number of epochs without improvement in validation accuracy that will stop training
LEARNING_RATE = 0.001  # Learning rate
BATCH_SIZE = 16  # Batch size (amount of sequences in one batch)

# Check if CUDA is available (learning on GPU is much faster)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

total_time = time.time()
start_time = time.time()

# Create model object
print("Creating model...")
model = AudioClassifier()
model = model.to(device)
print("Model created, time: ", time.time() - start_time)

# Define loss function and optimizer (network parameters)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# These are just for early stopping
best_val_accuracy = 0.0
early_stopping_counter = 0

print("Training model...")
start_time = time.time()

# Iterate through epochs
for epoch in range(NUM_EPOCHS):

    # Enable training on model object
    model.train()

    # Initialize running loss and accuracy
    running_loss = 0.0
    running_accuracy = 0.0
    # It's just a fancy progress bar in console
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}', unit='batch')

    # Iterate through batches
    for inputs, labels in progress_bar:

        # Move inputs and labels to the device (GPU or CPU)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Jeśli model zwraca więcej niż jedną wartość, przypisz odpowiednią wartość do outputs
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        # Flattening outputs and labels from [batch_size, max_length, num_classes]
        outputs = outputs.view(-1, outputs.size(-1))  # Flattening to [batch_size * max_length, num_classes]
        labels = labels.view(-1)  # Flattening to [batch_size * max_length]

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass (calculate gradients)
        loss.backward()

        # Update weights according to the calculated gradients
        optimizer.step()

        # Calculate running loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        running_accuracy += accuracy_score(labels.cpu(), predicted.cpu())

        # Update progress bar
        progress_bar.set_postfix(loss=running_loss / len(progress_bar),
                                  accuracy=running_accuracy / len(progress_bar))

    # Print the loss and accuracy for the epoch
    # print('Train Loss: {:.4f}, Train Accuracy: {:.4f}'.format(running_loss / len(train_loader),
    #                                                           running_accuracy / len(train_loader)))

    # After training on the whole training set, we can evaluate the model on the validation set
    model.eval()
    val_running_loss = 0.0
    val_running_accuracy = 0.0

    # We don't need to calculate gradients during validation
    with torch.no_grad():

        # Iterate through validation set
        for inputs, labels in val_loader:

            # Move inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Jeśli model zwraca więcej niż jedną wartość, przypisz odpowiednią wartość do outputs
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # As previous, we need to flatten outputs and labels
            outputs = outputs.view(-1, outputs.size(-1)) # Flattening to [batch_size * max_length, num_classes]
            labels = labels.view(-1) # Flattening to [batch_size * max_length]

            # Calculate loss
            loss = criterion(outputs, labels)

            # Calculate running loss (cumulative loss over batches) and add current epoch's accuracy to the running (cumulative) accuracy
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_running_accuracy += accuracy_score(labels.cpu(), predicted.cpu())

    # Calculate cumulative loss and accuracy for the validation set
    avg_val_loss = val_running_loss / len(val_loader)
    avg_val_accuracy = val_running_accuracy / len(val_loader)

    # And print it
    print('Val Loss: {:.4f}, Val Accuracy: {:.4f}'.format(avg_val_loss, avg_val_accuracy))

    # Learning rate scheduler (changing learning rate during training)
    scheduler.step()

    # Early stopping (if there is no improvement in validation accuracy for PATIENCE_TIME epochs, we stop training)
    if avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= PATIENCE_TIME:
            print("Early stopping triggered. No improvement in validation accuracy.")
            break

# And print final results
print('Finished Training, time: ', time.time() - start_time)
print('Saving model...')
start_time = time.time()
#TODO
torch.save(model.state_dict(), f'audio_lstm_classifier_{datetime.now()}.pth')
print("Model saved, time: ", time.time() - start_time)
print("Finished, Total time: ", time.time() - total_time)