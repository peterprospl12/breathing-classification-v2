import os
import random
import csv
import numpy as np
from scipy.io.wavfile import read, write

# Define the folder for sequences
sequence_folder = '../train-sequences'
os.makedirs(sequence_folder, exist_ok=True)

# Directories with audio files
folders = {
    'inhale': '../data-ours/inhale',
    'exhale': '../data-ours/exhale',
    'silence': '../data-ours/silence'
}

# Sequence length and max length of a single class (in seconds)
MAX_LENGTH = 30         # Full sequence: 30 seconds
MAX_CLASS_LENGTH = 3    # Single fragment: 3 seconds

# Function to load a wav file as a numpy array of samples (mono)
def load_wav(file_path_v):
    rate, data = read(file_path_v)
    # If the data is stereo (2 channels), average or take one channel:
    if data.ndim > 1 and data.shape[1] > 1:
        # Example – averaging channels:
        data = data.mean(axis=1).astype(data.dtype)
    return rate, data

# Function to save a wav file (mono) using samples
def save_wav(file_path_v, samples_v, rate):
    write(file_path_v, rate, samples_v)

# Function to concatenate a list of numpy arrays of samples
def concatenate_audios(audios):
    return np.concatenate(audios)

# Function to split audio into smaller chunks
def split_audio(samples_v, max_samples):
    return [samples_v[i:i + max_samples] for i in range(0, len(samples_v), max_samples)]

# Load files and group them by sample rate
# Create a dictionary: key = sample rate, value = a dictionary where each class (inhale, exhale, silence)
# contains a list of numpy arrays with samples.
audio_files = {rate: {cls: [] for cls in folders} for rate in [44100, 48000]}
for cls, folder in folders.items():
    for file_name in os.listdir(folder):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder, file_name)
            rate, samples = load_wav(file_path)
            if rate in audio_files:
                # Split the samples into smaller fragments
                fragments = split_audio(samples, MAX_CLASS_LENGTH * rate)
                audio_files[rate][cls].extend(fragments)  # Add the fragments to the class list

seq_num = 0  # Sequence number for file naming

# Iterate over sample rates
for rate, classes in audio_files.items():
    total_samples_max = MAX_LENGTH * rate  # Maximum number of samples for a sequence
    while any(len(files) > 0 for files in classes.values()):
        combined_audio_list = []   # List of audio fragments (numpy arrays)
        labels = []                # List of labels: (class, start_sample, end_sample)
        total_length_samples = 0   # Total number of samples in the sequence

        # Create a sequence until reaching MAX_LENGTH (in samples)
        while total_length_samples < total_samples_max:
            available_classes = [cls for cls in classes if len(classes[cls]) > 0]
            if not available_classes:
                break

            cls_choice = random.choice(available_classes)  # Randomly choose a class
            recording_id = random.randint(0, len(classes[cls_choice]) - 1)  # Randomly choose an index
            samples = classes[cls_choice][recording_id]  # Get the fragment using the id
            audio_samples = len(samples)  # Number of samples in this recording

            # Add the fragment to the list
            fragment = samples
            combined_audio_list.append(fragment)
            labels.append((cls_choice, total_length_samples, total_length_samples + audio_samples))
            total_length_samples += audio_samples

            classes[cls_choice].pop(recording_id)  # Remove the fragment using its index (id)

            if total_length_samples >= total_samples_max:
                break

        # If no fragments were collected, stop
        if len(combined_audio_list) == 0:
            break

        # Concatenate fragments into a single sequence
        output_audio = concatenate_audios(combined_audio_list)
        output_file = os.path.join(sequence_folder, f'sequence_{seq_num}.wav')
        save_wav(output_file, output_audio, rate)

        # Save label information to a CSV file
        csv_file = os.path.join(sequence_folder, f'sequence_{seq_num}.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['class', 'start_sample', 'end_sample'])
            writer.writerows(labels)

        print(f'Sequence {seq_num} saved')
        seq_num += 1
