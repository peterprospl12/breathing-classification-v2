import os
import random
import csv
import numpy as np
from scipy.io.wavfile import read, write
from scipy.signal import resample

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
MAX_LENGTH = 30  # Full sequence: 30 seconds
MAX_CLASS_LENGTH = 3  # Single fragment: 3 seconds


# Function to load a wav file, convert to mono, and resample to 44.1 kHz
def load_and_preprocess_wav(file_path_v):
    rate, data = read(file_path_v)

    # Convert to mono if stereo
    if data.ndim > 1 and data.shape[1] > 1:
        data = data.mean(axis=1).astype(data.dtype)

    # Resample to 44.1 kHz if necessary
    if rate != 44100:
        num_samples = int(len(data) * 44100 / rate)
        data = resample(data, num_samples)

    return data


# Function to save a wav file (mono) using samples
def save_wav(file_path_v, samples_v):
    write(file_path_v, 44100, samples_v.astype(np.int16))


# Function to concatenate a list of numpy arrays of samples
def concatenate_audios(audios):
    return np.concatenate(audios)


# Function to split audio into smaller chunks
def split_audio(samples_v, max_samples):
    return [samples_v[i:i + max_samples] for i in range(0, len(samples_v), max_samples)]


# Load all files, preprocess them, and split into fragments
audio_fragments = {cls: [] for cls in folders}
for cls, folder in folders.items():
    for file_name in os.listdir(folder):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder, file_name)
            samples = load_and_preprocess_wav(file_path)
            fragments = split_audio(samples, MAX_CLASS_LENGTH * 44100)
            audio_fragments[cls].extend(fragments)

# Randomly shuffle the fragments
for cls in audio_fragments:
    random.shuffle(audio_fragments[cls])

seq_num = 0  # Sequence number for file naming

# Create sequences
while any(len(fragments) > 0 for fragments in audio_fragments.values()):
    combined_audio_list = []  # List of audio fragments (numpy arrays)
    labels = []  # List of labels: (class, start_sample, end_sample)
    total_length_samples = 0  # Total number of samples in the sequence

    # Create a sequence until reaching MAX_LENGTH (in samples)
    while total_length_samples < MAX_LENGTH * 44100:
        available_classes = [cls for cls in audio_fragments if len(audio_fragments[cls]) > 0]
        if not available_classes:
            break

        # Choose a class based on the number of available fragments
        weights = [len(audio_fragments[cls]) for cls in available_classes]
        cls_choice = random.choices(available_classes, weights=weights, k=1)[0]

        # Pop the last fragment from the list
        samples = audio_fragments[cls_choice].pop()
        audio_samples = len(samples)

        # Add the fragment to the sequence
        combined_audio_list.append(samples)
        labels.append((cls_choice, total_length_samples, total_length_samples + audio_samples))
        total_length_samples += audio_samples

        # If the sequence is too long, stop
        if total_length_samples >= MAX_LENGTH * 44100:
            break

    # If no fragments were collected, stop
    if len(combined_audio_list) == 0:
        break

    # Concatenate fragments into a single sequence
    output_audio = concatenate_audios(combined_audio_list)
    output_file = os.path.join(sequence_folder, f'sequence_{seq_num}.wav')
    save_wav(output_file, output_audio)

    # Save label information to a CSV file
    csv_file = os.path.join(sequence_folder, f'sequence_{seq_num}.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'start_sample', 'end_sample'])
        writer.writerows(labels)

    print(f'Sequence {seq_num} saved')
    seq_num += 1