"""
    Script for separating the audio recordings into
    inhale, exhale, and silence segments.
    Each recording is corresponding for single inhale/exhale/silence.
"""

import os
from scipy.io import wavfile
import numpy as np

# Path to the folder containing audio files
folder_path = "../data-magisterka"  # Change this to the appropriate path if needed

# Create directories for inhale, exhale, and silence segments
os.makedirs(os.path.join(folder_path, "inhale"), exist_ok=True)
os.makedirs(os.path.join(folder_path, "exhale"), exist_ok=True)
os.makedirs(os.path.join(folder_path, "silence"), exist_ok=True)

# Get all .wav files in the directory
wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]

# Counters for naming output files
inhale_counter = 1
exhale_counter = 1
silence_counter = 1

# Minimum segment length in samples
min_segment_length = 20000

for wav_file in wav_files:
    base_name = wav_file[:-4]  # Extract filename without extension
    txt_file = base_name + ".txt"
    wav_path = os.path.join(folder_path, wav_file)
    txt_path = os.path.join(folder_path, txt_file)

    # Check if the corresponding .txt file exists
    if not os.path.exists(txt_path):
        print(f"Skipping {wav_file} - corresponding .txt file is missing.")
        continue

    # Load the WAV file
    sampling_rate, audio_data = wavfile.read(wav_path)

    # Convert to mono if the audio is stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1).astype(audio_data.dtype)

    # Read the .txt file with labeled segments
    with open(txt_path, "r") as file:
        lines = file.readlines()

    # Process inhale and exhale segments
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 3:
            print(f"Invalid line in {txt_file}: {line.strip()}")
            continue

        label, start, end = parts[0], int(parts[1]), int(parts[2])

        # Extract the segment based on sample indices
        segment = audio_data[start:end]

        # Save the segment to the corresponding folder if it meets the minimum length requirement
        if len(segment) >= min_segment_length:
            if label == "wdech":  # Inhale
                output_path = os.path.join(folder_path, "inhale", f"inhale{inhale_counter}.wav")
                wavfile.write(output_path, sampling_rate, segment)
                inhale_counter += 1
            elif label == "wydech":  # Exhale
                output_path = os.path.join(folder_path, "exhale", f"exhale{exhale_counter}.wav")
                wavfile.write(output_path, sampling_rate, segment)
                exhale_counter += 1

    # Process silence segments (regions not labeled in the .txt file)
    previous_end = 0
    for line in lines + ["end"]:
        if line == "end":  # Handle silence after the last labeled segment
            start = previous_end
            end = len(audio_data)
        else:
            _, start, _ = line.strip().split()
            start = int(start)

        # Extract and save silence if there is a valid range and it meets the minimum length requirement
        if previous_end < start:
            segment = audio_data[previous_end:start]
            if len(segment) >= min_segment_length:
                output_path = os.path.join(folder_path, "silence", f"silence{silence_counter}.wav")
                wavfile.write(output_path, sampling_rate, segment)
                silence_counter += 1

        # Update the previous segment's end position
        if line != "end":
            _, _, previous_end = line.strip().split()
            previous_end = int(previous_end)
        else:
            previous_end = len(audio_data)

print("Operation completed.")