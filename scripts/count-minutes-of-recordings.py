"""
Script to count the total number of minutes of recordings in the dataset and calculate the average duration of recordings.
"""

import os
import wave

# Define paths
folders = ['inhale', 'exhale', 'silence']
base_dir = '../data'

# Function to calculate total duration and average duration in minutes for a folder
def calculate_durations(folder):
    path = os.path.join(base_dir, folder)
    total_folder_duration = 0.0
    file_count = 0

    for file in os.listdir(path):
        if file.endswith('.wav'):
            file_path = os.path.join(path, file)
            with wave.open(file_path, 'r') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / float(rate)
                total_folder_duration += duration
                file_count += 1

    average_folder_duration = (total_folder_duration / file_count) if file_count > 0 else 0
    return total_folder_duration / 60, average_folder_duration  # Convert total duration to minutes

# Calculate and print total and average duration for each folder
for fold in folders:
    total_duration, average_duration = calculate_durations(fold)
    print(f"Total duration in {fold}: {total_duration:.2f} minutes")
    print(f"Average duration in {fold}: {average_duration:.2f} seconds")