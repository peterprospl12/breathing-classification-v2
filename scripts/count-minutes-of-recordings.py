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
    return total_folder_duration / 60, average_folder_duration, file_count  # Convert total duration to minutes

n_total = 0
# Calculate and print total and average duration for each folder
for fold in folders:
    total_duration, average_duration, n_files = calculate_durations(fold)
    n_total += n_files
    print(f"Total number of {fold} files: {n_files}")
    print(f"Total duration in {fold}: {total_duration:.2f} minutes")
    print(f"Average duration in {fold}: {average_duration:.2f} seconds")
    print()

print(f"Total number of files: {n_total}")

'''
Total number of inhale files: 2085
Total duration in inhale: 52.80 minutes
Average duration in inhale: 1.52 seconds

Total number of exhale files: 2240
Total duration in exhale: 67.30 minutes
Average duration in exhale: 1.80 seconds

Total number of silence files: 2641
Total duration in silence: 53.49 minutes
Average duration in silence: 1.22 seconds

Total number of files: 6966
'''