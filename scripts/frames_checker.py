import os
import wave

# Path to the folder containing .wav files
folder_path = 'data-seq'

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.wav'):
        file_path = os.path.join(folder_path, filename)

        # Open the .wav file
        with wave.open(file_path, 'r') as wav_file:
            # Get the number of frames
            num_frames = wav_file.getnframes()
            print(f"{filename}: {num_frames} frames")