"""
    Script aims to filter only this vaw files that have corresponding
    txt file with class labels. It also deletes all png plots that were in data set
    link to data set: https://github.com/FakkuGemu/masters_thesis
"""

import os

# Path to the folder containing files
folder_path = "../data-master-thesis"  # Change this to the appropriate path if needed

# Get all files in the directory
files = os.listdir(folder_path)

# Separate files by extension
wav_files = {f for f in files if f.endswith(".wav")}
txt_files = {f for f in files if f.endswith(".txt")}
png_files = {f for f in files if f.endswith(".png")}

# Remove all .png files
for png_file in png_files:
    os.remove(os.path.join(folder_path, png_file))
    print(f"Deleted: {png_file}")

# Find matching pairs of .wav and .txt files
matching_files = {f[:-4] for f in wav_files}.intersection(f[:-4] for f in txt_files)

# Remove .wav and .txt files that do not have a matching pair
for wav_file in wav_files:
    if wav_file[:-4] not in matching_files:
        os.remove(os.path.join(folder_path, wav_file))
        print(f"Deleted: {wav_file}")

for txt_file in txt_files:
    if txt_file[:-4] not in matching_files:
        os.remove(os.path.join(folder_path, txt_file))
        print(f"Deleted: {txt_file}")

print("Operation completed.")