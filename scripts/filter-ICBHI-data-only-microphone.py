"""
    Script to filter data downloaded from https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
    Dataset consists of recordings recorded both with microphone and stethoscope
    Here we only want the microphone recordings.
"""
import os

# Path to the folder containing files
folder_path = "../ICBHI_final_database"  # Change to the appropriate path

# Keywords for files to be removed
unwanted_keywords = ["Litt3200", "LittC2SE", "Meditron"]

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    # Check if the file has a .wav or .txt extension
    if filename.endswith(".wav") or filename.endswith(".txt"):
        # Remove the file if it contains any unwanted keywords
        if any(keyword in filename for keyword in unwanted_keywords):
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

print("Operation completed.")
