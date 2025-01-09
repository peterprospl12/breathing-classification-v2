import os
import time
import librosa
from sklearn.model_selection import train_test_split

REFRESH_TIME = 0.25  # seconds

exhale_dir = 'small-data/exhale'
inhale_dir = 'small-data/inhale'
silence_dir = 'small-data/silence'

exhale_files = [os.path.join(exhale_dir, file) for file in os.listdir(exhale_dir)]
inhale_files = [os.path.join(inhale_dir, file) for file in os.listdir(inhale_dir)]
silence_files = [os.path.join(silence_dir, file) for file in os.listdir(silence_dir)]

train_data = []
files_list = [exhale_files, inhale_files, silence_files]
files_names = ['exhale', 'inhale', 'silence']

print("Loading data...")
start_time = time.time()

exhale_frames_size = 0
inhale_frames_size = 0
silence_frames_size = 0

for label, files in enumerate(files_list):
    for file in files:
        y, sr = librosa.load(file, mono=True)
        chunk_size = int(sr * REFRESH_TIME)
        for i in range(0, len(y), chunk_size):
            frame = y[i:i + chunk_size]
            if len(frame) == chunk_size:  # Ignore the last frame if it's shorter
                mfcc = librosa.feature.mfcc(y=frame, sr=sr)
                print(mfcc.shape)
                train_data.append((mfcc, label))

    if label == 0:
        exhale_frames_size = len(train_data)
        print("Exhale frames size: ", exhale_frames_size)
    elif label == 1:
        inhale_frames_size = len(train_data) - exhale_frames_size
        print("Inhale frames size: ", inhale_frames_size)
    else:
        silence_frames_size = len(train_data) - exhale_frames_size - inhale_frames_size
        print("Silence frames size: ", silence_frames_size)
print("Data loaded, time: ", time.time() - start_time)

