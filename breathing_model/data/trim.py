import os
import pandas as pd
from pydub import AudioSegment
import shutil

# Settings
RAW_DIR = './raw'
LABEL_DIR = './label'
OUTPUT_RAW_DIR = './raw'        # Can be changed to a new folder, e.g., './raw_trimmed'
OUTPUT_LABEL_DIR = './label'    # Can be changed, e.g., './label_trimmed'

TARGET_LENGTH = 441000  # target number of samples

# Upewnij się, że katalogi wyjściowe istnieją
os.makedirs(OUTPUT_RAW_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# Iteracja po plikach wav w katalogu raw
for wav_file in os.listdir(RAW_DIR):
    if not wav_file.lower().endswith('.wav'):
        continue

    base_name = os.path.splitext(wav_file)[0]
    wav_path = os.path.join(RAW_DIR, wav_file)
    csv_path = os.path.join(LABEL_DIR, base_name + '.csv')

    # Sprawdź, czy istnieje odpowiadający plik CSV
    if not os.path.exists(csv_path):
        print(f"Missing CSV file for {wav_file}, skipping.")
        continue

    # Wczytaj plik WAV
    try:
        audio = AudioSegment.from_wav(wav_path)
    except Exception as e:
        print(f"Error loading {wav_file}: {e}")
        continue

    # Number of samples (AudioSegment has frame_count())
    sample_count = len(audio.get_array_of_samples())
    channels = audio.channels
    sample_width = audio.sample_width
    frame_rate = audio.frame_rate

    print(f"Processing: {wav_file}, sample count: {sample_count}")

    if sample_count < TARGET_LENGTH:
        # Usuń zarówno wav, jak i csv
        os.remove(wav_path)
        os.remove(csv_path)
        print(f"  Removed {wav_file} and {base_name}.csv (too short)")
    else:
        # Trim audio to the first TARGET_LENGTH samples
        # Calculate time in milliseconds: (samples / frame_rate) * 1000
        target_frames = TARGET_LENGTH
        target_duration_ms = (target_frames / frame_rate) * 1000

        trimmed_audio = audio[:target_duration_ms]  # Trim to this duration

        # Zapisz przycięty plik WAV
        output_wav_path = os.path.join(OUTPUT_RAW_DIR, wav_file)
        trimmed_audio.export(output_wav_path, format='wav')
        print(f"  Saved trimmed file: {output_wav_path}")

        # CSV processing: trim or remove segments outside [0, TARGET_LENGTH)
        try:
            df = pd.read_csv(csv_path)
            required_columns = ['class', 'start_sample', 'end_sample']
            if not all(col in df.columns for col in required_columns):
                print(f"  Error: missing required columns in {csv_path}")
                continue

            # Filter segments that start before TARGET_LENGTH
            valid_segments = []
            for _, row in df.iterrows():
                start = int(row['start_sample'])
                end = int(row['end_sample'])
                cls = row['class']

                # If the segment starts after TARGET_LENGTH — skip it
                if start >= TARGET_LENGTH:
                    continue

                # Trim the end to TARGET_LENGTH
                if end > TARGET_LENGTH:
                    end = TARGET_LENGTH

                # If the segment has a positive length after trimming, add it
                if start < end:
                    valid_segments.append({'class': cls, 'start_sample': start, 'end_sample': end})

            # Zapisz nowy CSV
            if valid_segments:
                new_df = pd.DataFrame(valid_segments)
                output_csv_path = os.path.join(OUTPUT_LABEL_DIR, base_name + '.csv')
                new_df.to_csv(output_csv_path, index=False)
                print(f"  Saved updated CSV: {output_csv_path}")
            else:
                # If no segments exist, remove the CSV
                if os.path.exists(os.path.join(OUTPUT_LABEL_DIR, base_name + '.csv')):
                    os.remove(os.path.join(OUTPUT_LABEL_DIR, base_name + '.csv'))
                print(f"  Removed {base_name}.csv (no segments after trimming)")

        except Exception as e:
            print(f"  Error processing {csv_path}: {e}")