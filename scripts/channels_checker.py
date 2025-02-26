import os
import wave

def check_wav_properties(wav_path):
    """Checks if the .wav file is mono or stereo and gets the sample rate."""
    with wave.open(wav_path, 'rb') as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        channel_type = 'mono' if channels == 1 else 'stereo'
        return channel_type, sample_rate

def check_folder(folder_path):
    """Checks all .wav files in the given folder."""
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            wav_path = os.path.join(folder_path, filename)
            channel_type, sample_rate = check_wav_properties(wav_path)
            print(f"{filename}: {channel_type}, {sample_rate} Hz")

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing .wav files: ")
    check_folder(folder_path)