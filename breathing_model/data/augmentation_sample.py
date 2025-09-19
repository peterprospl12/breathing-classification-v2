# augment_single_file.py
import os
import torch
import torchaudio
import numpy as np
import librosa

# --- USTAWIENIA ---
RAW_DIR = "./train/raw"
CSV_DIR = "./train/label"
OUTPUT_DIR = "./augmented"  # Zapisujemy nad oryginał (możesz zmienić na inny folder)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Wybierz indeks pliku do augmentacji
FILE_INDEX = 0  # ← ZMIEŃ TEN NUMER, jeśli chcesz inny plik

# Augmentacja: wybierz jedną z poniższych
AUG_TYPE = "volume_down"  # opcje: "noise", "pitch_up", "pitch_down", "speed_up", "speed_down", "volume_down"

# Parametry augmentacji
NOISE_FACTOR = 0.0001
PITCH_STEPS = 2
TIME_STRETCH_RATE = 0.8


# Wczytaj plik
files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.wav')])
if FILE_INDEX >= len(files):
    raise IndexError(f"Indeks {FILE_INDEX} przekracza liczbę dostępnych plików ({len(files)})")

selected_file = files[FILE_INDEX]
base_name = os.path.splitext(selected_file)[0]

wav_path = os.path.join(RAW_DIR, selected_file)
csv_path = os.path.join(CSV_DIR, f"{base_name}.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Brak pliku CSV: {csv_path}")

print(f"🔧 Przetwarzanie: {selected_file}")


# Funkcja ładowania
def load_wav(path):
    waveform, sr = torchaudio.load(path)
    if sr != 44100:
        resampler = torchaudio.transforms.Resample(sr, 44100)
        waveform = resampler(waveform)
        sr = 44100
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze().numpy(), sr


# Funkcje augmentacyjne
def add_noise(data, factor=0.015):
    noise = np.random.randn(len(data))
    return data + factor * noise

def pitch_shift(data, sr, n_steps=2):
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)

def time_stretch(data, rate=1.2):
    return librosa.effects.time_stretch(data, rate=rate)

def change_volume(data, factor=0.7):
    return data * factor


# Wczytaj audio
audio, sr = load_wav(wav_path)

# Zastosuj augmentację
if AUG_TYPE == "noise":
    augmented_audio = add_noise(audio, NOISE_FACTOR)
    suffix = "_noise"
elif AUG_TYPE == "speed_up":
    augmented_audio = time_stretch(audio, rate=1.2)
    suffix = "_speed_up"
elif AUG_TYPE == "speed_down":
    augmented_audio = time_stretch(audio, rate=TIME_STRETCH_RATE)
    suffix = "_slow_down"
elif AUG_TYPE == "volume_down":
    augmented_audio = change_volume(audio, factor=0.3)  # np. 50% głośności
    augmented_audio = np.clip(augmented_audio, -1.0, 1.0)
    suffix = "_quieter"
else:
    raise ValueError(f"Nieznana augmentacja: {AUG_TYPE}")

# Zapisz audio w formacie int16 (standard .wav), BEZ normalizacji
output_wav_path = os.path.join(OUTPUT_DIR, f"{base_name}{suffix}.wav")
scaled = np.int16(augmented_audio * 32767)
torchaudio.save(output_wav_path, torch.from_numpy(scaled).unsqueeze(0), sr, encoding='PCM_S', bits_per_sample=16)

# Skopiuj CSV z tą samą nazwą (tylko zmieniamy rozszerzenie .wav na nowe)
output_csv_path = os.path.join(OUTPUT_DIR, f"{base_name}{suffix}.csv")
import shutil
shutil.copy(csv_path, output_csv_path)

print(f"✅ Zapisano:")
print(f"   Audio: {output_wav_path}")
print(f"   Etykiety: {output_csv_path}")