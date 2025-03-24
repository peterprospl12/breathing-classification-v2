from pydub import AudioSegment
import csv
import os
import pandas as pd
import numpy as np
from tenso_model.tenso_model_interpreter import TFLiteModelWrapper  # Import wcześniej stworzonej klasy

# Ścieżki do plików
ROW_SOUND_DATA_PATH = "./data/row_sound/"
LABELLED_TENS_DATA_PATH = "./data/tenso_labeled_data/"
OUTPUT_DIR = "./data/labeled_sound/"
TENS_ROW_DATA_PATH = "./data/tenso_row_data"
TENS_CONVERTED_ROW_DATA_PATH = "./data/tenso_row_converted_data"
TENS_LABELLED_DATA_PATH = "./data/tenso_labeled_data"
MODEL_PATH = "./tenso_model/saved_models/GRUModel_tens.tflite"

WINDOW_SIZE = 6  # Wielkość okna dla normalizacji i średniej ruchomej


# Utwórz foldery dla etykiet, jeśli nie istnieją
for label in ['inhale', 'exhale', 'silence']:
    os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)

# Mapowanie stanu na etykiety
state_labels = {2: "inhale", 0: "exhale", 3: "silence", 1: "silence"}

# Funkcja do przetwarzania pliku WAV
def process_audio_file(wav_file_path, tensometer_data):
    audio = AudioSegment.from_wav(wav_file_path)
    current_state = None
    start_time = None
    state_buffer = []

    # Uzyskaj timestamp z nazwy pliku
    timestamp = wav_file_path.split('/')[-1].split('.')[0].split('_')[3] + '_' + wav_file_path.split('/')[-1].split('.')[0].split('_')[4]

    for i, (time_sec, state) in enumerate(tensometer_data):
        # Pomijamy pierwsze 25 wierszy
        if i < 25:
            continue

        state_buffer.append((time_sec, state))

        # Jeśli mamy już więcej niż 4 wiersze w buforze, sprawdzamy bezdech
        if len(state_buffer) > 4:
            if all(s == 3 or s == 1 for _, s in state_buffer[-4:]):
                state = state_buffer[-5][1] if len(state_buffer) > 4 else state
            state_buffer = state_buffer[-4:]

        # Jeśli stan się zmienia, wycinamy fragment
        if current_state != state:
            if current_state is not None:
                end_time = time_sec * 1000
                duration_ms = end_time - start_time
                if duration_ms >= 500:  # Sprawdzamy, czy fragment jest dłuższy niż 0,5 sekundy
                    fragment = audio[start_time:end_time]
                    label = state_labels.get(current_state, "unknown")
                    output_path = os.path.join(OUTPUT_DIR, label, f"{timestamp}_{start_time/1000:.2f}_{end_time/1000:.2f}.wav")
                    fragment.export(output_path, format="wav")
            current_state = state
            start_time = time_sec * 1000

    if current_state is not None:
        duration_ms = len(audio) - start_time
        if duration_ms >= 250:  # Sprawdzamy, czy końcowy fragment jest dłuższy niż 0,5 sekundy
            fragment = audio[start_time:]
            label = state_labels.get(current_state, "unknown")
            output_path = os.path.join(OUTPUT_DIR, label, f"{timestamp}_{start_time/1000:.2f}_end.wav")
            fragment.export(output_path, format="wav")

# Funkcja do wczytywania danych tensometrycznych
def load_tensometer_data(tensometer_data_path):
    tensometer_data = []
    with open(tensometer_data_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            time_sec = float(row[2])
            state = int(row[1])
            tensometer_data.append((time_sec, state))
    return tensometer_data

# Funkcja do dopasowywania plików WAV do plików CSV
def get_matching_tensometer_file(wav_filename):
    timestamp = wav_filename.split('_')[3] + '_' + wav_filename.split('_')[4].split('.')[0]
    matching_csv_filename = f"row_tenso_data_{timestamp}_tagged.txt"
    matching_csv_path = os.path.join(LABELLED_TENS_DATA_PATH, matching_csv_filename)
    if os.path.exists(matching_csv_path):
        return matching_csv_path
    else:
        print(f"Brak pasującego pliku tensometru dla: {wav_filename}")
        return None
def process_single_file(file_name):
    """
    Przetwarza pojedynczy plik .txt z TENS_ROW_DATA_PATH, konwertuje znaczniki czasu na sekundy
    i zapisuje jako plik .csv w TENS_CONVERTED_ROW_DATA_PATH.

    Args:
    - file_name (str): Nazwa wejściowego pliku .txt.
    """
    input_path = os.path.join(TENS_ROW_DATA_PATH, file_name)
    output_path = os.path.join(TENS_CONVERTED_ROW_DATA_PATH, os.path.splitext(file_name)[0] + ".csv")

    # Sprawdzenie, czy plik wejściowy istnieje
    if not os.path.isfile(input_path):
        print(f"Plik {input_path} nie istnieje.")
        return
    
    # Odczyt pliku .txt
    df = pd.read_csv(input_path, delimiter=",", header=None)

    # Konwersja znaczników czasu na datetime i obliczenie sekund od początku
    df[0] = pd.to_datetime(df[0])
    df[0] = (df[0] - df[0].min()).dt.total_seconds()

    # Nadanie nazw kolumnom
    df.columns = ["seconds", "data-raw"]

    # Tworzenie katalogu wyjściowego, jeśli nie istnieje
    os.makedirs(TENS_CONVERTED_ROW_DATA_PATH, exist_ok=True)

    # Zapis do pliku .csv
    df.to_csv(output_path, index=False)

    print(f"Przetworzono i zapisano: {output_path}")


def normalize_window(window):
    """Normalizuje okno danych do zakresu [-1, 1]."""
    min_val = min(window)
    max_val = max(window)
    range_val = max_val - min_val
    if range_val == 0:
        return [0 for _ in window]

    normalized = [(-1 + 2 * (x - min_val) / range_val) for x in window]
    return normalized


def normalize(numbers, normalization_range):
    """Normalizuje dane za pomocą okien o zadanym zakresie."""
    normalized_values = []

    for i in range(len(numbers)):
        window = numbers[max(0, i - normalization_range): i]
        try:
            normalized_window_values = normalize_window(window)
            normalized_values.append(normalized_window_values[-1])
        except ValueError:
            continue

    return normalized_values


def moving_average(data, window_size):
    """Oblicza średnią ruchomą."""
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def predict_tags(data: list[float], model_path: str, window_size: int) -> list[int]:
    """
    Przewiduje tagi na podstawie danych przy użyciu modelu TensorFlow Lite.
    
    Args:
    - data-raw (list[float]): Dane wejściowe.
    - model_path (str): Ścieżka do modelu TensorFlow Lite.
    - window_size (int): Rozmiar okna dla predykcji.

    Returns:
    - list[int]: Lista przewidzianych tagów.
    """
    model = TFLiteModelWrapper(model_path)

    tags = []
    data_to_predict = []
    for i in range(len(data) - 5):
        numbers = data[i : i + 5]
        numbers.extend([abs(max(data[i : i + 5]) - min(data[i : i + 5]))])
        data_to_predict.append(numbers)
    print(f"data-raw to pred{data_to_predict}")
    tags.append(model.predict(np.array(data_to_predict)))
    return tags[0]




def save_tagged_data(data, tags, time, filename):
    """
    Zapisuje dane z tagami do pliku w katalogu TENS_LABELLED_DATA_PATH.
    
    Args:
    - data-raw (list[float]): Dane wejściowe.
    - tags (list[int]): Przewidziane tagi.
    - time (list[float]): Czas w sekundach.
    - filename (str): Nazwa pliku wyjściowego.
    """
    os.makedirs(TENS_LABELLED_DATA_PATH, exist_ok=True)
    output_path = os.path.join(TENS_LABELLED_DATA_PATH, filename)
    with open(output_path, "w") as file:
        for i in range(len(data)):
            file.write(f"{data[i]},{tags[i]},{time[i]}\n")
    print(f"Zapisano dane z tagami do: {output_path}")

def load_raw_data(filename: str) -> tuple[list[float], list[float]]:
    numbers = []
    times = []
    # Geting data-raw from file
    with open(filename) as file:
        data = list(csv.reader(file))[1:]
        for data_line in data:
            numbers.append(float(data_line[1]))
            times.append(float(data_line[0]))
    return times, numbers


def process_directory():
    """
    Przetwarza wszystkie pliki w katalogu TENS_ROW_DATA_PATH i TENS_CONVERTED_ROW_DATA_PATH,
    oznaczając przetworzone pliki końcówką 'P', aby uniknąć ich ponownego przetwarzania.
    """
    # Pierwsza pętla: Przetwarzanie plików z TENS_ROW_DATA_PATH
    for file_name in os.listdir(TENS_ROW_DATA_PATH):
        if file_name.endswith("P.txt"):
            print(f"Plik {file_name} już został przetworzony. Pomijanie...")
            continue
        
        process_single_file(file_name)

        # Zmienianie nazwy przetworzonego pliku
        original_path = os.path.join(TENS_ROW_DATA_PATH, file_name)
        processed_path = os.path.join(TENS_ROW_DATA_PATH, file_name.replace(".txt", "P.txt"))
        os.rename(original_path, processed_path)
        print(f"Plik {file_name} oznaczono jako przetworzony: {processed_path}")

    # Druga pętla: Przetwarzanie plików z TENS_CONVERTED_ROW_DATA_PATH
    for file_name in os.listdir(TENS_CONVERTED_ROW_DATA_PATH):
        if file_name.endswith("P.csv"):
            print(f"Plik {file_name} już został przetworzony. Pomijanie...")
            continue

        input_path = os.path.join(TENS_CONVERTED_ROW_DATA_PATH, file_name)
        times, numbers = load_raw_data(input_path)
        print(f"Raw numbers: {numbers}")

        # Preprocessing danych
        numbers = moving_average(numbers, WINDOW_SIZE)
        numbers = normalize(numbers, 150)
        print(f"Normalized numbers: {numbers}")

        # Predykcja tagów
        mono_tags = predict_tags(numbers, MODEL_PATH, WINDOW_SIZE)

        # Zapisanie danych z tagami
        save_tagged_data(
            numbers[:-WINDOW_SIZE],
            mono_tags,
            times[:-WINDOW_SIZE],
            file_name[:-4] + "_tagged.txt",
        )

        # Zmienianie nazwy przetworzonego pliku
        processed_path = os.path.join(TENS_CONVERTED_ROW_DATA_PATH, file_name.replace(".csv", "P.csv"))
        os.rename(input_path, processed_path)
        print(f"Plik {file_name} oznaczono jako przetworzony: {processed_path}")



#process_directory()

# Przetwórz wszystkie pliki WAV w folderze
for filename in os.listdir(ROW_SOUND_DATA_PATH):
    if filename.endswith(".wav"):
        if filename.endswith("P.wav"):  # Pomijaj przetworzone pliki
            print(f"Plik {filename} już został przetworzony. Pomijanie...")
           # continue
        
        wav_file_path = os.path.join(ROW_SOUND_DATA_PATH, filename)
        tensometer_data_path = get_matching_tensometer_file(filename)
        
        if tensometer_data_path:
            tensometer_data = load_tensometer_data(tensometer_data_path)
            process_audio_file(wav_file_path, tensometer_data)
            
            # Oznacz plik jako przetworzony przez dodanie "P" do nazwy
            new_filename = filename.replace(".wav", "P.wav")
            new_wav_file_path = os.path.join(ROW_SOUND_DATA_PATH, new_filename)
            os.rename(wav_file_path, new_wav_file_path)
            print(f"Plik {filename} przetworzony i oznaczony jako {new_filename}.")
