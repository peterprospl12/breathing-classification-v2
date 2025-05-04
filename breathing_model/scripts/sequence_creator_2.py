import os
import random
import csv
from collections import deque, defaultdict
import math
import statistics # Dodano do obliczeń średniej i odchylenia

from pydub import AudioSegment
# pylint: disable=R0914,R0912,R0915

NUM_SEQUENCES = 100
NUM_SEGMENTS = 6 # Zmniejszenie może być konieczne, jeśli segmenty są dłuższe

MIN_SILENCE = 300
MAX_SILENCE = 1500 # Ten MAX może teraz działać bardziej jako górny limit bezpieczeństwa
PHASES = ['inhale', 'exhale', 'silence']
PHASE_CODES = {'exhale': 0, 'inhale': 1, 'silence': 2}
CODE_PHASES = {v: k for k, v in PHASE_CODES.items()}

FINAL_DURATION_MS = 10000
TARGET_FRAME_RATE = 44100
FINAL_SAMPLES = int((FINAL_DURATION_MS / 1000.0) * TARGET_FRAME_RATE)

exhale_folder = '../data-raw/exhale/'
inhale_folder = '../data-raw/inhale/'
silence_folder = '../data-raw/silence/'

# --- Funkcje pomocnicze (bez zmian) ---
def load_recordings(folder, min_duration=0):
    # ... (bez zmian - skopiuj z poprzedniej wersji) ...
    """
    Ładuje nagrania audio z określonego folderu.
    Konwertuje każde nagranie na mono i ustawia częstotliwość próbkowania na 44.1 kHz.

    Args:
        folder: ścieżka do folderu z nagraniami.
        min_duration: minimalny czas trwania (w ms) nagrania, aby zostało załadowane.

    Returns:
        Lista obiektów AudioSegment.
    """
    recordings = []
    print(f"Ładowanie nagrań z: {folder}")
    if not os.path.isdir(folder):
        print(f"Ostrzeżenie: Folder nie istnieje: {folder}")
        return recordings
    for filename in os.listdir(folder):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder, filename)
            try:
                recording = AudioSegment.from_wav(file_path)
                # Konwertuj na mono
                recording = recording.set_channels(1)
                # Ustaw częstotliwość próbkowania na 44.1 kHz
                recording = recording.set_frame_rate(TARGET_FRAME_RATE)
                if len(recording) >= min_duration:
                    recordings.append(recording)
            except Exception as e:
                print(f"Błąd podczas ładowania {file_path}: {e}")
    print(f"Załadowano {len(recordings)} nagrań.")
    return recordings

def calculate_average_duration(recordings):
    """Oblicza średni czas trwania nagrań w liście (w ms)."""
    if not recordings:
        return 0
    durations = [len(rec) for rec in recordings]
    return statistics.mean(durations) if durations else 0

def choose_phase_with_balancing(available_phases, total_durations_ms, current_target_duration_ms):
    # ... (bez zmian - skopiuj z poprzedniej wersji) ...
    """
    Wybiera fazę z dostępnych, faworyzując te z mniejszym łącznym czasem trwania.

    Args:
        available_phases: Lista dostępnych faz (stringi: 'inhale', 'exhale', 'silence').
        total_durations_ms: Słownik przechowujący całkowity czas trwania (w ms) dla każdej fazy
                           wygenerowany do tej pory we wszystkich sekwencjach.
        current_target_duration_ms: Całkowity czas trwania wygenerowany do tej pory we wszystkich fazach.

    Returns:
        Wybrana faza (string).
    """
    if not available_phases:
        return None

    # Jeśli nie ma jeszcze wygenerowanych danych lub wszystkie dostępne fazy mają zerowy czas, wybierz losowo
    if current_target_duration_ms == 0 or all(total_durations_ms.get(p, 0) == 0 for p in available_phases):
        return random.choice(available_phases)

    weights = []
    target_proportion = 1.0 / len(PHASES) # Idealna proporcja to 1/3
    epsilon = 1e-6 # Mała wartość, aby uniknąć problemów z dzieleniem przez zero

    # Debug print
    # print("Balancing Info:")
    # print(f"  Total Duration So Far: {current_target_duration_ms:.2f} ms")
    # for p in PHASES:
    #     dur = total_durations_ms.get(p, 0)
    #     prop = dur / current_target_duration_ms if current_target_duration_ms > 0 else 0
    #     print(f"  Phase: {p}, Duration: {dur:.2f} ms, Proportion: {prop:.3f}")

    for phase in available_phases:
        current_duration = total_durations_ms.get(phase, 0)
        current_proportion = current_duration / (current_target_duration_ms + epsilon)

        # Wzmocniona waga - odwrotność kwadratu proporcji, aby mocniej faworyzować brakujące
        # Dodanie małej wartości do current_proportion zapobiega dzieleniu przez zero i daje szansę
        # nawet fazom, które chwilowo przekroczyły cel.
        weight = 1.0 / (current_proportion + 0.05)**2 # Użyjmy potęgi 2 dla silniejszego efektu

        # Alternatywa: Liniowy deficyt (może być mniej agresywny)
        # weight = max(0.01, target_proportion - current_proportion) + 0.1

        weights.append(weight)
        # print(f"    Available: {phase}, Proportion: {current_proportion:.3f}, Weight: {weight:.3f}")


    total_weight = sum(weights)
    if total_weight == 0 or math.isnan(total_weight) or math.isinf(total_weight):
         # print(f"    Fallback: Zero, NaN or Inf total weight ({total_weight}). Choosing uniformly.")
         return random.choice(available_phases)

    # Wybierz fazę na podstawie wag
    try:
        chosen_phase = random.choices(available_phases, weights=weights, k=1)[0]
        # print(f"    Chosen phase by weight: {chosen_phase}")
        return chosen_phase
    except ValueError as e:
        # Może się zdarzyć, jeśli wszystkie wagi są np. ujemne lub zerowe (choć logika wyżej powinna temu zapobiec)
        print(f"    Fallback: Error in random.choices ({e}). Choosing uniformly.")
        return random.choice(available_phases)


def adjust_labels_for_final_duration(labels):
    # ... (bez zmian - skopiuj z poprzedniej wersji) ...
    """
    Dostosowuje etykiety na podstawie końcowej liczby próbek sekwencji.
    Jeśli etykieta wykracza poza końcową liczbę próbek, jest przycinana.
    Etykiety zaczynające się na lub po końcowej liczbie próbek są odrzucane.

    Args:
        labels: lista krotek (phase_code, start_sample, end_sample).

    Returns:
        adjusted_labels: lista dostosowanych krotek etykiet.
    """
    adjusted_labels = []
    for phase_code, start_sample, end_sample in labels:
        if start_sample >= FINAL_SAMPLES:
            # Ta etykieta zaczyna się po końcowym czasie trwania, więc ją pomiń.
            continue
        # Przytnij koniec etykiety, jeśli wykracza poza FINAL_SAMPLES
        adjusted_end_sample = min(end_sample, FINAL_SAMPLES - 1)

        # Upewnij się, że start jest mniejszy lub równy końcowi po przycięciu
        if start_sample <= adjusted_end_sample:
             adjusted_labels.append((phase_code, start_sample, adjusted_end_sample))
    return adjusted_labels

def finalize_sequence(sequence, labels):
    # ... (bez zmian - skopiuj z poprzedniej wersji) ...
    """
    Dostosowuje końcową sekwencję, aby miała dokładnie 30 sekund długości.
    Jeśli sekwencja jest dłuższa niż 30 sekund, jest przycinana.
    Jeśli jest krótsza, dodawana jest cisza.
    Etykiety są odpowiednio aktualizowane.

    Args:
        sequence: AudioSegment reprezentujący sekwencję.
        labels: lista krotek (phase_code, start_sample, end_sample).

    Returns:
        final_sequence: AudioSegment o długości dokładnie 30 sekund.
        final_labels: Dostosowane etykiety odpowiadające końcowej sekwencji.
    """
    final_sequence = sequence
    padding_duration_ms = 0

    if len(sequence) > FINAL_DURATION_MS:
        # Przytnij sekwencję do 30 sekund
        final_sequence = sequence[:FINAL_DURATION_MS]
    elif len(sequence) < FINAL_DURATION_MS:
        # Dodaj ciszę, jeśli sekwencja jest krótsza niż 30 sekund
        padding_duration_ms = FINAL_DURATION_MS - len(sequence)
        pad = AudioSegment.silent(
            duration=padding_duration_ms, frame_rate=TARGET_FRAME_RATE)
        final_sequence = sequence + pad
    # else: # Jeśli jest dokładnie 30 sekund, nie rób nic

    # Dostosuj etykiety na podstawie końcowej liczby próbek
    final_labels = adjust_labels_for_final_duration(labels)

    # Jeśli dodano ciszę, dodaj etykietę dla tej ciszy na końcu
    if padding_duration_ms > 0:
        start_sample_padding = int((len(sequence) / 1000.0) * TARGET_FRAME_RATE) # Początek ciszy = koniec oryginalnej sekwencji
        end_sample_padding = FINAL_SAMPLES - 1 # Koniec ciszy = koniec finalnej sekwencji
        # Upewnij się, że dodajemy tylko jeśli jest miejsce i czas trwania > 0
        if start_sample_padding <= end_sample_padding:
             final_labels.append((PHASE_CODES['silence'], start_sample_padding, end_sample_padding))

    # Posortuj etykiety na wypadek, gdyby dodana cisza zaburzyła kolejność
    final_labels.sort(key=lambda x: x[1])

    return final_sequence, final_labels


def save_sequence_and_labels(sequence, labels, sequence_id, output_folder="data-train"):
    # ... (bez zmian - skopiuj z poprzedniej wersji) ...
    """
    Zapisuje sekwencję audio i odpowiadające jej etykiety w określonym folderze.

    Args:
        sequence: AudioSegment do zapisania.
        labels: Lista krotek etykiet.
        sequence_id: Identyfikator sekwencji (używany w nazwach plików).
        output_folder: Folder docelowy.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Utworzono folder wyjściowy: {output_folder}")

    # Zapisz sekwencję audio
    audio_path = os.path.join(output_folder, f"ours{sequence_id}.wav")
    try:
        sequence.export(audio_path, format="wav")
    except Exception as e:
        print(f"Błąd podczas zapisywania pliku audio {audio_path}: {e}")
        return False # Sygnalizuj błąd

    # Zapisz etykiety do pliku CSV
    csv_path = os.path.join(output_folder, f"ours{sequence_id}.csv")
    try:
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["phase_code", "start_sample", "end_sample"])
            # Sortowanie etykiet po czasie rozpoczęcia dla pewności (już zrobione w finalize, ale dla pewności)
            labels.sort(key=lambda x: x[1])
            for phase_code, start_sample, end_sample in labels:
                 # Upewnijmy się, że start <= end
                 if start_sample <= end_sample:
                     writer.writerow([phase_code, start_sample, end_sample])
                 else:
                     print(f"Ostrzeżenie w sekwencji {sequence_id}: Pusta lub odwrócona etykieta ({phase_code}, {start_sample}, {end_sample}) po finalizacji, pomijanie zapisu.")

    except Exception as e:
        print(f"Błąd podczas zapisywania pliku CSV {csv_path}: {e}")
        return False # Sygnalizuj błąd

    return True # Sukces

def calculate_duration_ms_from_samples(start_sample, end_sample, frame_rate):
    # ... (bez zmian - skopiuj z poprzedniej wersji) ...
    """Oblicza czas trwania w milisekundach na podstawie próbek."""
    num_samples = end_sample - start_sample + 1
    if num_samples < 0:
        return 0
    duration_sec = num_samples / frame_rate
    return duration_sec * 1000.0
# --- Koniec funkcji pomocniczych ---


# --- Główna logika ---

# Załaduj nagrania
print("--- Ładowanie nagrań ---")
exhale_recordings = load_recordings(exhale_folder)
inhale_recordings = load_recordings(inhale_folder)
silence_recordings = load_recordings(silence_folder, MIN_SILENCE) # MIN_SILENCE jako min dla ciszy
print("--- Ładowanie zakończone ---")

# Sprawdzenie, czy mamy jakiekolwiek nagrania
if not exhale_recordings and not inhale_recordings and not silence_recordings:
    raise ValueError("Brak nagrań do przetworzenia. Sprawdź ścieżki folderów.")
if not silence_recordings:
     print("Ostrzeżenie: Brak nagrań ciszy! Faza ciszy nie będzie generowana.")
     PHASES.remove('silence')
     del PHASE_CODES['silence']
     # Trzeba by dostosować logikę balansowania, jeśli brakuje fazy, ale na razie załóżmy, że są wszystkie


# *** NOWOŚĆ: Oblicz średnią długość segmentów oddechowych ***
avg_inhale_ms = calculate_average_duration(inhale_recordings)
avg_exhale_ms = calculate_average_duration(exhale_recordings)

# Uśredniona długość jako punkt odniesienia dla ciszy
if avg_inhale_ms > 0 and avg_exhale_ms > 0:
     avg_breath_ms = (avg_inhale_ms + avg_exhale_ms) / 2.0
elif avg_inhale_ms > 0:
     avg_breath_ms = avg_inhale_ms
elif avg_exhale_ms > 0:
     avg_breath_ms = avg_exhale_ms
else:
     avg_breath_ms = 1000.0 # Domyślna wartość, jeśli brak nagrań oddechowych (mało prawdopodobne)
     print("Ostrzeżenie: Brak nagrań wdechu/wydechu do obliczenia średniej długości. Używanie domyślnej wartości 1000ms.")

print(f"\nŚrednia długość nagrania Inhale: {avg_inhale_ms:.2f} ms")
print(f"Średnia długość nagrania Exhale: {avg_exhale_ms:.2f} ms")
print(f"==> Uśredniona długość segmentu oddechowego (używana jako cel dla ciszy): {avg_breath_ms:.2f} ms\n")


# *** ZMIANA: Modyfikacja `create_sequence_with_rules` ***
def create_sequence_with_rules(num_segments, total_durations_ms, current_target_duration_ms, target_silence_duration_ms):
    """
    Tworzy sekwencję audio, stosując reguły, równoważenie
    i *dostosowaną długość ciszy*.

    Args:
        num_segments: liczba segmentów (faz) do wygenerowania.
        total_durations_ms: Globalny słownik z całkowitym czasem trwania każdej fazy.
        current_target_duration_ms: Globalny całkowity czas trwania.
        target_silence_duration_ms: Docelowa średnia długość segmentu ciszy (w ms).

    Returns:
        sequence: AudioSegment.
        labels: Lista krotek (phase_code, start_sample, end_sample).
    """
    sequence = AudioSegment.silent(duration=0, frame_rate=TARGET_FRAME_RATE)
    labels = []
    last_phases = deque(maxlen=3)
    consecutive_count = 0
    prev_phase = None
    current_sample = 0

    available_recordings = {
        'inhale': inhale_recordings,
        'exhale': exhale_recordings,
        'silence': silence_recordings
    }
    active_phases = [p for p, recs in available_recordings.items() if recs]
    if not active_phases:
         print("Błąd krytyczny: Brak jakichkolwiek aktywnych faz (nagrań).")
         return sequence, labels # Zwróć puste

    for i in range(num_segments):
        # --- Logika wyboru fazy (jak poprzednio, ale z uwzględnieniem active_phases) ---
        missing_phases = []
        if len(last_phases) >= 3:
            missing_phases = [p for p in active_phases if p not in last_phases] # Tylko aktywne fazy

        possible_phases = list(active_phases) # Start with all active phases

        # Wymuś brakującą fazę
        forced_phases = [p for p in missing_phases if p in possible_phases]

        # Ograniczenie: max 2 takie same pod rząd
        if consecutive_count >= 2 and prev_phase in possible_phases:
             possible_phases.remove(prev_phase)

        # Wybierz fazę
        phase = None
        if forced_phases:
             # Jeśli jest wymuszona i możliwa, wybierz ją losowo spośród wymuszonych
             phase = random.choice(forced_phases)
        elif possible_phases:
             # Wybierz z możliwych za pomocą balansowania
             phase = choose_phase_with_balancing(possible_phases, total_durations_ms, current_target_duration_ms)
        else:
             # Jeśli nic nie jest możliwe (np. przez reguły), spróbuj wybrać cokolwiek aktywnego
             if active_phases:
                  phase = random.choice(active_phases) # Wybierz cokolwiek, łamiąc reguły jeśli trzeba? Albo zakończ? Na razie wybierzmy.
                  # print(f"  Ostrzeżenie: Reguły uniemożliwiły wybór. Wybrano losowo: {phase}")
             else:
                 print("  Błąd: Brak możliwych faz do wyboru.")
                 continue # Pominięcie tego segmentu

        if not phase:
            print("  Ostrzeżenie: Nie udało się wybrać fazy, pomijanie segmentu.")
            continue

        # --- Wybór klipu dla fazy ---
        recordings_for_phase = available_recordings[phase]
        if not recordings_for_phase:
             # To nie powinno się zdarzyć, jeśli active_phases jest poprawne
             print(f"  Błąd krytyczny: Brak nagrań dla wybranej aktywnej fazy {phase}!")
             continue

        clip = None
        clip_duration_ms = 0

        if phase == 'inhale' or phase == 'exhale':
            clip = random.choice(recordings_for_phase)
            clip_duration_ms = len(clip)
        elif phase == 'silence':
            base_clip = random.choice(recordings_for_phase)
            base_clip_len = len(base_clip)

            # *** ZMIANA: Celuj w średnią długość oddechu dla ciszy ***
            # Dodaj losowość wokół celu (np. +/- 20%)
            target_duration = random.uniform(0.8 * target_silence_duration_ms,
                                             1.2 * target_silence_duration_ms)

            # Ogranicz przez MIN, MAX i długość bazowego klipu
            effective_max_silence = min(MAX_SILENCE, base_clip_len)
            # Najpierw ogranicz target przez MAX i długość klipu
            target_duration = min(target_duration, effective_max_silence)
            # Następnie upewnij się, że jest co najmniej MIN_SILENCE
            clip_duration_ms = max(MIN_SILENCE, target_duration)

            # Upewnij się, że nie przekraczamy długości base_clip po zaokrągleniach/ograniczeniach
            clip_duration_ms = int(min(clip_duration_ms, base_clip_len))

            if clip_duration_ms > 0:
                clip = base_clip[:clip_duration_ms]
            else:
                # Sytuacja awaryjna: jeśli obliczona długość jest <= 0
                print(f"  Ostrzeżenie: Obliczony czas trwania ciszy <= 0 ({clip_duration_ms}ms).")
                # Spróbuj użyć MIN_SILENCE, jeśli klip jest wystarczająco długi
                clip_duration_ms = int(min(max(MIN_SILENCE, 1), base_clip_len))
                if clip_duration_ms > 0:
                     clip = base_clip[:clip_duration_ms]
                else:
                     print(f"  Błąd: Nie można utworzyć segmentu ciszy z base_clip o długości {base_clip_len}ms.")
                     clip = None # Nie udało się utworzyć klipu

        if clip is None or len(clip) == 0:
            # print(f"  Ostrzeżenie: Wygenerowano pusty klip dla fazy {phase}, pomijanie segmentu {i+1}.")
            # Zamiast pomijać, co może prowadzić do krótszych sekwencji, spróbujmy jeszcze raz?
            # Na razie pomijamy, ale można by tu dodać logikę ponawiania.
            continue

        # --- Reszta logiki segmentu (jak poprzednio) ---
        sequence += clip
        num_samples = int((len(clip) / 1000.0) * clip.frame_rate)
        phase_code = PHASE_CODES[phase]
        start_sample = current_sample
        end_sample = current_sample + num_samples - 1
        labels.append((phase_code, start_sample, end_sample))
        current_sample += num_samples

        if phase == prev_phase:
            consecutive_count += 1
        else:
            consecutive_count = 1
            prev_phase = phase
        last_phases.append(phase)

    return sequence, labels


# *** ZMIANA: Główna pętla w `main` ***
def main():
    total_durations_ms = defaultdict(float)
    sequences_generated = 0
    output_dir = "data-raw-val" # Nowy folder wyjściowy

    # Pobierz docelową długość ciszy (obliczoną globalnie)
    target_silence_duration = avg_breath_ms
    if target_silence_duration <= 0:
         print("Błąd krytyczny: Docelowy czas ciszy jest zerowy lub ujemny.")
         return

    for i in range(NUM_SEQUENCES):
        print(f"\nGenerowanie sekwencji {i+1}/{NUM_SEQUENCES}...")

        current_total_duration = sum(total_durations_ms.values())

        # *** ZMIANA: Przekaż target_silence_duration ***
        sequence, labels = create_sequence_with_rules(
            NUM_SEGMENTS,
            total_durations_ms,
            current_total_duration,
            target_silence_duration # Przekazanie obliczonej średniej
        )

        if len(sequence) == 0 or not labels:
             print(f"Ostrzeżenie: Sekwencja {i} jest pusta po fazie tworzenia. Pomijanie.")
             continue

        final_sequence, final_labels = finalize_sequence(sequence, labels)

        if len(final_sequence) != FINAL_DURATION_MS:
             print(f"Błąd krytyczny: Finalizowana sekwencja {i} ma długość {len(final_sequence)}ms zamiast {FINAL_DURATION_MS}ms. Pomijanie.")
             continue

        if save_sequence_and_labels(final_sequence, final_labels, i, output_folder=output_dir):
            sequences_generated += 1
            # Aktualizacja całkowitych czasów trwania (jak poprzednio)
            current_seq_durations = defaultdict(float) # Śledzenie w tej sekwencji dla logowania
            for phase_code, start_sample, end_sample in final_labels:
                 duration_ms = calculate_duration_ms_from_samples(start_sample, end_sample, TARGET_FRAME_RATE)
                 phase_name = CODE_PHASES.get(phase_code)
                 if phase_name:
                     total_durations_ms[phase_name] += duration_ms
                     current_seq_durations[phase_name] += duration_ms
                 else:
                      print(f"Ostrzeżenie: Nieznany kod fazy {phase_code} w sekwencji {i}.")

            # Wyświetl podsumowanie dla tej sekwencji i postęp globalny
            print(f"  Sekwencja {i+1} zapisana. Czas trwania faz w tej sekwencji:")
            for phase in PHASES: # Użyj PHASES dla spójnej kolejności
                 print(f"    {phase:<8}: {current_seq_durations.get(phase, 0):>8.2f} ms")

            if (i + 1) % 10 == 0 or (i + 1) == NUM_SEQUENCES : # Pokaż co 10 i na końcu
                 print(f"  --- Postęp globalny po {i+1} sekwencjach ---")
                 current_total = sum(total_durations_ms.values())
                 if current_total > 0:
                      for phase in PHASES:
                            dur = total_durations_ms.get(phase, 0)
                            print(f"    {phase:<8}: {dur:>10.2f} ms ({dur/current_total*100:>5.1f}%)")
                 print(f"  ---------------------------------------")


    print("\n--- Generowanie zakończone ---")
    print(f"Pomyślnie wygenerowano i zapisano {sequences_generated} z {NUM_SEQUENCES} sekwencji w folderze '{output_dir}'.")

    # --- Końcowe podsumowanie ---
    print("\n--- Końcowe podsumowanie całkowitego czasu trwania faz ---")
    grand_total_duration_ms = sum(total_durations_ms.values())

    if grand_total_duration_ms == 0:
        print("Nie wygenerowano żadnych danych (całkowity czas trwania = 0).")
    else:
        print(f"Całkowity czas trwania wszystkich sekwencji: {grand_total_duration_ms / 1000.0:.2f} sekund ({grand_total_duration_ms:.2f} ms)")
        for phase in PHASES: # Użyj PHASES dla spójnej kolejności
             dur = total_durations_ms.get(phase, 0)
             print(f"  - {phase:<8}: {dur:>10.2f} ms ({dur / grand_total_duration_ms * 100:.2f} %)")

        expected_total_duration = sequences_generated * FINAL_DURATION_MS
        print(f"\nOczekiwany całkowity czas trwania ({sequences_generated} sekwencji * {FINAL_DURATION_MS} ms): {expected_total_duration:.2f} ms")
        discrepancy = abs(grand_total_duration_ms - expected_total_duration)
        print(f"Różnica między obliczonym a oczekiwanym czasem: {discrepancy:.2f} ms")
        if discrepancy > sequences_generated * 2: # Zwiększona tolerancja na błędy zaokrągleń
            print("Ostrzeżenie: Znaczna różnica między obliczonym a oczekiwanym całkowitym czasem trwania!")

if __name__ == "__main__":
    required_folders = [exhale_folder, inhale_folder, silence_folder]
    abort = False
    for folder in required_folders:
        if not os.path.isdir(folder):
            print(f"BŁĄD: Folder z danymi '{folder}' nie istnieje!")
            abort = True
    if abort:
        print("Przerwano działanie skryptu z powodu brakujących folderów z danymi.")
    else:
        # Uruchomienie głównej funkcji
        main()