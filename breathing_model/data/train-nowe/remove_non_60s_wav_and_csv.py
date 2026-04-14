from __future__ import annotations

import wave
from pathlib import Path

# Stala konfiguracja (bez CLI)
RAW_DIR = Path(r"C:\Users\sanko\Desktop\breathing-classification-v2\breathing_model\data\train-nowe\raw")
LABEL_DIR = Path(r"C:\Users\sanko\Desktop\breathing-classification-v2\breathing_model\data\train-nowe\label")

TARGET_SECONDS = 60.0
TOLERANCE_SECONDS = 0.05

# Bezpiecznik: False = tylko podglad, True = faktyczne usuwanie.
APPLY_CHANGES = True


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        sample_rate = wf.getframerate()
        if sample_rate <= 0:
            raise ValueError(f"Niepoprawny sample rate: {path}")
        return frames / float(sample_rate)


def should_keep_wav(duration: float) -> bool:
    return abs(duration - TARGET_SECONDS) <= TOLERANCE_SECONDS


def delete_file(path: Path) -> bool:
    if not path.exists():
        return False
    path.unlink()
    return True


def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Brak katalogu RAW: {RAW_DIR}")
    if not LABEL_DIR.exists():
        raise FileNotFoundError(f"Brak katalogu LABEL: {LABEL_DIR}")

    scanned_wav = 0
    kept_wav = 0
    removed_wav = 0
    removed_csv = 0
    missing_csv = 0
    errors = 0

    wav_files = sorted(RAW_DIR.glob("*.wav"))
    removal_candidates: list[tuple[Path, Path, float, bool]] = []

    for wav_path in wav_files:
        scanned_wav += 1

        try:
            duration = wav_duration_seconds(wav_path)
        except Exception as e:
            errors += 1
            print(f"[WARN] Nie udalo sie odczytac {wav_path.name}: {e}")
            continue

        if should_keep_wav(duration):
            kept_wav += 1
            continue

        csv_path = LABEL_DIR / f"{wav_path.stem}.csv"
        removal_candidates.append((wav_path, csv_path, duration, csv_path.exists()))

    print("\n=== PLIKI DO USUNIECIA (WAV != ~60s) ===")
    if not removal_candidates:
        print("Brak plikow do usuniecia.")
    else:
        for wav_path, csv_path, duration, csv_exists in removal_candidates:
            csv_info = csv_path.name if csv_exists else "BRAK CSV"
            print(f"  {wav_path.name}: {duration:.2f}s -> CSV: {csv_info}")

    for wav_path, csv_path, duration, csv_exists in removal_candidates:
        print(f"REMOVE WAV ({duration:.2f}s): {wav_path.name}")
        if APPLY_CHANGES:
            if delete_file(wav_path):
                removed_wav += 1
        else:
            removed_wav += 1

        if csv_exists:
            print(f"REMOVE CSV: {csv_path.name}")
            if APPLY_CHANGES:
                if delete_file(csv_path):
                    removed_csv += 1
            else:
                removed_csv += 1
        else:
            missing_csv += 1
            print(f"[INFO] Brak CSV dla: {wav_path.name}")

    print("\n=== PODSUMOWANIE ===")
    print(f"APPLY_CHANGES: {APPLY_CHANGES}")
    print(f"Skanowane WAV: {scanned_wav}")
    print(f"Zostawione ~60s: {kept_wav}")
    print(f"Do usuniecia WAV: {removed_wav}")
    print(f"Do usuniecia CSV: {removed_csv}")
    print(f"Brakujace CSV: {missing_csv}")
    print(f"Bledy odczytu WAV: {errors}")


if __name__ == "__main__":
    main()

