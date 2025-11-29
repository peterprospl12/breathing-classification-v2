import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# --- IMPORTY PROJEKTOWE ---
# Zakładamy, że struktura katalogów jest zachowana
try:
    from breathing_model.model.transformer.utils import Config, BreathType
    from breathing_model.model.transformer.dataset import BreathDataset, collate_fn
    from breathing_model.model.transformer.model import BreathPhaseTransformerSeq
except ImportError as e:
    print("Błąd importu! Upewnij się, że uruchamiasz skrypt z głównego katalogu projektu lub ustaw PYTHONPATH.")
    print(f"Szczegóły: {e}")
    sys.exit(1)

# --- KONFIGURACJA ---
TEST_RAW_FOLDER = "../data/eval/raw"  # Ścieżka do folderu z plikami WAV (testowe)
TEST_LABEL_FOLDER = "../data/eval/label"  # Ścieżka do folderu z etykietami CSV (testowe)
CONFIG_PATH = "../model/transformer/config.yaml"
MODEL_PATH = "best_model_epoch_31.pth"  # Ścieżka do zapisanego modelu

# Stała ignorowania (taka sama jak w train.py)
IGNORE_INDEX = -100


def evaluate_test_set():
    # 1. Konfiguracja i Urządzenie
    print(">>> Ładowanie konfiguracji...")
    config = Config.from_yaml(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Dataset i DataLoader (Kluczowe: Używamy BreathDataset, tak jak w treningu)
    # Wyłączamy augmentację dla testów!
    print(">>> Przygotowanie danych testowych...")
    try:
        test_dataset = BreathDataset(
            data_dir=TEST_RAW_FOLDER,
            label_dir=TEST_LABEL_FOLDER,
            sample_rate=config.data.sample_rate,
            n_mels=config.data.n_mels,
            n_fft=config.data.n_fft,
            hop_length=config.data.hop_length,
            augment=False  # WAŻNE: Ewaluacja zawsze na czystych danych
        )
    except Exception as e:
        print(f"Błąd tworzenia Datasetu: {e}")
        return

    # Używamy batch_size z configu lub 1 dla dokładnej analizy
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=(device.type == 'cuda')
    )

    print(f"Liczba plików testowych: {len(test_dataset)}")

    # 3. Ładowanie Modelu
    print(f">>> Ładowanie modelu z {MODEL_PATH}...")
    model = BreathPhaseTransformerSeq(
        n_mels=config.model.n_mels,
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_layers=config.model.num_layers,
        num_classes=config.model.num_classes
    ).to(device)

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        # Obsługa zapisu słownikowego (z train.py) lub samego state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except FileNotFoundError:
        print("Nie znaleziono pliku modelu!")
        sys.exit(1)

    model.eval()

    # 4. Pętla Ewaluacyjna (Logika z run_validation_epoch)
    all_preds = []
    all_targets = []

    total_loss = 0.0
    total_valid_frames = 0
    loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    print(">>> Rozpoczynanie ewaluacji...")

    with torch.no_grad():
        for i, (spectrograms, labels, padding_mask) in enumerate(test_loader):
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            padding_mask = padding_mask.to(device)

            # Etykiety do loss (z ignorowaniem paddingu)
            labels_for_loss = labels.clone()
            labels_for_loss[padding_mask] = IGNORE_INDEX

            # Forward pass
            # WAŻNE: Podajemy maskę paddingu, model widzi CAŁĄ sekwencję (jak w treningu)
            logits = model(spectrograms, src_key_padding_mask=padding_mask)  # [B, T, Classes]

            # Obliczanie Loss (dla statystyki)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = labels_for_loss.view(-1)

            # Filtrowanie tylko poprawnych ramek do metryk (usuwamy padding)
            # maska ~padding_mask wskazuje na prawdziwe dane
            valid_indices = ~padding_mask.view(-1)

            # Predykcje (argmax)
            preds = torch.argmax(logits, dim=-1)  # [B, T]

            # Zbieranie wyników (spłaszczamy i bierzemy tylko ważne ramki)
            flat_preds = preds.view(-1)[valid_indices].cpu().numpy()
            flat_targets = labels.view(-1)[valid_indices].cpu().numpy()

            all_preds.extend(flat_preds)
            all_targets.extend(flat_targets)

            # Aktualizacja loss
            batch_loss = loss_function(logits_flat, targets_flat)
            if not torch.isnan(batch_loss):
                valid_count = valid_indices.sum().item()
                total_loss += batch_loss.item() * valid_count
                total_valid_frames += valid_count

            if (i + 1) % 5 == 0:
                print(f"Przetworzono batch {i + 1}/{len(test_loader)}")

    # 5. Raport Końcowy
    print("\n" + "=" * 60)
    print("WYNIKI EWALUACJI NA ZBIORZE TESTOWYM")
    print("=" * 60)

    if len(all_targets) > 0:
        avg_loss = total_loss / total_valid_frames if total_valid_frames > 0 else 0
        acc = accuracy_score(all_targets, all_preds)

        print(f"Liczba przeanalizowanych ramek: {len(all_targets)}")
        print(f"Średnia strata (Loss):        {avg_loss:.6f}")
        print(f"Globalna Dokładność (Accuracy): {acc:.4f} ({acc * 100:.2f}%)")
        print("-" * 60)

        target_names = [BreathType(i).get_label() for i in range(config.model.num_classes)]

        print("Szczegółowy raport klasyfikacji (frame-by-frame):")
        print(classification_report(all_targets, all_preds, target_names=target_names, digits=4))
    else:
        print("Brak danych do oceny.")


if __name__ == "__main__":
    evaluate_test_set()