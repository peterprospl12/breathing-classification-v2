import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Subset
from itertools import cycle
from scipy.signal import medfilt  # Import do filtru medianowego

# Add the model directory to the Python path to import necessary modules
# Adjust the path based on the script's location relative to the model directory
model_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'model', 'transformer_model'))
if model_dir not in sys.path:
    sys.path.append(model_dir)

# Import classes from transformer_model.py
try:
    from transformer_model import BreathPhaseTransformerSeq, BreathSeqDataset
except ImportError as e:
    print(f"Error importing from transformer_model: {e}")
    print(
        f"Ensure the path '{model_dir}' is correct and contains transformer_model.py")
    sys.exit(1)

# --- Configuration ---
DATA_DIR = "../../data-sequences"  # Path relative to this script
# Path relative to this script
MODEL_PATH = "../model/transformer_model/best_breath_seq_transformer_model_CURR_BEST.pth"
BATCH_SIZE = 4  # Should match training, but can be adjusted based on memory
N_MELS = 40
NUM_CLASSES = 3
# Zaktualizowane parametry modelu, aby pasowały do trenowanego modelu
D_MODEL = 128
NHEAD = 8  # Zmieniono z 4 na 8
NUM_TRANSFORMER_LAYERS = 4  # Zmieniono z 2 na 4
SAMPLE_RATE = 44100
N_FFT = 1024
HOP_LENGTH = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = {0: "exhale", 1: "inhale", 2: "silence"}  # Map codes to names
# Rozmiar okna filtru medianowego (taki sam jak w przykładzie)
MEDIAN_FILTER_SIZE = 5

# --- Helper Functions ---


def get_test_loader(data_dir, batch_size, seed=42):
    """Creates the test DataLoader using a consistent split logic."""
    # Używamy tej samej logiki podziału jak w evaluate_model.py (70% train, 15% val, 15% test)
    # Zakładamy, że model był trenowany na pierwszych 85% (train+val)
    full_dataset = BreathSeqDataset(
        data_dir, sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    num_samples = len(full_dataset)
    indices = list(range(num_samples))

    # Use a fixed seed for reproducibility of the split
    np.random.seed(seed)
    np.random.shuffle(indices)

    # Split: 70% train, 15% validation, 15% test
    train_split = int(0.7 * num_samples)
    val_split = int(0.85 * num_samples)  # Koniec walidacji = początek testu
    test_indices = indices[val_split:]

    if not test_indices:
        print("Warning: No samples allocated for the test set. Check dataset size and split percentages.")
        return None, None

    print(f"Using {len(test_indices)} samples for testing.")
    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader, test_indices


def evaluate(model, test_loader, criterion, device):
    """Evaluates the model on the test set and returns metrics."""
    model.eval()
    all_preds_flat = []
    all_labels_flat = []
    all_probs_flat = []
    all_preds_seq = []  # Przechowuj predykcje dla każdej sekwencji osobno
    all_labels_seq = []  # Przechowuj etykiety dla każdej sekwencji osobno
    test_loss = 0.0
    total_frames = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)  # shape: (B, 1, n_mels, time_steps)
            labels = labels.to(device)  # shape: (B, time_steps)

            outputs = model(inputs)    # shape: (B, time_steps, num_classes)

            # Calculate loss
            loss = criterion(
                outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            # Accumulate loss weighted by batch size
            test_loss += loss.item() * inputs.size(0)

            # Get predictions and probabilities
            # (B, time_steps, num_classes)
            probabilities = torch.softmax(outputs, dim=-1)
            _, predicted = torch.max(outputs, dim=-1)      # (B, time_steps)

            # Store frame-level predictions and labels for metrics calculation (flattened)
            all_preds_flat.append(predicted.view(-1).cpu().numpy())
            all_labels_flat.append(labels.view(-1).cpu().numpy())
            # (B*time_steps, num_classes)
            all_probs_flat.append(
                probabilities.view(-1, NUM_CLASSES).cpu().numpy())

            # Store predictions and labels per sequence for smoothing
            for i in range(predicted.shape[0]):
                all_preds_seq.append(predicted[i].cpu().numpy())
                all_labels_seq.append(labels[i].cpu().numpy())

            total_frames += labels.numel()

    # Concatenate flattened results from all batches
    all_preds_flat = np.concatenate(all_preds_flat)
    all_labels_flat = np.concatenate(all_labels_flat)
    all_probs_flat = np.concatenate(all_probs_flat, axis=0)

    # Calculate average test loss
    avg_test_loss = test_loss / \
        len(test_loader.dataset) if test_loader.dataset else 0

    # Zwracamy zarówno spłaszczone wyniki, jak i sekwencyjne
    return all_labels_flat, all_preds_flat, all_probs_flat, avg_test_loss, all_labels_seq, all_preds_seq


def plot_confusion_matrix(y_true, y_pred, classes, filename="confusion_matrix.png", title_suffix=""):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes.values(), yticklabels=classes.values())
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Frame Level){title_suffix}')
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix saved to {filename}")


def plot_roc_curve(y_true, y_probs, n_classes, classes, filename="roc_curve.png"):
    """Plots and saves the ROC curve for each class (One-vs-Rest)."""
    # Binarize the labels for OvR
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        # Obsługa przypadku, gdy klasa nie występuje w y_true
        if np.sum(y_true_bin[:, i]) == 0:
            print(
                f"Warning: Class {classes[i]} not present in true labels. Skipping ROC calculation for this class.")
            fpr[i], tpr[i], roc_auc[i] = np.array(
                [0, 1]), np.array([0, 1]), 0.0  # Placeholder
            continue
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue',
                   'green', 'red', 'purple'])
    plotted_classes = 0
    for i, color in zip(range(n_classes), colors):
        if i in fpr:  # Sprawdź czy obliczono dla tej klasy
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:0.2f})')
            plotted_classes += 1

    if plotted_classes > 0:  # Tylko jeśli cokolwiek narysowano
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) - One-vs-Rest')
        plt.legend(loc="lower right")
        plt.savefig(filename)
        plt.close()
        print(f"ROC curve saved to {filename}")
    else:
        print("Could not plot ROC curve as no classes were present in true labels.")


# --- Main Execution ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # 1. Load Test Data
    print("Loading test data...")
    test_loader, test_indices = get_test_loader(DATA_DIR, BATCH_SIZE)
    if test_loader is None:
        print("Could not create test loader. Exiting.")
        sys.exit(1)

    # 2. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please ensure the model has been trained and saved correctly.")
        sys.exit(1)

    # Użyj zaktualizowanych parametrów
    model = BreathPhaseTransformerSeq(
        n_mels=N_MELS,
        num_classes=NUM_CLASSES,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_transformer_layers=NUM_TRANSFORMER_LAYERS
    )
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        sys.exit(1)

    # 3. Define Loss Function (needed for evaluation loss calculation)
    criterion = torch.nn.CrossEntropyLoss()

    # 4. Evaluate Model
    print("Evaluating model on the test set...")
    true_labels_flat, pred_labels_flat, pred_probs_flat, avg_loss, true_labels_seq, pred_labels_seq = evaluate(
        model, test_loader, criterion, DEVICE)

    if len(true_labels_flat) == 0:
        print(
            "Evaluation resulted in no labels or predictions. Check data loading and model.")
        sys.exit(1)

    # 5. Apply Smoothing (Median Filter)
    print(f"\nApplying median filter (kernel size: {MEDIAN_FILTER_SIZE})...")
    smoothed_preds_seq = []
    for seq in pred_labels_seq:
        # Upewnij się, że rozmiar okna nie jest większy niż długość sekwencji
        k_size = min(MEDIAN_FILTER_SIZE, len(seq))
        if k_size % 2 == 0:  # Filtr medianowy wymaga nieparzystego rozmiaru okna
            k_size -= 1
        if k_size < 1:
            k_size = 1  # Minimalny rozmiar okna to 1

        smoothed_seq = medfilt(seq, kernel_size=k_size).astype(np.int64)
        smoothed_preds_seq.append(smoothed_seq)

    # Spłaszcz wygładzone predykcje do porównania z true_labels_flat
    smoothed_preds_flat = np.concatenate(smoothed_preds_seq)

    # 6. Calculate and Print Metrics (Original Predictions)
    print("\n--- Evaluation Results (Original Predictions) ---")
    accuracy = accuracy_score(true_labels_flat, pred_labels_flat)
    print(f"Frame-Level Accuracy: {accuracy:.4f}")
    print(f"Average Test Loss: {avg_loss:.4f}")
    print("\nClassification Report (Frame Level):")
    report = classification_report(true_labels_flat, pred_labels_flat, target_names=[
                                   CLASS_NAMES[i] for i in range(NUM_CLASSES)], digits=4)
    print(report)

    # 7. Calculate and Print Metrics (Smoothed Predictions)
    print("\n--- Evaluation Results (Smoothed Predictions) ---")
    accuracy_smoothed = accuracy_score(true_labels_flat, smoothed_preds_flat)
    print(f"Frame-Level Accuracy (Smoothed): {accuracy_smoothed:.4f}")
    print("\nClassification Report (Frame Level, Smoothed):")
    report_smoothed = classification_report(true_labels_flat, smoothed_preds_flat, target_names=[
                                            CLASS_NAMES[i] for i in range(NUM_CLASSES)], digits=4)
    print(report_smoothed)

    # 8. Generate and Save Plots
    print("\nGenerating plots...")
    # Confusion Matrix (Original)
    plot_confusion_matrix(true_labels_flat, pred_labels_flat,
                          CLASS_NAMES, filename="confusion_matrix_test_original.png", title_suffix=" (Original)")

    # Confusion Matrix (Smoothed)
    plot_confusion_matrix(true_labels_flat, smoothed_preds_flat,
                          CLASS_NAMES, filename="confusion_matrix_test_smoothed.png", title_suffix=" (Smoothed)")

    # ROC Curve and AUC (based on original probabilities)
    plot_roc_curve(true_labels_flat, pred_probs_flat, NUM_CLASSES,
                   CLASS_NAMES, filename="roc_curve_test.png")

    print("\nEvaluation complete.")
