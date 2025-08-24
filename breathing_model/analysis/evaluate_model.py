import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Subset
from itertools import cycle
from scipy.signal import medfilt  

model_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'model', 'transformer_model'))
if model_dir not in sys.path:
    sys.path.append(model_dir)

try:
    from transformer_model import BreathPhaseTransformerSeq, BreathSeqDataset
except ImportError as e:
    print(f"Error importing from transformer_model: {e}")
    print(
        f"Ensure the path '{model_dir}' is correct and contains transformer_model.py")
    sys.exit(1)

DATA_DIR = "../archive/data-sequences"
MODEL_PATH = "../model/trained_models/1/transformer_model_88.pth"
BATCH_SIZE = 4  
N_MELS = 128 
NUM_CLASSES = 3
D_MODEL = 192 
NHEAD = 8 
NUM_TRANSFORMER_LAYERS = 6 
MEDIAN_FILTER_SIZE = 5 
SAMPLE_RATE = 44100
N_FFT = 2048  
HOP_LENGTH = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = {0: "exhale", 1: "inhale", 2: "silence"} 


def get_test_loader(data_dir, batch_size, seed=42):
    """Creates the test DataLoader using the same split logic as in training."""
    val_dataset = BreathSeqDataset(
        data_dir, sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    num_samples = len(val_dataset)
    indices = list(range(num_samples))

    test_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)
    return test_loader, indices


def evaluate(model, test_loader, criterion, device, median_filter_size=None):
    """Evaluates the model on the test set, applies optional smoothing, and returns metrics."""
    model.eval()
    all_preds_raw = []  # Store raw predictions before smoothing
    all_labels = []
    all_probs = []  # Store probabilities for ROC/AUC
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

            # Store frame-level predictions and labels for metrics calculation
            all_preds_raw.append(predicted.view(-1).cpu().numpy())  # Store raw preds
            all_labels.append(labels.view(-1).cpu().numpy())
            # (B*time_steps, num_classes)
            all_probs.append(probabilities.view(-1, NUM_CLASSES).cpu().numpy())

            total_frames += labels.numel()

    # Concatenate results from all batches
    all_preds_raw = np.concatenate(all_preds_raw)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs, axis=0)

    # --- Apply Median Filter Smoothing ---
    if median_filter_size and median_filter_size > 1:
        print(f"Applying median filter with size {median_filter_size}...")
        # Ensure the filter size is odd
        if median_filter_size % 2 == 0:
            median_filter_size += 1
            print(f"Adjusted median filter size to odd: {median_filter_size}")
        all_preds_smoothed = medfilt(all_preds_raw, kernel_size=median_filter_size)
        # Ensure dtype remains integer after filtering if medfilt returns float
        all_preds_smoothed = all_preds_smoothed.astype(np.int64)
        print("Smoothing complete.")
    else:
        print("No smoothing applied or filter size <= 1.")
        all_preds_smoothed = all_preds_raw  # Use raw predictions if no smoothing

    # Calculate average test loss
    avg_test_loss = test_loss / \
        len(test_loader.dataset) if test_loader.dataset else 0

    # Return smoothed predictions for metrics calculation
    return all_labels, all_preds_smoothed, all_probs, avg_test_loss


def plot_confusion_matrix(y_true, y_pred, classes, filename="confusion_matrix.png"):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes.values(), yticklabels=classes.values())
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Frame Level)')
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
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue',
                   'green', 'red', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:0.2f})')

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


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    print("Loading test data...")
    test_loader, test_indices = get_test_loader(DATA_DIR, BATCH_SIZE)
    if test_loader is None:
        print("Could not create test loader. Exiting.")
        sys.exit(1)
    test_dataset = test_loader.dataset
    # Recreate loader if Subset was used, ensuring correct n_mels from the original dataset
    if isinstance(test_dataset, Subset):
        original_dataset = test_dataset.dataset
        original_dataset.n_mels = N_MELS  
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    else:
        test_dataset.n_mels = N_MELS  
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please ensure the model has been trained and saved correctly.")
        sys.exit(1)

    model = BreathPhaseTransformerSeq(
        n_mels=N_MELS, 
        num_classes=NUM_CLASSES,
        d_model=D_MODEL,  
        nhead=NHEAD,  #
        num_transformer_layers=NUM_TRANSFORMER_LAYERS  
    )
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        sys.exit(1)

    criterion = torch.nn.CrossEntropyLoss()

    print("Evaluating model on the test set...")

    true_labels, pred_labels, pred_probs, avg_loss = evaluate(
        model, test_loader, criterion, DEVICE, median_filter_size=MEDIAN_FILTER_SIZE)

    if len(true_labels) == 0:
        print(
            "Evaluation resulted in no labels or predictions. Check data loading and model.")
        sys.exit(1)

    print("\n--- Evaluation Results ---")

    valid_indices = true_labels != -100
    true_labels_filtered = true_labels[valid_indices]
    pred_labels_filtered = pred_labels[valid_indices]
    pred_probs_filtered = pred_probs[valid_indices]  

    if len(true_labels_filtered) == 0:
        print(
            "Evaluation resulted in no valid (non-padded) labels or predictions. Check data and padding.")
        sys.exit(1)

    # Frame-level Accuracy (using filtered labels)
    accuracy = accuracy_score(true_labels_filtered, pred_labels_filtered)
    print(f"Frame-Level Accuracy: {accuracy:.4f}")

    # Loss (calculated earlier, does not need filtering as CrossEntropyLoss handles -100)
    print(f"Average Test Loss: {avg_loss:.4f}")

    # Precision, Recall, F1-score (per class, macro, weighted)
    print("\nClassification Report (Frame Level):")

    print("Unique true labels (filtered):", np.unique(true_labels_filtered))
    print("Unique predicted labels (filtered):", np.unique(pred_labels_filtered))

    # Calculate report using filtered labels
    report = classification_report(true_labels_filtered, pred_labels_filtered, target_names=[
        CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())], digits=4, zero_division=0)
    print(report)

    print("\nGenerating plots...")

    # Confusion Matrix
    plot_confusion_matrix(true_labels_filtered, pred_labels_filtered,
                          CLASS_NAMES, filename="../model/trained_models/15/confusion_matrix_test.png")

    # ROC Curve and AUC
    unique_classes_present = np.unique(true_labels_filtered)

    roc_class_names = {i: CLASS_NAMES[i] for i in unique_classes_present if i in CLASS_NAMES}
    plot_roc_curve(true_labels_filtered, pred_probs_filtered, len(roc_class_names),
                   roc_class_names, filename="../model/trained_models/15/roc_curve_test.png")

    print("\nEvaluation complete.")
