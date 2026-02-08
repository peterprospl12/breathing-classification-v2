import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

output_folder = "confusion_matrices"
os.makedirs(output_folder, exist_ok=True)

def plot_confusion_matrix(cm, title, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Rzeczywista klasa')
    plt.xlabel('Predykcja')
    plt.tight_layout()

    filename = f"{title.replace(' ', '_').replace(',', '').lower()}.png"
    plt_filename = os.path.join(output_folder, filename)
    plt.savefig(plt_filename, dpi=300)
    print(f"Saved confusion matrix: {plt_filename}")

    plt.show()

def calculate_accuracy(cm):
    return np.diag(cm).sum() / cm.sum()

labels = ['Wydech', 'Inne']

cm_good = np.array([
    [177, 2], # Exhale: correctly as Exhale, incorrectly as other
    [4, 185], # Other: incorrectly as Exhale, correctly as other
])

matrices = [
        (cm_good, "Wszystkie mikrofony (trzyklasowy transformer)")
    ]
print(f"Saving confusion matrices to folder: {output_folder}")
for cm, title in matrices:
    plot_confusion_matrix(cm, title, labels)
    accuracy = calculate_accuracy(cm)
    print(f"Accuracy ({title}): {accuracy:.2%}")
