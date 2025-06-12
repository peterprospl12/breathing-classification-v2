import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


# Define output folder for all confusion matrices
output_folder = "confusion_matrices"
os.makedirs(output_folder, exist_ok=True)


# Function to create and display confusion matrix
def plot_confusion_matrix(cm, title, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True class')
    plt.xlabel('Prediction')
    plt.tight_layout()

    # Save to PNG in the specified folder
    filename = f"{title.replace(' ', '_').replace(',', '').lower()}.png"
    plt_filename = os.path.join(output_folder, filename)
    plt.savefig(plt_filename, dpi=300)  # Higher resolution for better quality
    print(f"Confusion matrix visualization saved to: {plt_filename}")

    plt.show()


# Confusion matrix data (individual tests)
# 1. Tomasz - good microphone
y_true_tomasz_good = ['inhale'] * 15 + ['exhale'] * 15 + ['silence'] * 10
y_pred_tomasz_good = ['inhale'] * 13 + ['exhale'] * 2 + ['exhale'] * 15 + ['silence'] * 10
cm_tomasz_good = confusion_matrix(y_true_tomasz_good, y_pred_tomasz_good, labels=['inhale', 'exhale', 'silence'])

# 2. Piotr - good microphone
y_true_piotr_good = ['inhale'] * 10 + ['exhale'] * 9 + ['silence'] * 10
y_pred_piotr_good = ['inhale'] * 10 + ['exhale'] * 9 + ['exhale'] * 1 + ['silence'] * 9
cm_piotr_good = confusion_matrix(y_true_piotr_good, y_pred_piotr_good, labels=['inhale', 'exhale', 'silence'])

# 3. Tomasz - medium microphone
y_true_tomasz_medium = ['inhale'] * (13 + 36 + 8) + ['exhale'] * (9 + 36 + 8) + ['silence'] * 15
y_pred_tomasz_medium = ['inhale'] * (2 + 28 + 7) + ['exhale'] * (11 + 3) + ['silence'] * (0 + 5 + 1) + ['exhale'] * (
            9 + 36 + 8) + ['silence'] * 15
cm_tomasz_medium = confusion_matrix(y_true_tomasz_medium, y_pred_tomasz_medium, labels=['inhale', 'exhale', 'silence'])

# 4. Piotr - medium microphone
y_true_piotr_medium = ['inhale'] * (10 + 15 + 10) + ['exhale'] * (9 + 15 + 9) + ['silence'] * 15
y_pred_piotr_medium = ['inhale'] * (7 + 15 + 2) + ['exhale'] * (3 + 0 + 8) + ['exhale'] * (9 + 15 + 9) + [
    'silence'] * 15
cm_piotr_medium = confusion_matrix(y_true_piotr_medium, y_pred_piotr_medium, labels=['inhale', 'exhale', 'silence'])

# 5. Tomasz - poor microphone
y_true_tomasz_bad = ['inhale'] * 15 + ['exhale'] * 14 + ['silence'] * 10
y_pred_tomasz_bad = ['inhale'] * 3 + ['silence'] * 12 + ['inhale'] * 5 + ['exhale'] * 9 + ['silence'] * 10
cm_tomasz_bad = confusion_matrix(y_true_tomasz_bad, y_pred_tomasz_bad, labels=['inhale', 'exhale', 'silence'])

# Create aggregated data sets
# All Tomasz data
y_true_tomasz_all = y_true_tomasz_good + y_true_tomasz_medium + y_true_tomasz_bad
y_pred_tomasz_all = y_pred_tomasz_good + y_pred_tomasz_medium + y_pred_tomasz_bad
cm_tomasz_all = confusion_matrix(y_true_tomasz_all, y_pred_tomasz_all, labels=['inhale', 'exhale', 'silence'])

# All Piotr data
y_true_piotr_all = y_true_piotr_good + y_true_piotr_medium
y_pred_piotr_all = y_pred_piotr_good + y_pred_piotr_medium
cm_piotr_all = confusion_matrix(y_true_piotr_all, y_pred_piotr_all, labels=['inhale', 'exhale', 'silence'])

# All good microphone data
y_true_good_all = y_true_tomasz_good + y_true_piotr_good
y_pred_good_all = y_pred_tomasz_good + y_pred_piotr_good
cm_good_all = confusion_matrix(y_true_good_all, y_pred_good_all, labels=['inhale', 'exhale', 'silence'])

# All medium microphone data
y_true_medium_all = y_true_tomasz_medium + y_true_piotr_medium
y_pred_medium_all = y_pred_tomasz_medium + y_pred_piotr_medium
cm_medium_all = confusion_matrix(y_true_medium_all, y_pred_medium_all, labels=['inhale', 'exhale', 'silence'])

# All bad microphone data (only Tomasz data available)
cm_bad_all = cm_tomasz_bad  # Since we only have Tomasz's bad microphone data

# All data combined
y_true_all = y_true_tomasz_all + y_true_piotr_all
y_pred_all = y_pred_tomasz_all + y_pred_piotr_all
cm_all = confusion_matrix(y_true_all, y_pred_all, labels=['inhale', 'exhale', 'silence'])

# Generating and saving confusion matrices
labels = ['Inhale', 'Exhale', 'Silence']

print(f"Saving all confusion matrices to folder: {output_folder}")

# Save individual matrices
plot_confusion_matrix(cm_tomasz_good, "Tomasz dobry mikrofon", labels)
plot_confusion_matrix(cm_piotr_good, "Piotr dobry mikrofon", labels)
plot_confusion_matrix(cm_tomasz_medium, "Tomasz średni mikrofon", labels)
plot_confusion_matrix(cm_piotr_medium, "Piotr średni mikrofon", labels)
plot_confusion_matrix(cm_tomasz_bad, "Tomasz słaby mikrofon", labels)

# Save aggregated matrices
plot_confusion_matrix(cm_tomasz_all, "Wszystkie dane Tomasza", labels)
plot_confusion_matrix(cm_piotr_all, "Wszystkie dane Piotra", labels)
plot_confusion_matrix(cm_good_all, "Wszystkie dobre mikrofony", labels)
plot_confusion_matrix(cm_medium_all, "Wszystkie średnie mikrofony", labels)
plot_confusion_matrix(cm_bad_all, "Wszystkie słabe mikrofony", labels)
plot_confusion_matrix(cm_all, "Wszystkie dane", labels)

# Calculating accuracy for aggregated matrices
accuracy_tomasz_all = np.diag(cm_tomasz_all).sum() / cm_tomasz_all.sum()
accuracy_piotr_all = np.diag(cm_piotr_all).sum() / cm_piotr_all.sum()
accuracy_good_all = np.diag(cm_good_all).sum() / cm_good_all.sum()
accuracy_medium_all = np.diag(cm_medium_all).sum() / cm_medium_all.sum()
accuracy_bad_all = np.diag(cm_bad_all).sum() / cm_bad_all.sum()
accuracy_all = np.diag(cm_all).sum() / cm_all.sum()

print(f"Accuracy (All Tomasz data): {accuracy_tomasz_all:.2%}")
print(f"Accuracy (All Piotr data): {accuracy_piotr_all:.2%}")
print(f"Accuracy (All good microphone data): {accuracy_good_all:.2%}")
print(f"Accuracy (All medium microphone data): {accuracy_medium_all:.2%}")
print(f"Accuracy (All bad microphone data): {accuracy_bad_all:.2%}")
print(f"Accuracy (All data): {accuracy_all:.2%}")