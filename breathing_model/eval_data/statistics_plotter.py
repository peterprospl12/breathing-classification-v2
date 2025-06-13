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
    plt.ylabel('Klasa rzeczywista')
    plt.xlabel('Predykcja')
    plt.tight_layout()

    filename = f"{title.replace(' ', '_').replace(',', '').lower()}.png"
    plt_filename = os.path.join(output_folder, filename)
    plt.savefig(plt_filename, dpi=300)
    print(f"Zapisano macierz pomyłek: {plt_filename}")

    plt.show()

def calculate_accuracy(cm):
    return np.diag(cm).sum() / cm.sum()

labels = ['Wdech', 'Wydech', 'Cisza']

cm_tomasz = np.array([
    [45, 2, 1],  # Wdech: prawidłowo jako wdech, błędnie jako wydech, błędnie jako cisza
    [0, 47, 0],    # Wydech: błędnie jako wdech, prawidłowo jako wydech, błędnie jako cisza
    [0, 1, 23]     # Cisza: błędnie jako wdech, błędnie jako wydech, prawidłowo jako cisza
])

cm_piotr = np.array([
    [56, 11, 5],
    [0, 79, 0],
    [0, 0, 16]
])

cm_iwo = np.array([
    [42, 0, 1],
    [0, 29, 3],
    [0, 0, 25]
])

cm_good = np.array([
    [23, 2, 0],
    [0, 24, 0],
    [0, 1, 11]
])

cm_medium = np.array([
    [130, 13, 7],
    [0, 137, 8],
    [0, 0, 47]
])

cm_bad = np.array([
    [3, 0, 12],
    [5, 9, 0],
    [0, 0, 4]
])

cm_all = cm_bad + cm_medium + cm_good

matrices = [
    (cm_tomasz, "Wszystkie dane Tomasz"),
    (cm_piotr, "Wszystkie dane Piotr"),
    (cm_iwo, "Wszystkie dane Iwo"),
    (cm_good, "Wszystkie dobre mikrofony"),
    (cm_medium, "Wszystkie średnie mikrofony"),
    (cm_bad, "Wszystkie słabe mikrofony"),
    (cm_all, "Wszystkie dane")
]
print(f"Zapisywanie macierzy konfuzji do folderu: {output_folder}")

for cm, title in matrices:
    plot_confusion_matrix(cm, title, labels)
    accuracy = calculate_accuracy(cm)
    print(f"Dokładność ({title}): {accuracy:.2%}")



cykle_model_tomasz = 16 + 9 + 15 + 8
cykle_tomasz = 15 + 9 + 15 + 8
cykle_model_piotr = 11 + 16 + 13 + 13 + 34
cykle_piotr = 11 + 10 + 13 + 9 + 36
cykle_model_iwo = 6 + 13 + 7 + 6 + 12
cykle_iwo = 6 + 12 + 6 + 8 + 10
cykle_model_good = 16 + 9
cykle_good = 15 + 9
cykle_model_medium = 11 + 16 + 13 + 13 + 34 + 15 + 8 + 6 + 13 + 7 + 6 + 12
cykle_medium = 11 + 10 + 13 + 9 + 36 + 15 + 8 + 6 + 12 + 6 + 8 + 10
cykle_model_bad = 16
cykle_bad = + 14
cykle_model_all = cykle_model_bad + cykle_model_medium + cykle_model_good
cykle_all = cykle_bad + cykle_medium + cykle_good

# Print breathing cycle statistics
print("======= BREATHING CYCLE STATISTICS =======")

def calculate_difference(model_cycles, actual_cycles):
    if actual_cycles == 0:
        return 0  # Avoid division by zero
    return ((model_cycles - actual_cycles) / actual_cycles) * 100

# Tomasz statistics
print("\n=== TOMASZ ===")
print(f"Actual breathing cycles: {cykle_tomasz}")
print(f"Model breathing cycles: {cykle_model_tomasz}")
diff_tomasz = calculate_difference(cykle_model_tomasz, cykle_tomasz)
print(f"Percentage difference: {diff_tomasz:.2f}%")

# Piotr statistics
print("\n=== PIOTR ===")
print(f"Actual breathing cycles: {cykle_piotr}")
print(f"Model breathing cycles: {cykle_model_piotr}")
diff_piotr = calculate_difference(cykle_model_piotr, cykle_piotr)
print(f"Percentage difference: {diff_piotr:.2f}%")

# Iwo statistics
print("\n=== IWO ===")
print(f"Actual breathing cycles: {cykle_iwo}")
print(f"Model breathing cycles: {cykle_model_iwo}")
diff_iwo = calculate_difference(cykle_model_iwo, cykle_iwo)
print(f"Percentage difference: {diff_iwo:.2f}%")

# Good microphones statistics
print("\n=== GOOD MICROPHONES ===")
print(f"Actual breathing cycles: {cykle_good}")
print(f"Model breathing cycles: {cykle_model_good}")
diff_good = calculate_difference(cykle_model_good, cykle_good)
print(f"Percentage difference: {diff_good:.2f}%")

# Medium microphones statistics
print("\n=== MEDIUM MICROPHONES ===")
print(f"Actual breathing cycles: {cykle_medium}")
print(f"Model breathing cycles: {cykle_model_medium}")
diff_medium = calculate_difference(cykle_model_medium, cykle_medium)
print(f"Percentage difference: {diff_medium:.2f}%")

# Bad microphones statistics
print("\n=== POOR MICROPHONES ===")
print(f"Actual breathing cycles: {cykle_bad}")
print(f"Model breathing cycles: {cykle_model_bad}")
diff_bad = calculate_difference(cykle_model_bad, cykle_bad)
print(f"Percentage difference: {diff_bad:.2f}%")

# All data statistics
print("\n=== ALL DATA ===")
print(f"Actual breathing cycles: {cykle_all}")
print(f"Model breathing cycles: {cykle_model_all}")
diff_all = calculate_difference(cykle_model_all, cykle_all)
print(f"Percentage difference: {diff_all:.2f}%")