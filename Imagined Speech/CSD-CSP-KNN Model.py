import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# === Step 1: Load CSP features ===
with h5py.File(r'B:\Education\CNN\EEG\Imagined Speech\Processing\CSD_CSP_Transformed_Epochs.h5', 'r') as f:
    X = f['features'][:]
    y = f['labels'][:]

# Decode string labels if needed
y = np.array([label.decode() if isinstance(label, bytes) else str(label) for label in y])

# === Step 2: Standardize the CSP features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Step 3: KNN for multiple k values ===
k_values = [1, 3, 5, 7, 9]
results = {}
num_runs = 10

for k in k_values:
    accuracies = []
    for run in range(num_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=run)

        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    results[f'K={k}'] = (mean_acc, std_acc)
    print(f"K={k} → Mean Acc: {mean_acc:.4f} | Std: {std_acc:.4f}")

# === Step 4: Plot results ===
labels = list(results.keys())
means = [results[k][0] for k in labels]
stds = [results[k][1] for k in labels]

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, means, yerr=stds, capsize=10, color='skyblue', edgecolor='black', label='Offline Accuracy')
plt.axhline(y=0.1, linestyle='--', color='red', label='Chance Level (0.1)')
plt.ylabel('Accuracy')
plt.title('CSD-CSP-KNN Accuracy vs Different k Values (Standardized CSP Features)')
plt.ylim(0, 0.35)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
