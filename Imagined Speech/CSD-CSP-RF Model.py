import h5py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# === Step 1: Load CSP features ===
with h5py.File(r'B:\Education\CNN\EEG\Imagined Speech\Processing\CSD_CSP_Transformed_Epochs.h5', 'r') as f:
    X = f['features'][:]  # shape: (700, 6)
    y = f['labels'][:]

# Decode string labels
y = np.array([label.decode() if isinstance(label, bytes) else str(label) for label in y])

# === Step 2: Standardize CSP features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Step 3: Run Random Forest multiple times ===
num_runs = 10
accuracies = []

for run in range(num_runs):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=run)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=3,
        class_weight='balanced',
        random_state=run
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"Run {run + 1}: Accuracy = {acc:.4f}")

# === Step 4: Plot Accuracy ===
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)

plt.figure(figsize=(6, 5))
plt.bar('CSD-CSP-RF', mean_acc, yerr=std_acc, capsize=10, color='skyblue', edgecolor='black', label='Offline Accuracy')
plt.axhline(y=0.1, linestyle='--', color='red', label='Chance Level (0.1)')
plt.ylabel('Accuracy')
plt.title('CSD-CSP-RF Accuracy Over Multiple Runs')
plt.legend()
plt.ylim(0, 0.35)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
