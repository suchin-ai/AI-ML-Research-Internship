import h5py
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# === Load CSP data ===
with h5py.File(r'B:\Education\CNN\EEG\Imagined Speech\Processing\CSD_CSP_Transformed_Epochs.h5', 'r') as f:
    X = f['features'][:]  # shape: (700, 6)
    y = f['labels'][:]

# Decode labels
y = np.array([label.decode() if isinstance(label, bytes) else str(label) for label in y])

# Create run indices: 100 epochs per run, ordered
run_indices = np.repeat(np.arange(7), 100)

# === Preprocessing: Scaling + Polynomial Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)  # shape: (700, 27)

# === SVM Config ===
C_val = 10
gamma_val = 0.01

accuracies = []

# === Leave-One-Run-Out CV ===
for test_run in range(7):
    train_idx = run_indices != test_run
    test_idx = run_indices == test_run

    X_train, X_test = X_poly[train_idx], X_poly[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = SVC(kernel='rbf', C=C_val, gamma=gamma_val, class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"Run {test_run + 1}: Accuracy = {acc:.4f}")

# === Results ===
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)

plt.figure(figsize=(6, 5))
plt.bar('CSD-CSP-SVM (LORO)', mean_acc, yerr=std_acc, capsize=10, color='deepskyblue', edgecolor='black')
plt.axhline(y=0.1, linestyle='--', color='red', label='Chance Level (0.1)')
plt.ylabel('Accuracy')
plt.title(f'CSD-CSP-SVM Leave-One-Run-Out CV\nMean Acc = {mean_acc:.4f}')
plt.legend()
plt.ylim(0, 0.45)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
