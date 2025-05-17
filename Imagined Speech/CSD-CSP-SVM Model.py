import h5py
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from itertools import product

# === Load data ===
with h5py.File(r'B:\Education\CNN\EEG\Imagined Speech\Processing\CSD_CSP_Transformed_Epochs.h5', 'r') as f:
    X = f['features'][:]  # shape: (700, 6)
    y = f['labels'][:]
y = np.array([label.decode() if isinstance(label, bytes) else str(label) for label in y])

# === Preprocessing ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optional: Degree-3 polynomial features
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X_scaled)  # ~83 features from 6 CSP components

# === Extended hyperparameter grid ===
C_values = [0.1, 1, 10, 100]
gamma_values = ['scale', 'auto', 0.01, 0.001]
param_grid = list(product(C_values, gamma_values))

best_acc = 0
best_params = None
results = {}

# === Run grid search with 5-fold CV ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for C_val, gamma_val in param_grid:
    fold_accuracies = []
    
    for train_index, test_index in skf.split(X_poly, y):
        X_train, X_test = X_poly[train_index], X_poly[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = SVC(kernel='rbf', C=C_val, gamma=gamma_val, class_weight='balanced')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)
    
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    results[f'C={C_val}, gamma={gamma_val}'] = (mean_acc, std_acc)

    if mean_acc > best_acc:
        best_acc = mean_acc
        best_params = (C_val, gamma_val)

    print(f"[C={C_val}, gamma={gamma_val}] → Mean Acc: {mean_acc:.4f} | Std: {std_acc:.4f}")

# === Plot best result ===
best_label = f'C={best_params[0]}, gamma={best_params[1]}'
mean, std = results[best_label]

plt.figure(figsize=(6, 5))
plt.bar('CSD-CSP-SVM**', mean, yerr=std, capsize=10, color='skyblue', edgecolor='black', label='CV Accuracy')
plt.axhline(y=0.1, linestyle='--', color='red', label='Chance Level (0.1)')
plt.ylabel('Accuracy')
plt.title(f'Optimized CSD-CSP-SVM (Degree 3 Poly)\n{best_label} | 5-Fold CV')
plt.legend()
plt.ylim(0, 0.45)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
