import h5py
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from itertools import product

# === Step 1: Load data ===
with h5py.File(r'B:\Education\CNN\EEG\Imagined Speech\Processing\CSD_Transformed_Epochs.h5', 'r') as f:
    data = f['epochs'][:]
    labels = f['labels'][:]

X = data.reshape(data.shape[0], -1)
y = np.array([label.decode() if isinstance(label, bytes) else str(label) for label in labels])

# === Step 2: Preprocessing ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

selector = SelectKBest(score_func=f_classif, k=250)
X_selected = selector.fit_transform(X_scaled, y)

# Polynomial Expansion
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_selected)  # ~31,000 features

# === Step 3: SVM Grid Search ===
C_values = [1, 10, 100]
gamma_values = ['scale', 0.1, 0.01]
param_grid = list(product(C_values, gamma_values))

best_acc = 0
best_params = None
results = {}

num_runs = 10

for C_val, gamma_val in param_grid:
    acc_list = []
    for run in range(num_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X_poly, y, test_size=0.2, stratify=y, random_state=run)

        clf = SVC(kernel='rbf', C=C_val, gamma=gamma_val, class_weight='balanced')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        acc_list.append(acc)

    mean_acc = np.mean(acc_list)
    std_acc = np.std(acc_list)
    results[f'C={C_val}, gamma={gamma_val}'] = (mean_acc, std_acc)

    if mean_acc > best_acc:
        best_acc = mean_acc
        best_params = (C_val, gamma_val)

    print(f"[C={C_val}, gamma={gamma_val}] → Mean Acc: {mean_acc:.4f} | Std: {std_acc:.4f}")

# === Step 4: Plot best config ===
best_label = f'C={best_params[0]}, gamma={best_params[1]}'
mean, std = results[best_label]

plt.figure(figsize=(6, 5))
plt.bar('CSD-SVM*', mean, yerr=std, capsize=10, color='skyblue', edgecolor='black', label='Optimized SVM')
plt.axhline(y=0.1, linestyle='--', color='red', label='Chance Level (0.1)')
plt.ylabel('Accuracy')
plt.title(f'Optimized CSD-SVM\n{best_label}\nAccuracy Over 10 Runs')
plt.legend()
plt.ylim(0, 0.4)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
