import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# === Load Data ===
with h5py.File(r"B:\Education\CNN\EEG\Imagined Speech\Processing\CSD_Transformed_Epochs.h5", 'r') as f:
    X_csd = f['epochs'][:]
    y_csd = f['labels'][:]

with h5py.File(r"B:\Education\CNN\EEG\Imagined Speech\Processing\CSD_CSP_Transformed_Epochs.h5", 'r') as f:
    X_csp = f['features'][:]
    y_csp = f['labels'][:]

# === Decode labels if needed ===
y_csd = np.array([label.decode() if isinstance(label, bytes) else str(label) for label in y_csd])
y_csp = np.array([label.decode() if isinstance(label, bytes) else str(label) for label in y_csp])

# === Preprocess CSD ===
X_csd_flat = X_csd.reshape(X_csd.shape[0], -1)
scaler_csd = StandardScaler().fit(X_csd_flat)
X_csd_scaled = scaler_csd.transform(X_csd_flat)

selector = SelectKBest(f_classif, k=250).fit(X_csd_scaled, y_csd)
X_csd_selected = selector.transform(X_csd_scaled)

# === Preprocess CSP ===
scaler_csp = StandardScaler().fit(X_csp)
X_csp_scaled = scaler_csp.transform(X_csp)

# === Run indices (100 per run) ===
run_indices = np.repeat(np.arange(7), 100)

# === Model Definitions ===
models = {
    "CSD-LR": LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial'),
    "CSD-CSP-LR": LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial'),
    "CSD-KNN": KNeighborsClassifier(n_neighbors=5),
    "CSD-CSP-KNN": KNeighborsClassifier(n_neighbors=5),
    "CSD-RF": RandomForestClassifier(n_estimators=300, max_depth=15, class_weight='balanced'),
    "CSD-CSP-RF": RandomForestClassifier(n_estimators=300, max_depth=15, class_weight='balanced'),
    "CSD-SVM": SVC(kernel='rbf', C=10, gamma=0.01, class_weight='balanced'),
    "CSD-CSP-SVM": SVC(kernel='rbf', C=10, gamma=0.01, class_weight='balanced')
}

# === LORO-CV Evaluation ===
results = {}

for name, model in models.items():
    print(f"\nRunning {name}...")
    accuracies = []

    if "CSP" in name:
        X = X_csp_scaled
        y = y_csp
    else:
        X = X_csd_selected
        y = y_csd

    for run in range(7):
        train_idx = run_indices != run
        test_idx = run_indices == run

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f" Run {run+1}: {acc:.4f}")

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    results[name] = (mean_acc, std_acc)
    print(f" Mean Accuracy: {mean_acc:.4f} | Std: {std_acc:.4f}")

# === Plot Results ===
labels = list(results.keys())
means = [results[k][0] for k in labels]
stds = [results[k][1] for k in labels]

plt.figure(figsize=(14, 6))
bars = plt.bar(labels, means, yerr=stds, capsize=10, color='deepskyblue', edgecolor='black')
plt.axhline(y=0.1, linestyle='--', color='red', label='Chance Level (0.1)')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Accuracy for Classical Models')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
