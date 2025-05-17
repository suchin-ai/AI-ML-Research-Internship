import h5py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

# Load data
with h5py.File(r'B:\Education\CNN\EEG\Imagined Speech\Processing\CSD_Transformed_Epochs.h5', 'r') as f:
    data = f['epochs'][:]    # (700, 750, 7)
    labels = f['labels'][:]

X = data.reshape(data.shape[0], -1)  # (700, 5250)
y = np.array([label.decode() if isinstance(label, bytes) else str(label) for label in labels])

# Preprocess: Scaling + Feature Selection
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

selector = SelectKBest(score_func=f_classif, k=250)  # Tune this k if needed
X_selected = selector.fit_transform(X_scaled, y)

# Run 10 times
num_runs = 10
accuracies = []

for run in range(num_runs):
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, stratify=y, random_state=run)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=4,
        class_weight='balanced',
        random_state=run
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"Run {run + 1}: Accuracy = {acc:.4f}")

# Plot
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)

plt.figure(figsize=(6, 5))
plt.bar('CSD-RF', mean_acc, yerr=std_acc, capsize=10, color='skyblue', edgecolor='black', label='Offline Accuracy')
plt.axhline(y=0.1, linestyle='--', color='red', label='Chance Level (0.1)')
plt.ylabel('Accuracy')
plt.title('Tuned CSD-RF Accuracy Over Multiple Runs')
plt.legend()
plt.ylim(0, 0.35)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
