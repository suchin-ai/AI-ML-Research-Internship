import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load CSD-transformed data
with h5py.File(r'B:\Education\CNN\EEG\Imagined Speech\Processing\CSD_Transformed_Epochs.h5', 'r') as f:
    data = f['epochs'][:]    # (700, 750, 7)
    labels = f['labels'][:]

# Flatten EEG: (700, 750, 7) → (700, 5250)
X = data.reshape(data.shape[0], -1)
y = np.array([label.decode() if isinstance(label, bytes) else str(label) for label in labels])

# Store accuracies across runs
num_runs = 10
accuracies = []

for run in range(num_runs):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=run)
    
    # Train classifier
    clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"Run {run + 1}: Accuracy = {acc:.4f}")

# Convert to numpy array for stats
accuracies = np.array(accuracies)
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)

# Plot
plt.figure(figsize=(6, 5))
plt.bar('CSD-LR', mean_acc, yerr=std_acc, capsize=10, color='skyblue', edgecolor='black', label='Offline Accuracy')
plt.axhline(y=0.1, linestyle='--', color='red', label='Chance Level (0.1)')
plt.ylabel('Accuracy')
plt.title('CSD-LR Accuracy Over Multiple Runs')
plt.legend()
plt.ylim(0, 0.35)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
