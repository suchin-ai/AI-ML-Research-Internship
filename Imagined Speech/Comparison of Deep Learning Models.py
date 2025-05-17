import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# -------------------- EEGNet 1D Model --------------------
class EEGNet1D(nn.Module):
    def __init__(self, num_channels=7, num_samples=750, num_classes=10):
        super(EEGNet1D, self).__init__()
        
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(num_channels, 16, kernel_size=64, padding=32, bias=False),
            nn.BatchNorm1d(16)
        )

        self.depthwise_conv = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=1, groups=16, bias=False),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=4),
            nn.Dropout(0.25)
        )

        self.separable_conv = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=16, padding=8, bias=False),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=8),
            nn.Dropout(0.25)
        )

        self.fc = nn.Linear(32 * (num_samples // 32), num_classes)

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# -------------------- Function to Run LORO-CV --------------------
def run_loro_eegnet(file_path, label_text, color):
    with h5py.File(file_path, 'r') as f:
        X = f['epochs'][:]  # Expected shape: (700, 750, 7)
        y = f['labels'][:]

    X = np.transpose(X, (0, 2, 1))  # Convert to (700, 7, 750)

    le = LabelEncoder()
    y_encoded = le.fit_transform([label.decode() if isinstance(label, bytes) else label for label in y])

    run_indices = np.repeat(np.arange(7), 100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracies = []

    for run in range(7):
        print(f"{label_text} → Run {run+1}/7")
        train_idx = run_indices != run
        test_idx = run_indices == run

        X_train = torch.tensor(X[train_idx], dtype=torch.float32)
        X_test = torch.tensor(X[test_idx], dtype=torch.float32)
        y_train = torch.tensor(y_encoded[train_idx], dtype=torch.long)
        y_test = torch.tensor(y_encoded[test_idx], dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

        model = EEGNet1D().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(30):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                preds = model(xb).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(yb.numpy())

        acc = accuracy_score(all_true, all_preds)
        accuracies.append(acc)
        print(f"  Accuracy: {acc:.4f}")

    return label_text, np.mean(accuracies), np.std(accuracies), color

# -------------------- Run All EEGNet Variants --------------------

results = []
results.append(run_loro_eegnet(
    r"B:\Education\CNN\EEG\Imagined Speech\Processing\Extracted_Epochs.h5",
    "Raw-EEGNet (1D)", "royalblue"
))

results.append(run_loro_eegnet(
    r"B:\Education\CNN\EEG\Imagined Speech\Processing\Preprocessed_Epochs.h5",
    "Preprocessed-EEGNet (1D)", "green"
))

results.append(run_loro_eegnet(
    r"B:\Education\CNN\EEG\Imagined Speech\Processing\CSD_Transformed_Epochs.h5",
    "CSD-EEGNet (1D)", "orange"
))

# -------------------- Plot Final Comparison --------------------

labels, means, stds, colors = zip(*results)

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, means, yerr=stds, capsize=10, color=colors, edgecolor='black')
plt.axhline(0.1, linestyle='--', color='red', label='Chance Level (0.1)')
plt.title("EEGNet (1D) - Cross-Validation Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, max(means) + 0.1)
plt.legend()
plt.tight_layout()
plt.show()
