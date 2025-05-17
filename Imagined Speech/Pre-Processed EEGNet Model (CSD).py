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
            nn.Conv1d(16, 32, kernel_size=1, groups=16, bias=False),  # Depthwise
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=4),
            nn.Dropout(0.5)
        )

        self.separable_conv = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=16, padding=8, bias=False),  # Separable
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=8),
            nn.Dropout(0.5)
        )

        self.fc = nn.Linear(32 * (num_samples // 32), num_classes)

    def forward(self, x):  # (B, 7, 750)
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# -------------------- Load and Prepare Data --------------------

with h5py.File(r"B:\Education\CNN\EEG\Imagined Speech\Processing\CSD_Transformed_Epochs.h5", 'r') as f:
    X = f['epochs'][:]    # shape: (700, 750, 7)
    y = f['labels'][:]

X = np.transpose(X, (0, 2, 1))  # (700, 7, 750)

le = LabelEncoder()
y_encoded = le.fit_transform([label.decode() if isinstance(label, bytes) else label for label in y])

run_indices = np.repeat(np.arange(7), 100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accuracies = []

# -------------------- LORO Cross-Validation --------------------

for run in range(7):
    print(f"\n=== Run {run+1}/7 ===")
    train_idx = run_indices != run
    test_idx = run_indices == run

    X_train = X[train_idx]
    y_train = y_encoded[train_idx]
    X_test = X[test_idx]
    y_test = y_encoded[test_idx]

    # Reshape for 1D CNN input: (batch, channels, time)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Model
    model = EEGNet1D(num_channels=7, num_samples=750, num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train
    model.train()
    for epoch in range(30):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.numpy())

    acc = accuracy_score(all_true, all_preds)
    accuracies.append(acc)
    print(f"Accuracy: {acc:.4f}")

# -------------------- Plot Results --------------------

mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)

plt.figure(figsize=(6, 5))
plt.bar("Pre-Processed-CSD-EEGNet (1D)", mean_acc, yerr=std_acc, capsize=10, color="royalblue")
plt.axhline(0.1, color='red', linestyle='--', label='Chance Level')
plt.title(f"Pre-Processed-CSD-EEGNet (1D) LORO Accuracy\nMean = {mean_acc:.4f}")
plt.ylabel("Accuracy")
plt.ylim(0, 0.3)
plt.legend()
plt.tight_layout()
plt.show()
