import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
import matplotlib.pyplot as plt

# -------------------- Load and Prepare Data --------------------

file_path = r'B:\Education\CNN\EEG\Imagined Speech\Processing\Preprocessed_Split_Data.h5'
with h5py.File(file_path, 'r') as f:
    X_train = np.array(f['X_train'])
    X_test = np.array(f['X_test'])
    y_train = np.array(f['y_train']).astype(str)
    y_test = np.array(f['y_test']).astype(str)

unique_labels = np.unique(y_train)
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}

y_train_encoded = np.array([label_to_index[label] for label in y_train])
y_test_encoded = np.array([label_to_index[label] for label in y_test])

# For 1D CNN: reshape to (batch, channels, time)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

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

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# -------------------- Training Loop --------------------

def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=100):
    model.to(device)
    train_acc_list, test_acc_list = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        correct, total = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total * 100
        train_acc_list.append(train_acc)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        test_acc = correct / total * 100
        test_acc_list.append(test_acc)

        print(f"Epoch [{epoch}/100], Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%")

    return train_acc_list, test_acc_list

# -------------------- Run EEGNet1D --------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EEGNet1D()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

train_acc, test_acc = train_model(model, train_loader, test_loader, criterion, optimizer, device)

# -------------------- Accuracy Plot --------------------

plt.figure(figsize=(10, 5))
plt.plot(train_acc, label="Train Accuracy")
plt.plot(test_acc, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Train vs Test Accuracy (EEGNet 1D)")
plt.legend()
plt.grid(True)
plt.show()

# -------------------- Final Evaluation --------------------

model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=unique_labels))

kappa = cohen_kappa_score(y_true, y_pred)
print(f"Kappa Score: {kappa:.4f}")

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()
