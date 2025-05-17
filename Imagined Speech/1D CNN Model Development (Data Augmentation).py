import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
import matplotlib.pyplot as plt
import random

# -------------------- Load Preprocessed Data --------------------

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

# -------------------- Data Augmentation Functions --------------------

def add_jitter(signal, sigma=0.05):
    return signal + np.random.normal(0, sigma, signal.shape)

def time_shift(signal, max_shift=20):
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(signal, shift, axis=0)

def scaling(signal, sigma=0.1):
    scaling_factors = np.random.normal(1.0, sigma, size=(signal.shape[1],))  # per channel
    return signal * scaling_factors

# -------------------- Custom Dataset Class --------------------

class EEGDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.augment:
            if random.random() < 0.5:
                x = add_jitter(x)
            if random.random() < 0.5:
                x = time_shift(x)
            if random.random() < 0.5:
                x = scaling(x)
        x = torch.tensor(x, dtype=torch.float32).permute(1, 0)  # (7, 750)
        y = torch.tensor(y, dtype=torch.long)
        return x, y

# -------------------- DataLoaders --------------------

train_dataset = EEGDataset(X_train, y_train_encoded, augment=True)
test_dataset = EEGDataset(X_test, y_test_encoded, augment=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# -------------------- EEG_CNN Model Definition --------------------

class EEG_CNN(nn.Module):
    def __init__(self, num_channels=7, seq_len=750, num_classes=10):
        super(EEG_CNN, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.fc1 = nn.Linear(64 * (seq_len // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

# -------------------- Training Function --------------------

def train_model(model, train_loader, test_loader, device, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    train_acc_list, test_acc_list = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        correct, total = 0, 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
        train_acc = correct / total * 100
        train_acc_list.append(train_acc)

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                _, preds = torch.max(outputs, 1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        test_acc = correct / total * 100
        test_acc_list.append(test_acc)

        print(f"Epoch [{epoch}/50], Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%")

    return train_acc_list, test_acc_list

# -------------------- Train the Model --------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EEG_CNN()
train_acc, test_acc = train_model(model, train_loader, test_loader, device)

# -------------------- Plot Accuracy --------------------

plt.figure(figsize=(10, 5))
plt.plot(train_acc, label="Train Accuracy")
plt.plot(test_acc, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("1D-CNN with Data Augmentation")
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
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=unique_labels))

kappa = cohen_kappa_score(y_true, y_pred)
print(f"Kappa Score: {kappa:.4f}")

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix (1D-CNN + Augmentation)")
plt.show()
