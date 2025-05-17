import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
import matplotlib.pyplot as plt
import random

# -------------------- Load Data --------------------
data_path = r'B:\Education\CNN\EEG\Imagined Speech\Processing\Preprocessed_Split_Data.h5'
with h5py.File(data_path, 'r') as f:
    X_train = np.array(f['X_train'])
    X_test = np.array(f['X_test'])
    y_train = np.array(f['y_train']).astype(str)
    y_test = np.array(f['y_test']).astype(str)

# Label Encoding
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
    scale = np.random.normal(1.0, sigma, size=(signal.shape[1],))
    return signal * scale

# -------------------- Dataset --------------------
class EEGTransformerDataset(Dataset):
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

        x = torch.tensor(x, dtype=torch.float32)  # shape: (750, 7)
        y = torch.tensor(y, dtype=torch.long)
        return x, y

# -------------------- Positional Encoding --------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# -------------------- Transformer Model --------------------
class EEGTransformer(nn.Module):
    def __init__(self, input_dim=7, seq_len=750, num_classes=10, d_model=64, nhead=4, num_layers=1):
        super(EEGTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)       # (batch, seq_len, d_model)
        x = self.pos_enc(x)          # (batch, seq_len, d_model)
        x = self.transformer(x)      # (batch, seq_len, d_model)
        x = x.permute(0, 2, 1)       # (batch, d_model, seq_len)
        x = self.pool(x).squeeze(-1) # (batch, d_model)
        return self.classifier(x)

# -------------------- Data Loaders --------------------
train_dataset = EEGTransformerDataset(X_train, y_train_encoded, augment=True)
test_dataset = EEGTransformerDataset(X_test, y_test_encoded, augment=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# -------------------- Train Setup --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EEGTransformer().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

train_acc_list, test_acc_list, train_loss_list = [], [], []

# -------------------- Training Loop --------------------
for epoch in range(1, 51):
    model.train()
    correct, total, total_loss = 0, 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
        total_loss += loss.item()
    train_acc = correct / total * 100
    train_loss = total_loss / len(train_loader)
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    test_acc = correct / total * 100
    test_acc_list.append(test_acc)
    scheduler.step()

    print(f"Epoch [{epoch}/50] - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% - Test Acc: {test_acc:.2f}%")

# -------------------- Evaluation --------------------
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(y_batch.numpy())
        y_pred.extend(predicted.cpu().numpy())

# Generate Report
report = classification_report(y_true, y_pred, target_names=unique_labels)
kappa = cohen_kappa_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

# Show Results
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels).plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix - Transformer (w/ Fixes)")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(test_acc_list, label='Test Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Transformer Accuracy (with Positional Encoding + Scheduler)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_loss_list, label='Train Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.grid(True)
plt.legend()
plt.show()

print("\nClassification Report:\n", report)
print(f"Kappa Score: {kappa:.4f}")
