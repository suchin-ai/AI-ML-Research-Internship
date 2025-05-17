# === Imports ===
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === Augmentation Functions (no random flip now) ===
def add_gaussian_noise(signal, mean=0.0, std=0.01):
    noise = np.random.normal(mean, std, signal.shape)
    return signal + noise

def scale_amplitude(signal, min_scale=0.9, max_scale=1.1):
    factor = np.random.uniform(min_scale, max_scale)
    return signal * factor

def time_shift(signal, shift_max=20):
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(signal, shift, axis=-1)

def augment_signal(signal):
    signal = add_gaussian_noise(signal)
    signal = scale_amplitude(signal)
    signal = time_shift(signal)
    return signal

# === EEG Dataset Class ===
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
            x = augment_signal(x)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# === 1D CNN Model ===
class EEG1DCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(EEG1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(22, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(256 * 62, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# === LSTM Model ===
class EEGLSTM(nn.Module):
    def __init__(self, input_size=22, hidden_size=128, num_layers=2, num_classes=4):
        super(EEGLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        return self.fc2(out)

# === EEGNet Model ===
class EEGNet(nn.Module):
    def __init__(self, num_classes=4, Chans=22, Samples=501):
        super(EEGNet, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8)
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(Chans, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.25)
        )
        self.separable = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(0.25)
        )
        self.classify = nn.Linear(16 * (Samples // 32), num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = x.view(x.size(0), -1)
        return self.classify(x)

# === Training One Fold ===
def train_one_fold(model, train_loader, val_loader, device, fold_idx, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 100
    patience = 10
    best_val_loss = np.inf
    patience_counter = 0
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        val_acc = accuracy_score(all_labels, all_preds)
        val_accuracies.append(val_acc)

        print(f"{model_name} | Fold {fold_idx+1} | Epoch {epoch+1}: Train Loss {train_losses[-1]:.4f}, Val Loss {avg_val_loss:.4f}, Val Acc {val_acc*100:.2f}%")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_preds, best_labels = all_preds, all_labels
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"{model_name} | Fold {fold_idx+1} | Early stopping triggered")
                break

    return train_losses, val_losses, val_accuracies, best_preds, best_labels

# === Main Cross-Validation Script ===
if __name__ == "__main__":
    # === Load EEG Data ===
    X = np.load(r'B:\Education\CNN\EEG\New\Dataset\processed\EEG_X_CSD.npy')
    y = np.load(r'B:\Education\CNN\EEG\New\Dataset\processed\EEG_y_CSD.npy')
    y = y - 1
    print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for model_name, model_cls in {
        "1D_CNN": EEG1DCNN,
        "LSTM": EEGLSTM,
        "EEGNet": EEGNet
    }.items():
        print(f"\n==== Training {model_name} with 5-Fold CV ====")
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_train_losses = []
        fold_val_losses = []
        fold_val_accuracies = []
        all_preds = []
        all_labels = []
        fold_accuracies = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_dataset = EEGDataset(X_train, y_train, augment=True)
            val_dataset = EEGDataset(X_val, y_val, augment=False)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

            model = model_cls().to(device)
            train_losses, val_losses, val_accs, preds_fold, labels_fold = train_one_fold(model, train_loader, val_loader, device, fold_idx, model_name)

            fold_train_losses.append(train_losses)
            fold_val_losses.append(val_losses)
            fold_val_accuracies.append(val_accs)
            all_preds.append(preds_fold)
            all_labels.append(labels_fold)
            fold_accuracies.append(val_accs[-1])

        # === Plot Results ===
        plt.figure(figsize=(14, 6))
        plt.subplot(1,2,1)
        for losses in fold_train_losses:
            plt.plot(range(1, len(losses)+1), losses, label="Train Loss")
        for losses in fold_val_losses:
            plt.plot(range(1, len(losses)+1), losses, linestyle='--', label="Val Loss")
        plt.title(f"{model_name}: Loss Across Folds")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(); plt.legend()

        plt.subplot(1,2,2)
        for accs in fold_val_accuracies:
            plt.plot(range(1, len(accs)+1), [a*100 for a in accs], label="Val Accuracy")
        plt.title(f"{model_name}: Accuracy Across Folds")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.grid(); plt.legend()

        plt.tight_layout()
        plt.show()

        # === Combined Confusion Matrix ===
        all_preds_flat = np.concatenate(all_preds)
        all_labels_flat = np.concatenate(all_labels)
        cm = confusion_matrix(all_labels_flat, all_preds_flat)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Left Hand", "Right Hand", "Feet", "Tongue"])
        plt.figure(figsize=(8,6))
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f'{model_name}: Combined Confusion Matrix (5-Fold)')
        plt.grid(False)
        plt.show()

        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)
        print(f"\n✅ {model_name} Final 5-Fold CV Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%\n")
