# === Import Libraries ===
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === EEG Dataset Class ===
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === LSTM Model Class ===
class EEGLSTM(nn.Module):
    def __init__(self, input_size=22, hidden_size=128, num_layers=2, num_classes=4):
        super(EEGLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, timepoints, channels) for LSTM
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.relu(self.fc1(out))
        return self.fc2(out)

# === Train One Fold Function ===
def train_one_fold(model, train_loader, val_loader, device, fold_idx):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    patience = 10
    best_val_loss = np.inf
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_accuracies = []

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
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
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

        print(f"Fold {fold_idx+1} | Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc*100:.2f}%")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_preds, best_labels = all_preds, all_labels
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Fold {fold_idx+1} | Early stopping triggered.")
                break

    return train_losses, val_losses, val_accuracies, best_preds, best_labels

# === Main Script ===
if __name__ == "__main__":
    # === 1. Load EEG Data ===
    X = np.load(r'B:\Education\CNN\EEG\New\Dataset\processed\EEG_X_CSD.npy')
    y = np.load(r'B:\Education\CNN\EEG\New\Dataset\processed\EEG_y_CSD.npy')
    y = y - 1
    print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")

    dataset = EEGDataset(X, y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    fold_train_losses = []
    fold_val_losses = []
    fold_val_accuracies = []
    all_preds = []
    all_labels = []
    fold_accuracies = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n=== Fold {fold_idx+1} ===")

        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)

        train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

        model = EEGLSTM(input_size=22, hidden_size=128, num_layers=2, num_classes=4).to(device)

        train_losses, val_losses, val_accs, preds_fold, labels_fold = train_one_fold(model, train_loader, val_loader, device, fold_idx)

        fold_train_losses.append(train_losses)
        fold_val_losses.append(val_losses)
        fold_val_accuracies.append(val_accs)
        all_preds.append(preds_fold)
        all_labels.append(labels_fold)

        fold_accuracies.append(val_accs[-1])

    # === Final Combined Results ===
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"\n✅ Final 5-Fold CV Accuracy (LSTM): {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

    # === Plotting Combined Loss and Accuracy ===
    plt.figure(figsize=(14, 6))

    plt.subplot(1,2,1)
    for idx, losses in enumerate(fold_train_losses):
        plt.plot(range(1,len(losses)+1), losses, label=f'Fold {idx+1} Train')
    for idx, losses in enumerate(fold_val_losses):
        plt.plot(range(1,len(losses)+1), losses, linestyle='--', label=f'Fold {idx+1} Val')
    plt.title("Training and Validation Loss Across Folds (LSTM)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.subplot(1,2,2)
    for idx, accs in enumerate(fold_val_accuracies):
        plt.plot(range(1,len(accs)+1), [a*100 for a in accs], label=f'Fold {idx+1} Val Acc')
    plt.title("Validation Accuracy Across Folds (LSTM)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    # === Combined Confusion Matrix ===
    all_preds_flat = np.concatenate(all_preds)
    all_labels_flat = np.concatenate(all_labels)

    cm = confusion_matrix(all_labels_flat, all_preds_flat)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Left Hand", "Right Hand", "Feet", "Tongue"])
    plt.figure(figsize=(8,6))
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Combined Confusion Matrix (LSTM 5-Fold)')
    plt.grid(False)
    plt.show()
