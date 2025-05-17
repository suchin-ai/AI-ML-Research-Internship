# === Import Libraries ===
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
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

# === EEGNet Model Class ===
class EEGNet(nn.Module):
    def __init__(self, num_classes=4, Chans=22, Samples=501):
        super(EEGNet, self).__init__()
        
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8)
        )
        
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(Chans, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.25)
        )
        
        self.separableConv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(0.25)
        )
        
        self.classify = nn.Linear(16 * ((Samples // 32)), num_classes)

    def forward(self, x):
        # x: (batch_size, channels, samples)
        x = x.unsqueeze(1)  # (batch_size, 1, channels, samples)
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classify(x)
        return x

# === Main Script ===
if __name__ == "__main__":
    # === 1. Load EEG Data ===
    X = np.load(r'B:\Education\CNN\EEG\New\Dataset\processed\EEG_X_CSD.npy')
    y = np.load(r'B:\Education\CNN\EEG\New\Dataset\processed\EEG_y_CSD.npy')
    y = y - 1  # Shift labels to [0,1,2,3]
    print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")

    # === 2. Train-Test Split ===
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    # === 3. Create Dataset and DataLoader ===
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

    # === 4. Initialize Model, Loss, Optimizer ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = EEGNet(num_classes=4, Chans=22, Samples=501).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # === 5. Train the Model ===
    num_epochs = 100
    patience = 10  # Early stopping patience
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

        # === Validation Step ===
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
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        # === Early Stopping Check ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_eegnet_model.pth")  # Save best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # === 6. Load Best Model ===
    model.load_state_dict(torch.load("best_eegnet_model.pth"))

    # === 7. Final Evaluation on Validation Set ===
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n✅ Final Validation Accuracy (EEGNet): {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    # === 8. Plot Loss and Accuracy Graphs ===
    epochs_range = range(1, len(train_losses)+1)
    plt.figure(figsize=(14, 6))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.title('EEGNet: Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, [acc*100 for acc in val_accuracies], label='Validation Accuracy')
    plt.title('EEGNet: Validation Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # === 9. Confusion Matrix ===
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Left Hand", "Right Hand", "Feet", "Tongue"])
    plt.figure(figsize=(8,6))
    disp.plot(cmap='Blues', values_format='d')
    plt.title('EEGNet: Confusion Matrix')
    plt.grid(False)
    plt.show()
