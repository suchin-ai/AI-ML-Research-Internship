import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset

# ========== User Inputs ==========
dataset_path = r"B:\Education\CNN\EEG\Max's File\Processed_Phonetic_Numpy\Augmented_Balanced"
batch_size = 32
epochs = 20
learning_rate = 0.0005

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======= Load Augmented Training Data and Original Test Data =======
X_train = np.load(os.path.join(dataset_path, "X_train_augmented_balanced.npy"))
Y_train = np.load(os.path.join(dataset_path, "Y_train_augmented_balanced.npy"))
X_test = np.load(os.path.join(dataset_path, "X_test.npy"))
Y_test = np.load(os.path.join(dataset_path, "Y_test.npy"))

print(f"✅ Training Data Shape: {X_train.shape}, Labels: {Y_train.shape}")
print(f"✅ Test Data Shape: {X_test.shape}, Labels: {Y_test.shape}")

# Label Mapping (Numeric -> Phonetic)
phonetic_mapping = {
    0: 'il', 1: 'to', 2: 'te', 3: 'Li', 4: 'al', 5: 'be', 6: 'ri',
    7: 'ti', 8: 'e', 9: 've', 10: 'le', 11: 'un', 12: 'no', 13: 'a',
    14: 'kon', 15: 'u', 16: 'na', 17: 'ra', 18: 'la', 19: 'i',
    20: 'ka', 21: 'ni', 22: 'ma', 23: 'pa', 24: 'do', 25: 'ta', 
    26: 'ce', 27: 'ko', 28: 'si', 29: 'so', 30: 'li', 31: 'mi', 
    32: 'men', 33: 'ne', 34: 'po', 35: 'di', 36: 'ci'
}

labels_text = [phonetic_mapping[i] for i in range(len(phonetic_mapping))]

# Convert data for LSTM (Samples, Channels, Time) -> (Samples, Time, Channels)
X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
Y_train = torch.tensor(Y_train, dtype=torch.long)
Y_test = torch.tensor(Y_test, dtype=torch.long)

# Create Data Loaders
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False)

# ========== LSTM Model Definition ==========
class EEG_LSTM(nn.Module):
    def __init__(self, input_size=61, hidden_size=128, num_layers=2, num_classes=37):
        super(EEG_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.5, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # bidirectional x2

    def forward(self, x):
        h, _ = self.lstm(x)
        h_last = h[:, -1, :]  # use the last hidden state
        out = self.fc(h_last)
        return out

# Initialize Model, Loss, and Optimizer
num_classes = len(torch.unique(Y_train))
model = EEG_LSTM(input_size=X_train.shape[2], num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ========== Training and Evaluation Function ==========
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, epochs):
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, correct, total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100. * correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc)

        # Evaluation phase
        model.eval()
        test_loss, correct, total = 0, 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        test_acc = 100. * correct / total
        test_losses.append(test_loss / len(test_loader))
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_losses[-1]:.4f}, Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_losses[-1]:.4f}, Acc: {test_acc:.2f}%")

    # ===== Plot Confusion Matrix with phonetic labels =====
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
                xticklabels=labels_text, yticklabels=labels_text)
    plt.title("Confusion Matrix - LSTM Model")
    plt.xlabel("Predicted Phonetic Syllable")
    plt.ylabel("True Phonetic Syllable")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # ===== Classification Report =====
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels_text))

# Run Training and Evaluation
train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, epochs)
