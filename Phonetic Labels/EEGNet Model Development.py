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

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
Y_train = torch.tensor(Y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
Y_test = torch.tensor(Y_test, dtype=torch.long)

# Create Data Loaders
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False)

# EEGNet Model Definition
class EEGNet(nn.Module):
    def __init__(self, num_classes, Chans=61, Samples=5120):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.depthwiseConv = nn.Conv2d(16, 32, (Chans, 1), groups=16, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(0.5)

        self.separableConv = nn.Conv2d(32, 32, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(0.5)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * (Samples // 32), num_classes)
        self.elu = nn.ELU()

    def forward(self, x):
        x = self.elu(self.bn1(self.conv1(x)))
        x = self.elu(self.bn2(self.depthwiseConv(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.elu(self.bn3(self.separableConv(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.fc(x)
        return x

# Initialize model, loss, and optimizer
num_classes = len(torch.unique(Y_train))
model = EEGNet(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training and evaluation
def train_and_evaluate(train_loader, test_loader, model, criterion, optimizer, epochs):
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_losses.append(train_loss / len(train_loader))
        train_acc = 100 * correct / total
        train_accs.append(train_acc)

        # Evaluation
        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        test_acc = 100 * np.mean(np.array(y_true) == np.array(y_pred))
        test_losses.append(test_loss / len(test_loader))
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_acc:.2f}%")

    # Plot Confusion Matrix with phonetic labels
    labels_text = [phonetic_mapping[i] for i in range(num_classes)]
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
                xticklabels=labels_text, yticklabels=labels_text)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Phonetic Syllable")
    plt.ylabel("True Phonetic Syllable")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels_text))

    # Plot Accuracy Curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label="Train Accuracy", marker='o')
    plt.plot(test_accs, label="Test Accuracy", marker='s')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# Run training and evaluation
train_and_evaluate(train_loader, test_loader, model, criterion, optimizer, epochs)
