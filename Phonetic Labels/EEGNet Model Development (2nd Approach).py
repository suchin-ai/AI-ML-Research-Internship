import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset

# === User Inputs ===
dataset_path = r"B:\Education\CNN\EEG\Max's File\Processed_Phonetic_Numpy"
num_folds = 5
batch_size = 32
epochs = 20
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Load EEG Data ===
X = np.load(os.path.join(dataset_path, "X.npy"))
Y = np.load(os.path.join(dataset_path, "Y.npy"))

print(f"✅ Loaded EEG Data: {X.shape}, Labels: {Y.shape}")

# === Phonetic Label Mapping ===
phonetic_mapping = {
    0: 'il', 1: 'to', 2: 'te', 3: 'Li', 4: 'al', 5: 'be', 6: 'ri',
    7: 'ti', 8: 'e', 9: 've', 10: 'le', 11: 'un', 12: 'no', 13: 'a',
    14: 'kon', 15: 'u', 16: 'na', 17: 'ra', 18: 'la', 19: 'i',
    20: 'ka', 21: 'ni', 22: 'ma', 23: 'pa', 24: 'do', 25: 'ta', 
    26: 'ce', 27: 'ko', 28: 'si', 29: 'so', 30: 'li', 31: 'mi', 
    32: 'men', 33: 'ne', 34: 'po', 35: 'di', 36: 'ci'
}

num_classes = len(np.unique(Y))

# === EEGNet Model ===
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

# === Data Augmentation Functions ===
def gaussian_noise(data, std=0.005):
    return data + np.random.normal(0, std, data.shape)

def time_shift(data, shift_max=50):
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(data, shift, axis=-1)

def amplitude_scaling(data, scale_range=(0.9, 1.1)):
    scale_factor = np.random.uniform(*scale_range)
    return data * scale_factor

# === Training Function ===
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), 100 * correct / total

# === Testing Function ===
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), 100 * correct / total, y_true, y_pred

# === Cross-Validation Training ===
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
fold_accuracies = []

fold = 1
for train_idx, test_idx in skf.split(X, Y):
    print(f"\n====== Fold {fold}/{num_folds} ======")

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    # Data Augmentation ONLY on Training Set
    X_train_aug, Y_train_aug = [], []
    for x, y in zip(X_train, Y_train):
        X_train_aug.append(x)
        Y_train_aug.append(y)
        X_train_aug.append(gaussian_noise(x))
        Y_train_aug.append(y)
        X_train_aug.append(time_shift(x))
        Y_train_aug.append(y)
        X_train_aug.append(amplitude_scaling(x))
        Y_train_aug.append(y)

    X_train_aug = np.array(X_train_aug)
    Y_train_aug = np.array(Y_train_aug)

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_aug, dtype=torch.float32).unsqueeze(1),
                      torch.tensor(Y_train_aug, dtype=torch.long)),
        batch_size=batch_size, shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32).unsqueeze(1),
                      torch.tensor(Y_test, dtype=torch.long)),
        batch_size=batch_size, shuffle=False
    )

    # Initialize Model
    model = EEGNet(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train & Evaluate
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch+1}/{epochs}: Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion)
    fold_accuracies.append(test_acc)
    print(f"Fold {fold} Test Accuracy: {test_acc:.2f}%")

    fold += 1

# Average Accuracy
print("\n=== Cross-validation Results ===")
print(f"Average Test Accuracy: {np.mean(fold_accuracies):.2f}% ± {np.std(fold_accuracies):.2f}%")

# Confusion Matrix of last fold (for illustration)
labels_text = [phonetic_mapping[i] for i in range(num_classes)]
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_text, yticklabels=labels_text)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Fold 5)')
plt.show()

# Classification Report
print("\nClassification Report (Last Fold):")
print(classification_report(y_true, y_pred, target_names=labels_text))
