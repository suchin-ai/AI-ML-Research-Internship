import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

# ==========================
# Load All Data From HDF5 File
# ==========================
data_path = r'B:\Education\CNN\EEG\Imagined Speech\Processing\Preprocessed_Data.h5'

X_all = []
Y_all = []

with h5py.File(data_path, 'r') as f:
    for key in f.keys():
        X_all.append(f[key]['X'][:])
        Y_all.append(f[key]['Y'][:])

X_all = np.concatenate(X_all, axis=0)
Y_all = np.concatenate(Y_all, axis=0)

print(f"Combined Data Shape (X_all): {X_all.shape}")
print(f"Combined Labels Shape (Y_all): {Y_all.shape}")

# Display the label distribution
label_counts = Counter(Y_all)
print(f"\nLabel Distribution: {label_counts}")

# ==========================
# Convert Labels to Integers (Label Encoding)
# ==========================
label_encoder = LabelEncoder()
Y_all_encoded = label_encoder.fit_transform(Y_all)

# Display the label mapping
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(f"\nLabel Mapping: {label_mapping}")

# ==========================
# Train-Test Split
# ==========================
X_train, X_test, Y_train, Y_test = train_test_split(
    X_all, Y_all_encoded, test_size=0.2, stratify=Y_all_encoded, random_state=42
)

print(f"\nTraining Data Shape (X_train): {X_train.shape}")
print(f"Training Labels Shape (Y_train): {Y_train.shape}")
print(f"Testing Data Shape (X_test): {X_test.shape}")
print(f"Testing Labels Shape (Y_test): {Y_test.shape}")

# Displaying label distribution in training and testing sets
train_label_counts = Counter(Y_train)
test_label_counts = Counter(Y_test)
print(f"\nTraining Label Distribution: {train_label_counts}")
print(f"Testing Label Distribution: {test_label_counts}")

# ==========================
# Convert to PyTorch Tensors
# ==========================
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

# Prepare DataLoaders
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, Y_test_tensor), batch_size=batch_size, shuffle=False)

print(f"\nNumber of Training Batches per Epoch: {len(train_loader)}")
print(f"Number of Testing Batches: {len(test_loader)}")

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

# ==========================
# Define EEGNet Model
# ==========================
class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 5), stride=1, padding=(0, 2))  # Output: (32, 7, 750)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))  # Output: (32, 7, 375)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 5), stride=1, padding=(0, 2))  # Output: (64, 7, 375)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))  # Output: (64, 7, 187)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 5), stride=1, padding=(0, 2))  # Output: (128, 7, 187)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))  # Output: (128, 7, 93)
        
        self.dropout = nn.Dropout(0.5)
        
        self.flattened_size = self._get_flattened_size()
        self.fc1 = nn.Linear(self.flattened_size, num_classes)
        
    def _get_flattened_size(self):
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 7, 750)  # (Batch size, Channels, EEG Channels, Time Samples)
            x = self.conv1(dummy_input)
            x = self.batchnorm1(x)
            x = torch.relu(x)
            x = self.pool1(x)
            
            x = self.conv2(x)
            x = self.batchnorm2(x)
            x = torch.relu(x)
            x = self.pool2(x)
            
            x = self.conv3(x)
            x = self.batchnorm3(x)
            x = torch.relu(x)
            x = self.pool3(x)
            
            x = x.view(x.size(0), -1)
            return x.shape[1]

    def forward(self, x):
        x = x.unsqueeze(1)  # Adding channel dimension: (batch_size, 1, 7, 750)
        
        x = torch.relu(self.batchnorm1(self.conv1(x)))
        x = self.pool1(x)
        
        x = torch.relu(self.batchnorm2(self.conv2(x)))
        x = self.pool2(x)
        
        x = torch.relu(self.batchnorm3(self.conv3(x)))
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x
    
# ==========================
# Training and Evaluation
# ==========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_acc_list = []
test_acc_list = []
num_epochs = 100
best_test_acc = 0
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    correct, total = 0, 0
    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total += Y_batch.size(0)
        correct += (predicted == Y_batch).sum().item()
    
    train_acc = 100 * correct / total
    train_acc_list.append(train_acc)
    
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += Y_batch.size(0)
            correct += (predicted == Y_batch).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(Y_batch.cpu().numpy())
    
    test_acc = 100 * correct / total
    test_acc_list.append(test_acc)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%')
    
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early Stopping Triggered at Epoch {epoch+1}")
        break

# ==========================
# Plotting Accuracy Graph
# ==========================
plt.figure(figsize=(10, 5))
plt.plot(train_acc_list, label='Training Accuracy')
plt.plot(test_acc_list, label='Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training & Testing Accuracy per Epoch')
plt.legend()
plt.show()

# ==========================
# Generate Confusion Matrix and Report
# ==========================
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

report = classification_report(all_labels, all_preds)
kappa = cohen_kappa_score(all_labels, all_preds)
print(report)
print(f"Cohen's Kappa Score: {kappa:.4f}")
