import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np

# File path of the preprocessed split data
file_path = r'B:\Education\CNN\EEG\Imagined Speech\Processing\Preprocessed_Split_Data.h5'

# Load the data from the .h5 file
with h5py.File(file_path, 'r') as h5_file:
    X_train = np.array(h5_file['X_train'])
    X_test = np.array(h5_file['X_test'])
    y_train = np.array(h5_file['y_train']).astype(str)
    y_test = np.array(h5_file['y_test']).astype(str)
    
    # Extracting metadata (optional but useful)
    num_train_epochs = h5_file.attrs['num_train_epochs']
    num_test_epochs = h5_file.attrs['num_test_epochs']
    channels = h5_file.attrs['channels'].split(',')
    sampling_rate = h5_file.attrs['sampling_rate']
    epoch_duration = h5_file.attrs['epoch_duration']

print(f"Training Set Size: {X_train.shape}")
print(f"Testing Set Size: {X_test.shape}")
print(f"Number of Channels: {len(channels)}")

# Display unique labels to confirm
print(f"Unique Labels in Training Set: {np.unique(y_train)}")
print(f"Unique Labels in Testing Set: {np.unique(y_test)}")

# Convert labels to indices
unique_labels = np.unique(y_train)
label_to_index = {label: index for index, label in enumerate(unique_labels)}
index_to_label = {index: label for label, index in label_to_index.items()}

y_train_indices = np.array([label_to_index[label] for label in y_train])
y_test_indices = np.array([label_to_index[label] for label in y_test])

# Normalizing data to have zero mean and unit variance (Z-score normalization)
X_train = (X_train - np.mean(X_train, axis=(0, 1))) / (np.std(X_train, axis=(0, 1)) + 1e-6)
X_test = (X_test - np.mean(X_test, axis=(0, 1))) / (np.std(X_test, axis=(0, 1)) + 1e-6)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
y_train_tensor = torch.tensor(y_train_indices, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
y_test_tensor = torch.tensor(y_test_indices, dtype=torch.long)

# Create DataLoader objects
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

# Define the 1D CNN model
class EEG_1D_CNN(nn.Module):
    def __init__(self, num_channels=7, num_classes=10):
        super(EEG_1D_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (750 // 8), 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Instantiate model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EEG_1D_CNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Training and evaluation functions
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)


def evaluate(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return test_loss / len(test_loader), accuracy

# Training loop with early stopping
num_epochs = 50
early_stopping = EarlyStopping(patience=5, min_delta=0.001)

train_losses = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    # Check for early stopping
    early_stopping(test_loss)
    if early_stopping.early_stop:
        print(f"\nEarly stopping triggered at epoch {epoch+1}.")
        break