import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os

# ------------------ Settings ------------------

data_path = r"B:\Education\CNN\EEG\EEGdenoiseNet\Data"
save_model_path = r"B:\Education\CNN\EEG\EEGdenoiseNet\Models"
os.makedirs(save_model_path, exist_ok=True)

noise_types = ['EMG', 'EOG', 'BOTH']
snr_values = list(range(-7, 3))  # [-7, ..., 2]

batch_size = 32
epochs = 100
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Dataset Class ------------------

class EEGDataset(Dataset):
    def __init__(self, noisy_path, clean_path):
        self.X = np.load(noisy_path)
        self.Y = np.load(clean_path)

        self.X = (self.X - np.mean(self.X, axis=1, keepdims=True)) / (np.std(self.X, axis=1, keepdims=True) + 1e-6)
        self.Y = (self.Y - np.mean(self.Y, axis=1, keepdims=True)) / (np.std(self.Y, axis=1, keepdims=True) + 1e-6)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.Y[idx], dtype=torch.float32).unsqueeze(0)
        return x, y

# ------------------ CNN Model ------------------

class CNNDenoiser(nn.Module):
    def __init__(self):
        super(CNNDenoiser, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=7, padding=3),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ------------------ Metrics ------------------

def compute_rrmse(true, pred):
    return np.sqrt(np.mean((true - pred) ** 2)) / (np.sqrt(np.mean(true ** 2)) + 1e-8)

def compute_acc(true, pred):
    corr_list = []
    for i in range(true.shape[0]):
        corr = np.corrcoef(true[i], pred[i])[0, 1]
        corr_list.append(corr)
    return np.mean(corr_list)

# ------------------ Results Dictionary ------------------

results = {}

# ------------------ Full Training and Evaluation Loop ------------------

for noise_type in noise_types:
    results[noise_type] = {'rrmse': [], 'acc': [], 'r2': []}
    for snr in snr_values:
        print(f"\n--- Training on {noise_type} noise at {snr} dB ---")

        X_path = os.path.join(data_path, f"Noisy_EEG_{noise_type}_{snr}dB.npy")
        Y_path = os.path.join(data_path, "EEG_clean.npy")
        
        if not os.path.exists(X_path):
            print(f"File {X_path} not found. Skipping...")
            results[noise_type]['rrmse'].append(None)
            results[noise_type]['acc'].append(None)
            results[noise_type]['r2'].append(None)
            continue
        
        dataset = EEGDataset(X_path, Y_path)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = CNNDenoiser().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Train
        for epoch in range(epochs):
            model.train()
            running_loss = 0
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(Y_batch.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0).squeeze()
        all_targets = np.concatenate(all_targets, axis=0).squeeze()

        rrmse = compute_rrmse(all_targets, all_preds)
        acc = compute_acc(all_targets, all_preds)
        r2 = r2_score(all_targets.flatten(), all_preds.flatten())

        results[noise_type]['rrmse'].append(rrmse)
        results[noise_type]['acc'].append(acc)
        results[noise_type]['r2'].append(r2)

        print(f"RRMSE: {rrmse:.6f}, ACC: {acc:.6f}, R2: {r2:.6f}")

# ------------------ Plot Results ------------------

metrics = ['rrmse', 'acc']

for metric in metrics:
    plt.figure(figsize=(10,6))
    for noise_type in noise_types:
        values = results[noise_type][metric]
        plt.plot(snr_values, values, label=f"{noise_type}", marker='o')
    plt.title(f"{metric.upper()} across SNR levels")
    plt.xlabel('SNR (dB)')
    plt.ylabel(metric.upper())
    plt.legend()
    plt.grid(True)
    plt.show()

# ------------------ Print Final Table ------------------

print("\n📋 Final Evaluation Table:")
for noise_type in noise_types:
    print(f"\n=== {noise_type} Noise ===")
    print(f"SNR (dB): {snr_values}")
    print(f"RRMSE:    {[f'{x:.4f}' if x is not None else '---' for x in results[noise_type]['rrmse']]}")
    print(f"ACC:      {[f'{x:.4f}' if x is not None else '---' for x in results[noise_type]['acc']]}")
    print(f"R2:       {[f'{x:.4f}' if x is not None else '---' for x in results[noise_type]['r2']]}")
