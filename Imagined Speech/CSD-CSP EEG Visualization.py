import h5py
import matplotlib.pyplot as plt
import numpy as np

# Load CSD-transformed data
with h5py.File(r'B:\Education\CNN\EEG\Imagined Speech\Processing\CSD_Transformed_Epochs.h5', 'r') as f:
    data = f['epochs'][:]   # shape: (700, 750, 7)
    labels = f['labels'][:]

# Plot a single epoch (e.g., index 0) for all channels
epoch_idx = 0
epoch = data[epoch_idx]  # shape: (750, 7)
time = np.arange(epoch.shape[0]) / 500  # time in seconds assuming 500 Hz

plt.figure(figsize=(12, 6))
for ch in range(epoch.shape[1]):
    plt.plot(time, epoch[:, ch] + ch*20, label=f'Ch{ch+1}')  # vertical offset

plt.title(f'CSD EEG - Epoch {epoch_idx} | Label: {labels[epoch_idx]}')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV) [offset per channel]")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

# Load CSP features
with h5py.File(r'B:\Education\CNN\EEG\Imagined Speech\Processing\CSD_CSP_Transformed_Epochs.h5', 'r') as f:
    features = f['features'][:]  # shape: (700, 6)
    labels = f['labels'][:]

plt.figure(figsize=(10, 6))
plt.boxplot(features, vert=True, patch_artist=True)
plt.title("CSP Component Feature Distribution")
plt.xlabel("CSP Component Index")
plt.ylabel("Log-Variance")
plt.grid(True)
plt.show()

from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

# Dimensionality reduction
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_proj = tsne.fit_transform(features)

# Plot
df = pd.DataFrame()
df['x'] = tsne_proj[:, 0]
df['y'] = tsne_proj[:, 1]
df['label'] = [label.decode() if isinstance(label, bytes) else str(label) for label in labels]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='x', y='y', hue='label', palette='tab10')
plt.title("t-SNE of CSP Features (CSD+CSP)")
plt.grid(True)
plt.show()
