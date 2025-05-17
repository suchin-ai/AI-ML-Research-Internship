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
