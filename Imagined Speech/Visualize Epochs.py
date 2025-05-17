import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load the extracted epochs file
file_path = r'B:\Education\CNN\EEG\Imagined Speech\Processing\Extracted_Epochs.h5'

with h5py.File(file_path, 'r') as h5_file:
    epochs = np.array(h5_file['epochs'])
    labels = np.array(h5_file['labels']).astype(str)
    channels = ['Cz', 'F7', 'F8', 'T7', 'T8', 'F3', 'F4']

# Select a single epoch (for example, the first one)
selected_epoch = epochs[0]  # You can change the index to view other samples
selected_label = labels[0]

# Plot all 7 channels of the selected epoch
plt.figure(figsize=(15, 10))
time = np.arange(selected_epoch.shape[0]) / 500  # Convert samples to seconds (500 Hz sampling rate)

for ch in range(selected_epoch.shape[1]):
    plt.subplot(7, 1, ch + 1)
    plt.plot(time, selected_epoch[:, ch])
    plt.title(f"Channel: {channels[ch]}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

plt.suptitle(f"Epoch Visualization for Label: {selected_label}", fontsize=16)
plt.tight_layout()
plt.show()
