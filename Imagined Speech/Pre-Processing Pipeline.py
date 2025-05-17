import h5py
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# Load the extracted epochs file
file_path = r'B:\Education\CNN\EEG\Imagined Speech\Processing\Extracted_Epochs.h5'

with h5py.File(file_path, 'r') as h5_file:
    epochs = np.array(h5_file['epochs'])
    labels = np.array(h5_file['labels']).astype(str)
    channels = ['Cz', 'F7', 'F8', 'T7', 'T8', 'F3', 'F4']

# Parameters
sampling_rate = 500  # Hz
lowcut = 0.5  # Hz (lower bound for bandpass filter)
highcut = 40.0  # Hz (upper bound for bandpass filter)

# Bandpass Filter Function
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data

# Preprocessing Function (Artifact Removal Removed)
def preprocess_epochs(epochs):
    preprocessed_epochs = []

    for epoch in epochs:
        # 1. Re-referencing (Average Reference)
        epoch = epoch - np.mean(epoch, axis=1, keepdims=True)

        # 2. Filtering
        epoch = bandpass_filter(epoch, lowcut, highcut, sampling_rate)

        # 3. Baseline Correction (Subtracting the mean of the epoch)
        epoch = epoch - np.mean(epoch, axis=0)

        # 4. Normalization (Z-score normalization)
        epoch = (epoch - np.mean(epoch)) / (np.std(epoch) + 1e-6)
        
        # Store the clean epoch (No artifact removal step applied)
        preprocessed_epochs.append(epoch)
    
    return np.array(preprocessed_epochs)

# Apply preprocessing to all epochs
preprocessed_epochs = preprocess_epochs(epochs)
preprocessed_labels = labels  # No epochs are discarded, so labels remain the same

print(f"Total Preprocessed Epochs: {preprocessed_epochs.shape[0]}")
print(f"Epoch Shape (Single Epoch): {preprocessed_epochs.shape[1:]}")

# Save preprocessed data
save_path = r'B:\Education\CNN\EEG\Imagined Speech\Processing\Preprocessed_Epochs.h5'

with h5py.File(save_path, 'w') as h5_file:
    h5_file.create_dataset('epochs', data=preprocessed_epochs)
    h5_file.create_dataset('labels', data=np.array(preprocessed_labels, dtype='S50'))
    
    # Save metadata for future reference
    h5_file.attrs['num_epochs'] = preprocessed_epochs.shape[0]
    h5_file.attrs['epoch_shape'] = preprocessed_epochs.shape[1:]
    h5_file.attrs['channels'] = 'Cz,F7,F8,T7,T8,F3,F4'
    h5_file.attrs['sampling_rate'] = 500
    h5_file.attrs['epoch_duration'] = 1.5  # seconds

print(f"Preprocessed data successfully saved to {save_path}.")

# Load the preprocessed epochs file
file_path = r'B:\Education\CNN\EEG\Imagined Speech\Processing\Preprocessed_Epochs.h5'

with h5py.File(file_path, 'r') as h5_file:
    preprocessed_epochs = np.array(h5_file['epochs'])
    labels = np.array(h5_file['labels']).astype(str)
    channels = ['Cz', 'F7', 'F8', 'T7', 'T8', 'F3', 'F4']

# Select a single preprocessed epoch (for example, the first one)
selected_epoch = preprocessed_epochs[0]  # You can change the index to view other samples
selected_label = labels[0]

# Plot all 7 channels of the selected epoch
plt.figure(figsize=(15, 10))
time = np.arange(selected_epoch.shape[0]) / 500  # Convert samples to seconds (500 Hz sampling rate)

for ch in range(selected_epoch.shape[1]):
    plt.subplot(7, 1, ch + 1)
    plt.plot(time, selected_epoch[:, ch])
    plt.title(f"Channel: {channels[ch]}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (Normalized)")

plt.suptitle(f"Preprocessed Epoch Visualization for Label: {selected_label}", fontsize=16)
plt.tight_layout()
plt.show()