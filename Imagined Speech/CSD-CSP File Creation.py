import h5py
import numpy as np
import mne
from mne.preprocessing import compute_current_source_density
from mne.decoding import CSP

# === File Paths ===
input_path = r'B:\Education\CNN\EEG\Imagined Speech\Processing\Preprocessed_Epochs.h5'
csd_output_path = r'B:\Education\CNN\EEG\Imagined Speech\Processing\CSD_Transformed_Epochs.h5'
csd_csp_output_path = r'B:\Education\CNN\EEG\Imagined Speech\Processing\CSD_CSP_Transformed_Epochs.h5'

# === Parameters ===
sfreq = 500  # Sampling rate in Hz
ch_names = ['Cz', 'F7', 'F8', 'T7', 'T8', 'F3', 'F4']
ch_types = ['eeg'] * len(ch_names)

# === Step 1: Load Preprocessed Data ===
with h5py.File(input_path, 'r') as f:
    data = f['epochs'][:]   # shape: (700, 750, 7)
    labels = f['labels'][:] # shape: (700,)

# Transpose for MNE format: (n_epochs, n_channels, n_times)
data_mne = np.transpose(data, (0, 2, 1))

# Create MNE Info & Epochs
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
epochs = mne.EpochsArray(data_mne, info)

# === Step 2: Set Montage with Realistic Positions ===
ch_positions = {
    'Cz': [0.0, 0.0, 0.09],
    'F7': [-0.05, 0.07, 0.02],
    'F8': [0.05, 0.07, 0.02],
    'T7': [-0.07, 0.0, 0.01],
    'T8': [0.07, 0.0, 0.01],
    'F3': [-0.03, 0.06, 0.06],
    'F4': [0.03, 0.06, 0.06],
}
montage = mne.channels.make_dig_montage(ch_pos=ch_positions, coord_frame='head')
epochs.set_montage(montage)

# === Step 3: Apply CSD ===
csd_epochs = compute_current_source_density(epochs)

# Save CSD-transformed data (back to shape: n_epochs, n_times, n_channels)
csd_data = np.transpose(csd_epochs.get_data(), (0, 2, 1))
with h5py.File(csd_output_path, 'w') as f:
    f.create_dataset('epochs', data=csd_data)
    f.create_dataset('labels', data=labels)
print("✅ Saved CSD-transformed epochs to:", csd_output_path)

# === Step 4: Apply CSP ===
csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)
X_csp = csp.fit_transform(csd_epochs.get_data(), labels)  # shape: (n_epochs, n_components)

# Save CSP-transformed features
with h5py.File(csd_csp_output_path, 'w') as f:
    f.create_dataset('features', data=X_csp)
    f.create_dataset('labels', data=labels)
print("✅ Saved CSD+CSP-transformed features to:", csd_csp_output_path)
