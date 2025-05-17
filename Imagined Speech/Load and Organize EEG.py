import h5py
import numpy as np
import os

# Path to dataset
dataset_path = r"B:\Education\CNN\EEG\Imagined Speech\Dataset"

# Store data for each run
eeg_runs = []
run_labels = []

# Sampling rate and epoch duration
srate = 500  # Hz
epoch_duration = 1.5  # seconds
samples_per_epoch = int(srate * epoch_duration)  # 750

# Process each H5 file
for filename in sorted(os.listdir(dataset_path)):
    if filename.endswith(".h5"):
        file_path = os.path.join(dataset_path, filename)
        print(f"Loading {filename}...")

        with h5py.File(file_path, 'r') as h5_file:
            eeg = np.array(h5_file['continuous_eeg'])  # (n_samples, 7)
            events = h5_file['events']

            sample_indices = [
                int(e['sample'][0]) if isinstance(e['sample'], np.ndarray) else int(e['sample'])
                for e in events
            ]
            class_labels = [e['class'].decode('utf-8') for e in events]

            epochs = []
            labels = []

            for idx, label in zip(sample_indices, class_labels):
                if label.lower() != 'pause':
                    end_idx = idx + samples_per_epoch
                    if end_idx <= eeg.shape[0]:
                        epoch = eeg[idx:end_idx]  # Shape: (750, 7)
                        epochs.append(epoch)
                        labels.append(label)

            eeg_runs.append(np.array(epochs))      # shape: (n_epochs, 750, 7)
            run_labels.append(np.array(labels))    # shape: (n_epochs,)

# Optional: Print shape summary
for i, (run, labels) in enumerate(zip(eeg_runs, run_labels)):
    print(f"Run {i+1}: {run.shape[0]} epochs, Shape: {run.shape}")

import mne
from mne.preprocessing import compute_current_source_density

# Your channel names from the dataset
channel_names = ['Cz', 'F7', 'F8', 'T7', 'T8', 'F3', 'F4']
channel_type = ['eeg'] * 7

csd_runs = []

for i, run_data in enumerate(eeg_runs):  # run_data shape: (n_epochs, 750, 7)
    print(f"Applying CSD on Run {i+1}...")

    csd_epochs = []

    for epoch in run_data:  # Each epoch shape: (750, 7)
        # Create MNE RawArray for this single epoch
        info = mne.create_info(ch_names=channel_names, sfreq=500, ch_types=channel_type)
        raw = mne.io.RawArray(epoch.T, info)

        # Apply CSD
        raw_csd = compute_current_source_density(raw)

        # Transpose back to (750, 7) and store
        csd_epoch = raw_csd.get_data().T
        csd_epochs.append(csd_epoch)

    csd_runs.append(np.array(csd_epochs))  # shape: (n_epochs, 750, 7)

print("✅ CSD filtering completed.")
