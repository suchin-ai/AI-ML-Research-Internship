import h5py
import numpy as np
import os

# Directory containing all the .h5 files
directory_path = r'B:\Education\CNN\EEG\Imagined Speech\Dataset'
files = [f for f in os.listdir(directory_path) if f.endswith('.h5')]

# Sampling rate and epoch duration
srate = 500  # Hz
epoch_duration = 1.5  # seconds
samples_per_epoch = int(srate * epoch_duration)  # 750 samples

# To store extracted epochs and labels
all_epochs = []
all_labels = []

for file_name in files:
    file_path = os.path.join(directory_path, file_name)
    print(f"Processing file: {file_name}")

    with h5py.File(file_path, 'r') as h5_file:
        # Load continuous EEG data and events
        continuous_eeg = np.array(h5_file['continuous_eeg'])
        events = h5_file['events']
        
        # Extracting all events, excluding "pause"
        event_samples = np.array([int(e['sample'][0]) if isinstance(e['sample'], np.ndarray) else int(e['sample']) for e in events])
        event_classes = np.array([e['class'].decode('utf-8') for e in events])

        for i, (start_index, label) in enumerate(zip(event_samples, event_classes)):
            # Skip the "pause" label
            if label.lower() == "pause":
                continue
            
            # Extracting epoch for the current event
            end_index = start_index + samples_per_epoch
            if end_index <= continuous_eeg.shape[0]:
                epoch = continuous_eeg[start_index:end_index, :]
                all_epochs.append(epoch)
                all_labels.append(label)

    print(f"Extracted {len(all_epochs)} epochs from {file_name}\n")

# Convert to numpy arrays for further processing
all_epochs = np.array(all_epochs)
all_labels = np.array(all_labels)

print(f"Total Epochs Extracted (excluding 'pause'): {all_epochs.shape[0]}")
print(f"Epoch Shape (Single Epoch): {all_epochs.shape[1:]}")

# Save path for the preprocessed data
save_path = r'B:\Education\CNN\EEG\Imagined Speech\Processing\Extracted_Epochs_No_Pause.h5'

# Convert labels to numpy array (if not already done)
all_labels = np.array(all_labels, dtype='S50')  # Converting labels to byte strings for storage

# Saving data to .h5 file
with h5py.File(save_path, 'w') as h5_file:
    # Save the epochs and labels
    h5_file.create_dataset('epochs', data=all_epochs)
    h5_file.create_dataset('labels', data=all_labels)
    
    # Save metadata for future reference
    h5_file.attrs['num_epochs'] = all_epochs.shape[0]
    h5_file.attrs['epoch_shape'] = all_epochs.shape[1:]
    h5_file.attrs['channels'] = 'Cz,F7,F8,T7,T8,F3,F4'
    h5_file.attrs['sampling_rate'] = 500
    h5_file.attrs['epoch_duration'] = 1.5  # seconds
    h5_file.attrs['experiment_date'] = '2025-03-20'
    h5_file.attrs['subject_id'] = 'S6-phono-1'

print(f"Epochs and labels (excluding 'pause') successfully saved to {save_path}.")
