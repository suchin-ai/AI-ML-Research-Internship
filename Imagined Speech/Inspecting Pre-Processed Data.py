import h5py
import numpy as np

# Load the preprocessed file
file_path = r'B:\Education\CNN\EEG\Imagined Speech\Processing\Preprocessed_Epochs.h5'

with h5py.File(file_path, 'r') as h5_file:
    # Access the datasets
    epochs = np.array(h5_file['epochs'])
    labels = np.array(h5_file['labels']).astype(str)
    
    # Access metadata (attributes)
    num_epochs = h5_file.attrs['num_epochs']
    epoch_shape = h5_file.attrs['epoch_shape']
    channels = h5_file.attrs['channels']
    sampling_rate = h5_file.attrs['sampling_rate']
    epoch_duration = h5_file.attrs['epoch_duration']
    
    print(f"Number of Epochs: {num_epochs}")
    print(f"Shape of a Single Epoch: {epoch_shape}")
    print(f"Channels: {channels}")
    print(f"Sampling Rate: {sampling_rate}")
    print(f"Epoch Duration: {epoch_duration} seconds")
