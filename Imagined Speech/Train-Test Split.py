import h5py
import numpy as np
from sklearn.model_selection import train_test_split

# Load the preprocessed epochs file (without artifact removal)
file_path = r'B:\Education\CNN\EEG\Imagined Speech\Processing\Preprocessed_Epochs.h5'

with h5py.File(file_path, 'r') as h5_file:
    epochs = np.array(h5_file['epochs'])
    labels = np.array(h5_file['labels']).astype(str)

# Splitting the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    epochs, labels, test_size=0.2, random_state=42, stratify=labels
)

# Display the size of the splits
print(f"Training Set Size: {X_train.shape}")
print(f"Testing Set Size: {X_test.shape}")
print(f"Number of Channels: {X_train.shape[2]}")
print(f"Unique Labels in Training Set: {np.unique(y_train)}")
print(f"Unique Labels in Testing Set: {np.unique(y_test)}")

# Save the split data to a new .h5 file
save_path = r'B:\Education\CNN\EEG\Imagined Speech\Processing\Preprocessed_Split_Data.h5'

with h5py.File(save_path, 'w') as h5_file:
    h5_file.create_dataset('X_train', data=X_train)
    h5_file.create_dataset('X_test', data=X_test)
    h5_file.create_dataset('y_train', data=np.array(y_train, dtype='S50'))
    h5_file.create_dataset('y_test', data=np.array(y_test, dtype='S50'))
    
    # Save metadata for future reference
    h5_file.attrs['num_train_epochs'] = X_train.shape[0]
    h5_file.attrs['num_test_epochs'] = X_test.shape[0]
    h5_file.attrs['epoch_shape'] = X_train.shape[1:]
    h5_file.attrs['channels'] = 'Cz,F7,F8,T7,T8,F3,F4'
    h5_file.attrs['sampling_rate'] = 500
    h5_file.attrs['epoch_duration'] = 1.5  # seconds

print(f"Training and Testing sets successfully saved to {save_path}.")
