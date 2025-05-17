import h5py
import numpy as np

# Load the split preprocessed data
file_path = r'B:\Education\CNN\EEG\Imagined Speech\Processing\Preprocessed_Split_Data.h5'

with h5py.File(file_path, 'r') as h5_file:
    X_train = np.array(h5_file['X_train'])
    X_test = np.array(h5_file['X_test'])
    y_train = np.array(h5_file['y_train']).astype(str)
    y_test = np.array(h5_file['y_test']).astype(str)

# Create label-to-index mapping
unique_labels = np.unique(y_train)
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}  # Optional: for decoding predictions

# Encode string labels to integers
y_train_encoded = np.array([label_to_index[label] for label in y_train])
y_test_encoded = np.array([label_to_index[label] for label in y_test])

# Show results
print("Label Encoding Mapping:")
for label, idx in label_to_index.items():
    print(f"{label} -> {idx}")

print("\nSample Encoded Labels (train):", y_train_encoded[:10])
print("Sample Encoded Labels (test):", y_test_encoded[:10])
