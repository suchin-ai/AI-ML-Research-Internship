import os
import numpy as np
import pandas as pd
from mat73 import loadmat

# ========== User Inputs ==========
dataset_path = r"B:\Education\CNN\EEG\Max's File\Dataset"  # Path to dataset
save_path = r"B:\Education\CNN\EEG\Max's File\Processed_Numpy"  # Where to save NumPy files
file_name = "processed_ica_clean.mat"  # Choose the .mat file to inspect
window_size = 5120  # EEG segment length (adjust as needed)

os.makedirs(save_path, exist_ok=True)  # Ensure save directory exists

# Full path to file
file_path = os.path.join(dataset_path, file_name)

# ========== Load EEG Dataset ==========
print(f"\n🔹 Loading {file_name}...")
EEG = loadmat(file_path)['EEG']

# Extract EEG Data
eeg_data = np.array(EEG['data'])  # Shape: (Channels, Time)

# Extract Events
events = pd.DataFrame([{k: v[0] for k, v in event.items()} for event in EEG['event_gdf']])
events = events.dropna(subset=['TYP', 'POS'])  # Remove invalid events
events["TYP"] = events["TYP"].astype(int)
events["POS"] = events["POS"].astype(int)

# Filter only phonetic syllables (TYP == 781)
phonetic_events = events[events["TYP"] == 781]

# Extract Labels and EEG Segments
X = []
Y = []

for _, event in phonetic_events.iterrows():
    label = event['TYP']  # Extract phonetic label
    pos = event['POS']  # Event start time

    # Ensure valid EEG segment
    if pos + window_size <= eeg_data.shape[1]:
        segment = eeg_data[:, pos:pos + window_size]  # Extract EEG segment
        X.append(segment)
        Y.append(label)

# Convert to NumPy Arrays
X = np.array(X)  # EEG Data: (Samples, Channels, Time)
Y = np.array(Y)  # Labels: (Samples,)

# Save Processed Data
X_save_path = os.path.join(save_path, f"X_{file_name.replace('.mat', '.npy')}")
Y_save_path = os.path.join(save_path, f"Y_{file_name.replace('.mat', '.npy')}")

np.save(X_save_path, X)
np.save(Y_save_path, Y)

print(f"\n✅ Saved EEG Data: {X.shape} to {X_save_path}")
print(f"✅ Saved Labels: {Y.shape} to {Y_save_path}")
