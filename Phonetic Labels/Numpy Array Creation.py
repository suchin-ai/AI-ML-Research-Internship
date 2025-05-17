import os
import numpy as np
import pandas as pd
from mat73 import loadmat

# ========== User Inputs ==========
dataset_path = r"B:\Education\CNN\EEG\Max's File\Dataset"  # Path to dataset
save_path = r"B:\Education\CNN\EEG\Max's File\Processed_Phonetic_Numpy"  # Save location for NumPy files
file_name = "processed_merged.mat"  # Choose the .mat file to inspect

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
events = events.dropna(subset=['TYP', 'POS', 'PHONESYLL'])  # Remove missing values
events["TYP"] = events["TYP"].astype(int)
events["POS"] = events["POS"].astype(int)

# ✅ **Remove Empty Phonetic Labels ("" or NaN)**
events = events[events["PHONESYLL"].str.strip() != ""]  # Remove empty strings

# Count occurrences of each phonetic syllable
phonetic_counts = events["PHONESYLL"].value_counts()

# ✅ **Filter only phonetic syllables with ≥10 occurrences**
valid_syllables = phonetic_counts[phonetic_counts >= 10].index
filtered_events = events[events["PHONESYLL"].isin(valid_syllables)]

# Create unique numeric labels for phonetic syllables
unique_syllables = filtered_events["PHONESYLL"].unique()
label_mapping = {syll: i for i, syll in enumerate(unique_syllables)}

# Apply mapping to create numeric labels
filtered_events["NUMERIC_LABEL"] = filtered_events["PHONESYLL"].map(label_mapping)

# Extract Labels and EEG Segments
X = []
Y = []

for _, event in filtered_events.iterrows():
    label = event['NUMERIC_LABEL']  # Get numeric label from phonetic syllable
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
X_save_path = os.path.join(save_path, "X.npy")
Y_save_path = os.path.join(save_path, "Y.npy")

np.save(X_save_path, X)
np.save(Y_save_path, Y)

# Save label mapping for reference
label_mapping_path = os.path.join(save_path, "label_mapping.csv")
pd.DataFrame(list(label_mapping.items()), columns=["Phonetic_Syllable", "Numeric_Label"]).to_csv(label_mapping_path, index=False)

print(f"\n✅ Saved EEG Data: {X.shape} to {X_save_path}")
print(f"✅ Saved Labels: {Y.shape} to {Y_save_path}")
print(f"✅ Saved Label Mapping: {label_mapping_path}")
