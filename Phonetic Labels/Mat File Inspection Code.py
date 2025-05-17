import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mat73 import loadmat

# ========== User Inputs ==========
dataset_path = r"B:\Education\CNN\EEG\Max's File\Dataset"  # Path to dataset
file_name = "processed_merged.mat"  # Choose the .mat file to inspect
MODE = "overt"  # 'overt' or 'silent'
PROCESS_STAGE = "merged"  # 'merged' or 'ica_clean'

# Full path to file
file_path = os.path.join(dataset_path, file_name)

# ========== Load EEG Dataset ==========
print(f"\n🔹 Loading {file_name}...")

EEG = loadmat(file_path)['EEG']

# Fix nested dictionary structures
def fix_dict(d):
    out = d.copy()
    for i in range(len(out)):
        out[i] = {k: v[0] for k, v in out[i].items()}
    return out

# Define event & phonetic feature columns
base_cols = ['TYP', 'POS', 'DUR', 'SENT_LEN']
sound_cols = ['NASAL', 'OCCLUSIVE', 'AFFRICATE', 'FRICATIVE', 'VIBRANT',
              'LATERAL', 'APPROXIMANT', 'HIGH', 'MEDIUM_HIGH', 'MEDIUM_LOW', 'LOW']
pos_cols = ['BILABIAL', 'LABIODENTAL', 'DENTIALVEOLAR', 'POSTALVEOLAR',
            'PALATAL', 'VELAR', 'ANTERIOR', 'CENTRAL', 'POSTERIOR']

# Extract EEG Metadata
chanlocs = pd.DataFrame(fix_dict(EEG['chanlocs']))
fs = int(EEG['srate'].item())  # Sampling rate
print(f"\n✅ Sampling Rate: {fs} Hz")

# Extract Events
events = pd.DataFrame(fix_dict(EEG['event_gdf']))
events = events[base_cols + sound_cols + pos_cols]
events[base_cols] = events[base_cols].astype(int)
for column in sound_cols + pos_cols:
    events[column] = events[column].apply(lambda x: x.astype(bool) if not np.isnan(x) else pd.NA)

# Display EEG Channel Locations and Events
print("\n✅ EEG Channel Locations:")
print(chanlocs.head())

print("\n✅ First 10 EEG Events:")
print(events.head(10))

# ========== Visualizations ==========
# 1️⃣ **Plot Event Type Distribution**
plt.figure(figsize=(10, 5))
sns.countplot(data=events, x="TYP", palette="viridis")
plt.title("Distribution of Event Types in EEG Dataset")
plt.xlabel("Event Type (TYP)")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.show()

# 2️⃣ **Plot EEG Signals for Random Trials**
def plot_eeg_trial(eeg_data, sample_idx):
    """Plot EEG signals from a specific trial."""
    plt.figure(figsize=(12, 5))
    for ch in range(eeg_data.shape[0]):  # Loop through EEG channels
        plt.plot(eeg_data[ch] + ch * 10, label=f"Channel {ch}")  # Offset for clarity

    plt.title(f"EEG Signal - Trial {sample_idx}")
    plt.xlabel("Time Steps")
    plt.ylabel("EEG Amplitude")
    plt.legend()
    plt.show()

# Extract EEG Data (assuming data is in `EEG['data']`)
eeg_data = np.array(EEG['data']).T  # Shape (Time, Channels)
print(f"\n✅ EEG Data Shape: {eeg_data.shape}")  # Expecting (time_steps, channels)
num_channels = eeg_data.shape[1]
print(f"\n✅ EEG Data Shape: {eeg_data.shape}")  # (Time steps, Channels)
print(f"🔹 Total EEG Channels: {num_channels}")

# Select a random EEG trial and plot it
random_sample_idx = np.random.randint(0, len(eeg_data))
plot_eeg_trial(eeg_data, random_sample_idx)

# 3️⃣ **Phonetic Syllables Occurrence Plot**
syll_events = events[events.TYP == 781]  # Extract syllable-related events
syllable_counts = syll_events["TYP"].value_counts()

plt.figure(figsize=(10, 5))
sns.barplot(x=syllable_counts.index, y=syllable_counts.values, palette="mako")
plt.title("Phonetic Syllables Occurrence")
plt.xlabel("Phonetic Syllable Event Types")
plt.ylabel("Count")
plt.grid(axis='y')
plt.show()

# 4️⃣ **Speech Events Occurrence Plot**
speak_events = events[events.TYP == 782]  # Extract speech-related events
speech_counts = speak_events["TYP"].value_counts()

plt.figure(figsize=(10, 5))
sns.barplot(x=speech_counts.index, y=speech_counts.values, palette="coolwarm")
plt.title("Speech Events Occurrence")
plt.xlabel("Speech Event Types")
plt.ylabel("Count")
plt.grid(axis='y')
plt.show()
