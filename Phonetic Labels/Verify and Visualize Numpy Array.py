import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========== User Inputs ==========
dataset_path = r"B:\Education\CNN\EEG\Max's File\Processed_Phonetic_Numpy"  # Path to processed NumPy files
X_file = os.path.join(dataset_path, "X.npy")  # EEG data file
Y_file = os.path.join(dataset_path, "Y.npy")  # Labels file
label_mapping_file = os.path.join(dataset_path, "label_mapping.csv")  # Label mapping file

# ========== Load EEG Data & Labels ==========
print("\n🔹 Loading saved NumPy arrays...")
X = np.load(X_file)  # Shape: (Samples, Channels, Time)
Y = np.load(Y_file)  # Shape: (Samples,)

# Load label mapping
label_mapping_df = pd.read_csv(label_mapping_file)
label_mapping = dict(zip(label_mapping_df["Numeric_Label"], label_mapping_df["Phonetic_Syllable"]))

print(f"✅ Loaded EEG Data: {X.shape}")  # Expecting (Samples, Channels, Time)
print(f"✅ Loaded Labels: {Y.shape}")  # Expecting (Samples,)

# Unique phonetic labels & their counts
unique_labels, label_counts = np.unique(Y, return_counts=True)
print(f"\n✅ Unique Phonetic Labels: {len(unique_labels)}")
for label, count in zip(unique_labels, label_counts):
    print(f"Label {label_mapping[label]} ({label}): {count} occurrences")

# ========== Plot EEG Signals for Random Samples ==========
def plot_eeg_sample(sample_idx):
    """ Plot EEG signal from a given sample index. """
    eeg_sample = X[sample_idx]  # Extract EEG trial
    label = int(Y[sample_idx])  # Corresponding phonetic label (numeric)
    label_name = label_mapping[label]  # Get phonetic syllable name

    plt.figure(figsize=(12, 5))
    for ch in range(eeg_sample.shape[0]):  # Loop through EEG channels
        plt.plot(eeg_sample[ch] + ch * 10, label=f"Channel {ch}")  # Offset for clarity

    plt.title(f"EEG Signal - Sample {sample_idx} (Phonetic Label: {label_name} - {label})")
    plt.xlabel("Time Steps")
    plt.ylabel("EEG Amplitude (per channel)")
    plt.legend()
    plt.show()

# Select a few random EEG trials and visualize them
for i in range(3):  # Plot 3 random samples
    random_sample_idx = np.random.randint(0, len(X))
    plot_eeg_sample(random_sample_idx)

# ========== Visualize Label Distribution ==========
plt.figure(figsize=(12, 5))
sns.barplot(x=[label_mapping[lbl] for lbl in unique_labels], y=label_counts, palette="viridis")
plt.title("Phonetic Label Distribution in EEG Dataset")
plt.xlabel("Phonetic Syllable")
plt.ylabel("Count")
plt.xticks(rotation=90, fontsize=8)
plt.grid(axis='y')
plt.show()
