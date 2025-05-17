import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mat73 import loadmat

# ========== User Inputs ==========
dataset_path = r"B:\Education\CNN\EEG\Max's File\Dataset"  # Path to dataset
file_name = "processed_merged.mat"  # Choose the .mat file to inspect
file_path = os.path.join(dataset_path, file_name)

# ========== Load EEG Dataset ==========
print(f"\n🔹 Loading {file_name}...")

EEG = loadmat(file_path)['EEG']

# Fix nested dictionary structures
def fix_dict(d):
    """Fix nested MATLAB structures by extracting first elements."""
    out = d.copy()
    for i in range(len(out)):
        out[i] = {k: v[0] for k, v in out[i].items()}
    return out

# Extract Events
events = pd.DataFrame(fix_dict(EEG['event_gdf']))

# Check if PHONESYLL exists in dataset
if 'PHONESYLL' in events.columns:
    print("\n✅ Extracting Phonetic Labels...")

    # Remove empty or null phonetic labels
    phonetic_labels = events[['TYP', 'PHONESYLL']].dropna()
    phonetic_labels = phonetic_labels[phonetic_labels['PHONESYLL'] != ""]  # Remove empty strings

    # Display all unique phonetic labels
    unique_phonetic_labels = phonetic_labels['PHONESYLL'].unique()
    print(f"\n✅ Unique Phonetic Labels Found (After Removing Nulls): {len(unique_phonetic_labels)}")
    print(unique_phonetic_labels)  # Print list of all phonetic labels

    # Count occurrences of each phonetic syllable
    phonetic_counts = phonetic_labels['PHONESYLL'].value_counts()

    # Display first 10 phonetic labels with their counts
    print("\n✅ Top 10 Phonetic Labels:")
    print(phonetic_counts.head(10))

    # ========== Visualization ==========
    plt.figure(figsize=(14, 6))
    sns.barplot(x=phonetic_counts.index, y=phonetic_counts.values, palette="magma")  
    plt.title("Phonetic Syllable Occurrence (After Removing Nulls)")
    plt.xlabel("Phonetic Syllable")
    plt.ylabel("Count")
    plt.xticks(rotation=90, fontsize=8)
    plt.grid(axis='y')
    plt.show()

else:
    print("\n⚠ No PHONESYLL column found in the dataset.")
