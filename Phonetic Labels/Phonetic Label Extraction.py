import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mat73 import loadmat

# ========== User Inputs ==========
dataset_path = r"B:\Education\CNN\EEG\Max's File\Dataset"  # Path to dataset
save_path = r"B:\Education\CNN\EEG\Max's File\Processed_Phonetic_CSV"  # Save location for CSV
file_name = "processed_merged.mat"  # Choose the .mat file to inspect
csv_output = os.path.join(save_path, "filtered_phonetic_labels.csv")

os.makedirs(save_path, exist_ok=True)  # Ensure save directory exists

# Full path to file
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

    # Remove empty phonetic labels
    phonetic_labels = events[['TYP', 'PHONESYLL']].dropna()
    phonetic_labels = phonetic_labels[phonetic_labels['PHONESYLL'] != ""]  # Remove empty strings

    # Count occurrences of each phonetic syllable
    phonetic_counts = phonetic_labels['PHONESYLL'].value_counts()

    # Filter out phonetic labels with fewer than 10 occurrences
    filtered_labels = phonetic_labels[phonetic_labels['PHONESYLL'].isin(
        phonetic_counts[phonetic_counts >= 10].index
    )]

    # Save to CSV with phonetic syllable names as labels
    filtered_labels.to_csv(csv_output, index=False)
    print(f"\n✅ Saved Filtered Phonetic Labels to {csv_output}")

    # ========== Visualization ==========
    plt.figure(figsize=(14, 6))
    sns.barplot(x=phonetic_counts[phonetic_counts >= 10].index, 
                y=phonetic_counts[phonetic_counts >= 10].values, palette="magma")  
    plt.title("Phonetic Syllable Occurrence (≥10 Occurrences)")
    plt.xlabel("Phonetic Syllable")
    plt.ylabel("Count")
    plt.xticks(rotation=90, fontsize=8)
    plt.grid(axis='y')
    plt.show()

else:
    print("\n⚠ No PHONESYLL column found in the dataset.")
