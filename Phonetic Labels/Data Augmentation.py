import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.utils import shuffle

# ======= User Inputs =======
dataset_path = r"B:\Education\CNN\EEG\Max's File\Processed_Phonetic_Numpy"
save_path_augmented = os.path.join(dataset_path, "Augmented_Balanced")
os.makedirs(save_path_augmented, exist_ok=True)

# Load original EEG Data
X = np.load(os.path.join(dataset_path, "X.npy"))
Y = np.load(os.path.join(dataset_path, "Y.npy"))
print(f"✅ Loaded Original EEG Data: {X.shape}, Labels: {Y.shape}")

# ======= Split into Train and Test (80%-20%) =======
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

print(f"🔹 Training set: {X_train.shape}, {Y_train.shape}")
print(f"🔹 Test set: {X_test.shape}, {Y_test.shape}")

# Save original test set immediately (without augmentation)
np.save(os.path.join(save_path_augmented, "X_test.npy"), X_test)
np.save(os.path.join(save_path_augmented, "Y_test.npy"), Y_test)

# ======= Data Augmentation Functions =======
def gaussian_noise(data, std=0.005):
    return data + np.random.normal(0, std, data.shape)

def time_shift(data, shift_max=50):
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(data, shift, axis=-1)

def amplitude_scaling(data, scale_range=(0.9, 1.1)):
    scale_factor = np.random.uniform(*scale_range)
    return data * scale_factor

augmentation_methods = [gaussian_noise, time_shift, amplitude_scaling]

# ======= Balancing via Augmentation ONLY on Training Data =======
class_counts = Counter(Y_train)
max_class_size = max(class_counts.values())
print(f"\n🔸 Class distribution before balancing: {class_counts}")

X_train_balanced, Y_train_balanced = [], []

for label in np.unique(Y_train):
    samples = X_train[Y_train == label]
    current_class_size = len(samples)
    samples_needed = max_class_size - current_class_size

    # Include original samples
    X_train_balanced.extend(samples)
    Y_train_balanced.extend([label] * current_class_size)

    # Augment minority classes
    for i in range(samples_needed):
        sample = samples[i % current_class_size]
        
        # Cycle through augmentation methods
        augmented_sample = augmentation_methods[i % len(augmentation_methods)](sample)
        
        X_train_balanced.append(augmented_sample)
        Y_train_balanced.append(label)

# Shuffle balanced data
X_train_balanced, Y_train_balanced = shuffle(
    np.array(X_train_balanced), np.array(Y_train_balanced), random_state=42
)

print(f"\n✅ After balancing:")
print(f"🔹 X_train_balanced shape: {X_train_balanced.shape}")
print(f"🔹 Y_train_balanced shape: {Y_train_balanced.shape}")
print(f"🔸 Class distribution after balancing: {Counter(Y_train_balanced)}")

# ======= Save Augmented & Balanced Training Data =======
np.save(os.path.join(save_path_augmented, "X_train_augmented_balanced.npy"), X_train_balanced)
np.save(os.path.join(save_path_augmented, "Y_train_augmented_balanced.npy"), Y_train_balanced)

print("\n✅ Balanced, Augmented Training Data Saved!")
print("✅ Original Test Data Saved (NO augmentation)!")
