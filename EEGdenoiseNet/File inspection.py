import numpy as np
import os
import matplotlib.pyplot as plt

# Define your data path
data_path = r"B:\Education\CNN\EEG\EEGdenoiseNet\Data"

# Load datasets
EEG_clean = np.load(os.path.join(data_path, "EEG_clean.npy"))
EMG_noise = np.load(os.path.join(data_path, "Noisy_EEG_EMG_-7dB.npy"))
EOG_noise = np.load(os.path.join(data_path, "Noisy_EEG_EOG_-7dB.npy"))

# Print shapes
print(f"EEG_clean shape: {EEG_clean.shape}")
print(f"EMG_noise shape: {EMG_noise.shape}")
print(f"EOG_noise shape: {EOG_noise.shape}")

# Check datatype
print(f"EEG_clean dtype: {EEG_clean.dtype}")
print(f"EMG_noise dtype: {EMG_noise.dtype}")
print(f"EOG_noise dtype: {EOG_noise.dtype}")

# Min and Max values
print(f"EEG_clean min-max: {EEG_clean.min()} to {EEG_clean.max()}")
print(f"EMG_noise min-max: {EMG_noise.min()} to {EMG_noise.max()}")
print(f"EOG_noise min-max: {EOG_noise.min()} to {EOG_noise.max()}")


# Plot
fig, axs = plt.subplots(3, 1, figsize=(12, 8))

axs[0].plot(EEG_clean[1000, :])
axs[0].set_title(f"Clean EEG - Epoch 1000")

axs[1].plot(EMG_noise[1000, :])
axs[1].set_title(f"EMG Noise - Epoch 1000")

axs[2].plot(EOG_noise[1000, :])
axs[2].set_title(f"EOG Noise - Epoch 1000")

plt.tight_layout()
plt.show()