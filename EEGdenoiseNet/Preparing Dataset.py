import numpy as np
import os

# ------------------ Paths ------------------

data_path = r"B:\Education\CNN\EEG\EEGdenoiseNet\Data"

# Load datasets
EEG_clean = np.load(os.path.join(data_path, "EEG_all_epochs.npy"))
EMG_noise = np.load(os.path.join(data_path, "EMG_all_epochs.npy"))
EOG_noise = np.load(os.path.join(data_path, "EOG_all_epochs.npy"))

print(f"EEG_clean shape: {EEG_clean.shape}")
print(f"EMG_noise shape: {EMG_noise.shape}")
print(f"EOG_noise shape: {EOG_noise.shape}")

# ------------------ Functions ------------------

def standardize_noise(noise):
    noise_mean = np.mean(noise, axis=1, keepdims=True)
    noise_std = np.std(noise, axis=1, keepdims=True) + 1e-6
    return (noise - noise_mean) / noise_std

def mix_eeg_noise(eeg_clean, noise, target_snr_db):
    eeg_power = np.mean(eeg_clean ** 2, axis=1, keepdims=True)
    noise_power = np.mean(noise ** 2, axis=1, keepdims=True)
    snr_linear = 10 ** (target_snr_db / 10)
    scaling_factor = np.sqrt(eeg_power / (snr_linear * noise_power + 1e-8))
    mixed = eeg_clean + scaling_factor * noise
    return mixed

# ------------------ Standardize Noises ------------------

EMG_noise_std = standardize_noise(EMG_noise)
EOG_noise_std = standardize_noise(EOG_noise)

# ------------------ Main Loop for All SNRs ------------------

snr_values = list(range(-7, 3))  # [-7, -6, ..., 2]
num_epochs = EEG_clean.shape[0]

# Randomly select noise samples
emg_indices = np.random.choice(EMG_noise_std.shape[0], num_epochs, replace=True)
eog_indices = np.random.choice(EOG_noise_std.shape[0], num_epochs, replace=True)

emg_selected = EMG_noise_std[emg_indices]
eog_selected = EOG_noise_std[eog_indices]
combined_noise = emg_selected + eog_selected

for snr in snr_values:
    print(f"\nProcessing SNR {snr} dB...")

    # Generate noisy EEG for EMG only
    noisy_eeg_emg = mix_eeg_noise(EEG_clean, emg_selected, snr)
    np.save(os.path.join(data_path, f"Noisy_EEG_EMG_{snr}dB.npy"), noisy_eeg_emg)
    print(f"Saved Noisy_EEG_EMG_{snr}dB.npy")

    # Generate noisy EEG for EOG only
    noisy_eeg_eog = mix_eeg_noise(EEG_clean, eog_selected, snr)
    np.save(os.path.join(data_path, f"Noisy_EEG_EOG_{snr}dB.npy"), noisy_eeg_eog)
    print(f"Saved Noisy_EEG_EOG_{snr}dB.npy")

    # Generate noisy EEG for BOTH (EMG + EOG)
    noisy_eeg_both = mix_eeg_noise(EEG_clean, combined_noise, snr)
    np.save(os.path.join(data_path, f"Noisy_EEG_BOTH_{snr}dB.npy"), noisy_eeg_both)
    print(f"Saved Noisy_EEG_BOTH_{snr}dB.npy")

print("\nPreprocessing and saving for all SNR levels completed!")
