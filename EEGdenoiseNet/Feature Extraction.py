import numpy as np
import scipy.signal
import scipy.stats
import os

# ------------------ Feature Extraction ------------------

bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 80)
}

def bandpower(epoch, sfreq, band):
    fmin, fmax = band
    freqs, psd = scipy.signal.welch(epoch, sfreq, nperseg=256)
    band_power = np.trapz(psd[(freqs >= fmin) & (freqs <= fmax)], freqs[(freqs >= fmin) & (freqs <= fmax)])
    return band_power

def spectral_entropy(epoch, sfreq):
    freqs, psd = scipy.signal.welch(epoch, sfreq, nperseg=256)
    psd_norm = psd / np.sum(psd)
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    return entropy

def spectral_edge_freq(epoch, sfreq, edge=0.9):
    freqs, psd = scipy.signal.welch(epoch, sfreq, nperseg=256)
    cumulative_power = np.cumsum(psd)
    total_power = cumulative_power[-1]
    idx_edge = np.where(cumulative_power >= total_power * edge)[0][0]
    return freqs[idx_edge]

def extract_features_single_epoch(epoch, sfreq=512):
    features = []
    for band in bands.values():
        features.append(bandpower(epoch, sfreq, band))
    features.append(np.mean(epoch))
    features.append(np.std(epoch))
    features.append(scipy.stats.skew(epoch))
    features.append(scipy.stats.kurtosis(epoch))
    features.append(spectral_entropy(epoch, sfreq))
    features.append(spectral_edge_freq(epoch, sfreq))
    return np.array(features)

def extract_features_batch(epochs, sfreq=512):
    return np.array([extract_features_single_epoch(epoch, sfreq) for epoch in epochs])

# ------------------ Paths and Setup ------------------

data_path = r"B:\Education\CNN\EEG\EEGdenoiseNet\Data"
save_path = os.path.join(data_path, "features")
os.makedirs(save_path, exist_ok=True)

EEG_clean = np.load(os.path.join(data_path, "EEG_clean.npy"))
EMG_noise = np.load(os.path.join(data_path, "EMG_all_epochs.npy"))
EOG_noise = np.load(os.path.join(data_path, "EOG_all_epochs.npy"))

n_epochs = EEG_clean.shape[0]
snr_values = list(range(-7, 3))

# ------------------ Extract Features for EMG and EOG ------------------

for noise_type in ['EMG', 'EOG']:
    for snr in snr_values:
        noisy_file = f"Noisy_EEG_{noise_type}_{snr}dB.npy"
        noisy_path = os.path.join(data_path, noisy_file)
        if not os.path.exists(noisy_path):
            print(f"File {noisy_file} does not exist. Skipping...")
            continue

        noisy_epochs = np.load(noisy_path)

        X_noisy = extract_features_batch(noisy_epochs)
        Y_clean = extract_features_batch(EEG_clean)

        np.save(os.path.join(save_path, f"features_X_noisy_{noise_type}_{snr}dB.npy"), X_noisy)
        np.save(os.path.join(save_path, f"features_Y_clean_{noise_type}_{snr}dB.npy"), Y_clean)

        print(f"Saved features for {noise_type} at {snr}dB.")

# ------------------ Generate and Extract Features for BOTH ------------------

emg_idx = np.random.choice(EMG_noise.shape[0], n_epochs, replace=True)
eog_idx = np.random.choice(EOG_noise.shape[0], n_epochs, replace=True)

for snr in snr_values:
    snr_linear = 10 ** (snr / 10)
    eeg_power = np.mean(EEG_clean ** 2, axis=1, keepdims=True)

    combined_noise = EMG_noise[emg_idx] + EOG_noise[eog_idx]
    combined_noise_power = np.mean(combined_noise ** 2, axis=1, keepdims=True)

    scale = np.sqrt(eeg_power / (snr_linear * combined_noise_power + 1e-8))
    noisy_both = EEG_clean + scale * combined_noise

    X_noisy_both = extract_features_batch(noisy_both)
    Y_clean_both = extract_features_batch(EEG_clean)

    np.save(os.path.join(save_path, f"features_X_noisy_BOTH_{snr}dB.npy"), X_noisy_both)
    np.save(os.path.join(save_path, f"features_Y_clean_BOTH_{snr}dB.npy"), Y_clean_both)

    print(f"Saved features for BOTH noise at {snr}dB.")
