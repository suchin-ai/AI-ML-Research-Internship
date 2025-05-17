import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------ Paths ------------------

data_path = r"B:\Education\CNN\EEG\EEGdenoiseNet\Data"
snr_values = list(range(-7, 3))  # -7dB to +2dB
noise_types = ['EMG', 'EOG', 'BOTH']

# ------------------ Feature Names ------------------

feature_names = [
    'Delta Bandpower',
    'Theta Bandpower',
    'Alpha Bandpower',
    'Beta Bandpower',
    'Gamma Bandpower',
    'Mean',
    'Std Dev',
    'Skewness',
    'Kurtosis',
    'Spectral Entropy',
    'Spectral Edge Frequency'
]

# ------------------ Visualization Loop ------------------

for noise_type in noise_types:
    for snr in snr_values:
        noisy_path = os.path.join(data_path, "features", f"features_X_noisy_{noise_type}_{snr}dB.npy")
        clean_path = os.path.join(data_path, "features", f"features_Y_clean_{noise_type}_{snr}dB.npy")
        
        if not (os.path.exists(noisy_path) and os.path.exists(clean_path)):
            print(f"Files missing for {noise_type} at {snr}dB. Skipping...")
            continue
        
        X_noisy = np.load(noisy_path)
        Y_clean = np.load(clean_path)

        print(f"Plotting features for {noise_type} Noise at {snr} dB...")

        n_features = len(feature_names)
        
        plt.figure(figsize=(22, 18))
        for i in range(n_features):
            plt.subplot(4, 3, i + 1)
            plt.hist(Y_clean[:, i], bins=30, alpha=0.5, label='Clean', color='blue', density=True)
            plt.hist(X_noisy[:, i], bins=30, alpha=0.5, label='Noisy', color='red', density=True)
            plt.title(feature_names[i])
            plt.xlabel('Feature Value')
            plt.ylabel('Density')
            plt.legend()

        plt.suptitle(f"Feature Comparison: {noise_type} Noise at {snr} dB", fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
