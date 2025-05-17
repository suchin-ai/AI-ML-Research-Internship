import numpy as np
import os
import scipy.signal
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

# ------------------ Settings ------------------

data_path = r"B:\Education\CNN\EEG\EEGdenoiseNet\Data\features"
save_result_path = r"B:\Education\CNN\EEG\EEGdenoiseNet\Data"

# ✅ ADD COMBINED NOISE TYPE HERE
noise_types = ['EMG', 'EOG', 'BOTH']
snr_values = list(range(-7, 3))  # [-7, ..., 2]
models_to_train = ['RandomForest', 'XGBoost', 'SVR', 'KNN']

# ------------------ Metrics Functions ------------------

def compute_rrmse(true, pred):
    return np.sqrt(np.mean((true - pred) ** 2)) / (np.sqrt(np.mean(true ** 2)) + 1e-8)

def compute_rrmse_spectral(true, pred, sfreq=512):
    freqs, psd_true = scipy.signal.welch(true, sfreq, nperseg=256)
    freqs, psd_pred = scipy.signal.welch(pred, sfreq, nperseg=256)
    return np.sqrt(np.mean((psd_true - psd_pred) ** 2)) / (np.sqrt(np.mean(psd_true ** 2)) + 1e-8)

def compute_acc(true, pred):
    corr_list = []
    for i in range(true.shape[0]):
        corr = np.corrcoef(true[i], pred[i])[0, 1]
        corr_list.append(corr)
    return np.mean(corr_list)

# ------------------ Model Builder ------------------

def get_model(name):
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    elif name == "XGBoost":
        return xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    elif name == "SVR":
        return SVR(kernel='rbf', C=1.0, epsilon=0.1)
    elif name == "KNN":
        return KNeighborsRegressor(n_neighbors=5)
    else:
        raise ValueError("Unknown model name")

# ------------------ Training & Evaluation ------------------

results = {}

for model_name in models_to_train:
    print(f"\n=== Training Model: {model_name} ===\n")
    results[model_name] = {}
    
    for noise_type in noise_types:
        print(f"--- Noise Type: {noise_type} ---")
        results[model_name][noise_type] = {'rrmse': [], 'rrmse_spectral': [], 'acc': []}
        
        for snr in snr_values:
            print(f"Processing SNR: {snr} dB")
            X_path = os.path.join(data_path, f"features_X_noisy_{noise_type}_{snr}dB.npy")
            Y_path = os.path.join(data_path, f"features_Y_clean_{noise_type}_{snr}dB.npy")
            
            if not os.path.exists(X_path) or not os.path.exists(Y_path):
                print(f"Missing file at {snr} dB for {noise_type}. Skipping.")
                continue

            X_noisy = np.load(X_path)
            Y_clean = np.load(Y_path)

            n_samples = X_noisy.shape[0]
            split_idx = int(0.8 * n_samples)

            X_train, X_test = X_noisy[:split_idx], X_noisy[split_idx:]
            Y_train, Y_test = Y_clean[:split_idx], Y_clean[split_idx:]

            base_model = get_model(model_name)
            model = MultiOutputRegressor(base_model)

            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)

            # Metrics
            rrmse_val = compute_rrmse(Y_test, Y_pred)
            rrmse_spectral_val = compute_rrmse_spectral(Y_test, Y_pred)
            acc_val = compute_acc(Y_test, Y_pred)

            results[model_name][noise_type]['rrmse'].append(rrmse_val)
            results[model_name][noise_type]['rrmse_spectral'].append(rrmse_spectral_val)
            results[model_name][noise_type]['acc'].append(acc_val)

# ------------------ Visualization ------------------

colors = {
    'RandomForest': 'blue',
    'XGBoost': 'green',
    'SVR': 'red',
    'KNN': 'purple'
}

metrics_names = ['rrmse', 'rrmse_spectral', 'acc']

for noise_type in noise_types:
    for metric in metrics_names:
        plt.figure(figsize=(10,6))
        for model_name in models_to_train:
            plt.plot(snr_values, results[model_name][noise_type][metric],
                     label=model_name, marker='o', color=colors[model_name])
        
        plt.title(f"{metric.upper()} vs SNR ({noise_type} Noise)")
        plt.xlabel('SNR (dB)')
        plt.ylabel(metric.upper())
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_result_path, f"{metric}_{noise_type}.png"))
        plt.show()

# ------------------ Bar Plot: Average ACC Comparison ------------------

for noise_type in noise_types:
    avg_accs = []
    for model_name in models_to_train:
        acc_list = results[model_name][noise_type]['acc']
        avg_acc = np.mean(acc_list)
        avg_accs.append(avg_acc)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models_to_train, avg_accs, color=[colors[m] for m in models_to_train])
    plt.title(f"Average ACC across SNRs ({noise_type} Noise)")
    plt.ylabel("Average Correlation Coefficient (ACC)")
    plt.ylim(0.9, 1.0)
    plt.grid(axis='y')
    
    # Annotate values on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height - 0.005,
                 f'{height:.4f}', ha='center', color='white', fontweight='bold')

    plt.savefig(os.path.join(save_result_path, f"Average_ACC_{noise_type}.png"))
    plt.show()

print("\nAll models trained, evaluated and results plotted successfully.")
