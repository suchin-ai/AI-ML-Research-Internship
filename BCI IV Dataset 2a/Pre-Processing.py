import mne
import numpy as np
import os
import matplotlib.pyplot as plt

def plot_raw(raw, title):
    """Plot EEG signal."""
    fig = raw.plot(n_channels=22, scalings='auto', title=title, show=True)
    plt.show()

def preprocess_and_visualize(file_path, subject_id):
    print(f"\n--- Processing {subject_id} ---")
    raw = mne.io.read_raw_gdf(file_path, preload=True)

    # === Step 1: Correct channel names using known BCI IV 2a mapping ===
    rename_map = {
        'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 'EEG-3': 'FC2',
        'EEG-4': 'FC4', 'EEG-5': 'C5', 'EEG-C3': 'C3', 'EEG-6': 'C1', 'EEG-Cz': 'Cz',
        'EEG-7': 'C2', 'EEG-C4': 'C4', 'EEG-8': 'C6', 'EEG-9': 'CP3', 'EEG-10': 'CP1',
        'EEG-11': 'CPz', 'EEG-12': 'CP2', 'EEG-13': 'CP4', 'EEG-14': 'P1', 'EEG-Pz': 'Pz',
        'EEG-15': 'P2', 'EEG-16': 'POz',
        'EOG-left': 'EOG-left', 'EOG-central': 'EOG-central', 'EOG-right': 'EOG-right'
    }
    raw.rename_channels(rename_map)
    raw.set_channel_types({
        'EOG-left': 'eog',
        'EOG-central': 'eog',
        'EOG-right': 'eog'
    })
    
    # === Step 2: Apply standard montage with missing EOG ignored ===
    raw.set_montage('standard_1020', match_case=False, on_missing='ignore')

    # === Step 3: Plot Raw EEG ===
    plot_raw(raw.copy(), f'{subject_id} - Raw EEG')

    # === Step 4: Bandpass Filtering (1–40 Hz) ===
    raw.filter(1., 40., fir_design='firwin')
    plot_raw(raw.copy(), f'{subject_id} - After Bandpass (1–40Hz)')

    # === Step 5: Amplitude clipping to remove extreme artifacts ===
    raw._data[np.abs(raw._data) > 100e-6] = 0

    # === Step 6: Common Average Referencing (CAR) ===
    raw.set_eeg_reference('average', projection=False)
    plot_raw(raw.copy(), f'{subject_id} - After CAR')

    # === Step 7: ICA (Retaining EOG for blink detection) ===
    ica = mne.preprocessing.ICA(n_components=20, random_state=42, max_iter='auto')
    ica.fit(raw)
    eog_inds, _ = ica.find_bads_eog(raw)
    ica.exclude = eog_inds
    raw = ica.apply(raw)
    plot_raw(raw.copy(), f'{subject_id} - After ICA')

    # === Step 8: CSD (Current Source Density) ===
    raw_csd = mne.preprocessing.compute_current_source_density(raw)
    plot_raw(raw_csd.copy(), f'{subject_id} - After CSD')

    # === Step 9: Drop EOG channels now that ICA is done ===
    for ch in ['EOG-left', 'EOG-central', 'EOG-right']:
        if ch in raw_csd.ch_names:
            raw_csd.drop_channels(ch)

    # === Step 10: Extract motor imagery epochs ===
    print("Annotation descriptions available:", set(raw_csd.annotations.description))
    
    # Map motor imagery tasks
    motor_imagery_ids = {'769': 1, '770': 2, '771': 3, '772': 4}
    
    events, event_id = mne.events_from_annotations(raw_csd, event_id=motor_imagery_ids)
    print(f"Events extracted: {events.shape}")
    print(f"Event ID mapping: {event_id}")
    
    if events.shape[0] == 0:
        raise ValueError("No motor imagery events found. Check event extraction!")

    epochs = mne.Epochs(raw_csd, events, event_id=event_id,
                        tmin=0.5, tmax=2.5, baseline=None, preload=True)

    # === Step 11: Visualize 1 Epoch per class ===
    for class_label in range(1, 5):
        try:
            class_epochs = epochs[class_label]
            class_epochs.plot(n_epochs=1, title=f'{subject_id} - Class {class_label} Epoch')
        except Exception as e:
            print(f"Class {class_label} plot skipped: {e}")

    # === Step 12: Return numpy arrays ===
    X = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1]  # class labels
    return X, y

# === Batch Runner ===
def run_full_pipeline(data_path, output_path):
    subject_files = [f'A0{i}T.gdf' for i in range(1, 10) if i != 4]  # <-- exclude A04T.gdf
    all_X, all_y = [], []

    for file in subject_files:
        file_path = os.path.join(data_path, file)
        subject_id = file.split('.')[0]
        X, y = preprocess_and_visualize(file_path, subject_id)
        all_X.append(X)
        all_y.append(y)

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    np.save(os.path.join(output_path, 'EEG_X_CSD.npy'), X_all)
    np.save(os.path.join(output_path, 'EEG_y_CSD.npy'), y_all)

    print(f"\nSaved EEG_X_CSD.npy with shape {X_all.shape}")
    print(f"Saved EEG_y_CSD.npy with shape {y_all.shape}")

# === File Paths ===
data_path = r'B:\Education\CNN\EEG\New\Dataset'
output_path = os.path.join(data_path, "processed")
os.makedirs(output_path, exist_ok=True)

# === RUN ===
run_full_pipeline(data_path, output_path)
