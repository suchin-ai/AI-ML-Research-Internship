import mne
import os

data_path = r'B:\Education\CNN\EEG\New\Dataset'
file_path = os.path.join(data_path, 'A01T.gdf')

# Load a single subject's data
raw = mne.io.read_raw_gdf(file_path, preload=True)
print(raw.info)  # Channel info, sampling rate, etc.

# Plot power spectral density
raw.plot_psd()

# View raw EEG signals
raw.plot(n_channels=22, duration=10, scalings='auto')

# Extract and print events
events, event_id = mne.events_from_annotations(raw)
print("Event IDs:", event_id)
print("Events shape:", events.shape)
