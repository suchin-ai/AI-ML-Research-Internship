import mne
import os

# Path to one GDF file
file_path = r'B:\Education\CNN\EEG\New\Dataset\A01T.gdf'

# Load GDF
raw = mne.io.read_raw_gdf(file_path, preload=True)

# Print detailed info
print("\n--- Channel Names ---")
print(raw.ch_names)

print("\n--- Number of Channels ---")
print(len(raw.ch_names))

print("\n--- Info Summary ---")
print(raw.info)

print("\n--- Annotations ---")
print(raw.annotations)

# Optional: View events
events, event_id = mne.events_from_annotations(raw)
print("\n--- Event IDs ---")
print(event_id)

print("\n--- Event samples shape ---")
print(events.shape)
