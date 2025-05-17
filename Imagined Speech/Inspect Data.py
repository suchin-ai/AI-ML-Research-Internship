import h5py

# Path to one of the dataset files
file_path = r'B:\Education\CNN\EEG\Imagined Speech\Dataset\S6-phono-1_imagined_speech.h5'

with h5py.File(file_path, 'r') as h5_file:
    # List all keys (groups or datasets) in the file
    keys = list(h5_file.keys())
    print("Keys in the .h5 file:", keys)
    
    # Inspecting the contents of each key
    for key in keys:
        data = h5_file[key]
        if isinstance(data, h5py.Dataset):
            print(f"\nDataset: {key}")
            print(f" - Shape: {data.shape}")
            print(f" - Datatype: {data.dtype}")
        elif isinstance(data, h5py.Group):
            print(f"\nGroup: {key}")
            print(f" - Contains {len(data.keys())} items.")
    # Print all attributes (metadata) in the file
    print("Metadata in the file:")
    for key, value in h5_file.attrs.items():
        print(f" - {key}: {value}")

