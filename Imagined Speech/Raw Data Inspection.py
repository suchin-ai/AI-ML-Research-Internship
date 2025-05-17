import h5py
import os

# Directory containing all the .h5 files
directory_path = r'B:\Education\CNN\EEG\Imagined Speech\Dataset'
files = [f for f in os.listdir(directory_path) if f.endswith('.h5')]

# Inspect each file
for file_name in files:
    file_path = os.path.join(directory_path, file_name)
    print(f"\nInspecting File: {file_name}")
    
    with h5py.File(file_path, 'r') as h5_file:
        # List all keys in the file
        keys = list(h5_file.keys())
        print(f"Keys: {keys}")
        
        # Display information about each key
        for key in keys:
            data = h5_file[key]
            
            if isinstance(data, h5py.Dataset):
                print(f" - Dataset: {key}")
                print(f"   - Shape: {data.shape}")
                print(f"   - Datatype: {data.dtype}")
            elif isinstance(data, h5py.Group):
                print(f" - Group: {key}")
                print(f"   - Contains {len(data.keys())} items.")
        
        # Display metadata if available
        print("\nMetadata in the file:")
        for attr, value in h5_file.attrs.items():
            print(f" - {attr}: {value}")
