import numpy as np

# Path to your label_mapping.npy file
label_mapping_file = "K:/Thesis/labelMapping/label_mapping.npy"

# Load the file
try:
    label_mapping = np.load(label_mapping_file, allow_pickle=True)
    print("Contents of label_mapping.npy:")
    print(label_mapping)

    # Additional checks to print more details
    print("\nDetails:")
    print(f"Type of the contents: {type(label_mapping)}")

    if isinstance(label_mapping, dict):
        print("\nLabels and their corresponding indices:")
        for label, index in label_mapping.items():
            print(f"{label}: {index}")

    elif isinstance(label_mapping, np.ndarray):
        print("\nArray shape:", label_mapping.shape)

    else:
        print("\nThe content is neither a dictionary nor an array.")

except Exception as e:
    print(f"Error loading {label_mapping_file}: {e}")
