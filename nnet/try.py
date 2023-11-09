import numpy as np

# Replace 'your_file.npy' with the actual path to your .npy file
file_path = 'K:/Thesis/label_mapping.npy'

# Load the contents of the .npy file
loaded_data = np.load(file_path)

# Print the contents
print(loaded_data)