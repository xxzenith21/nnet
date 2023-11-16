import numpy as np

# Replace 'path_to_your_file.npy' with the actual file path
file_path = 'K:/Thesis/labelMapping/label_to_index.npy'

# Load the file
data = np.load(file_path, allow_pickle=True)

# Print the data
print(data)