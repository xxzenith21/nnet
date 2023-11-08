import numpy as np

# Replace 'your_file.npy' with the actual path to your .npy file
file_path = 'K:/Thesis/features/514lowattackhighreleasemetallicanaloglightvibratopluck_spectrogrampng_features.npy'

# Load the contents of the .npy file
loaded_data = np.load(file_path)

# Print the contents
print(loaded_data)