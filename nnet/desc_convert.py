import numpy as np
from sklearn.decomposition import PCA

# Load the feature matrix
feature_matrix = np.load("K:/Thesis/featureMatrix/feature_matrix.npy")

# Dimensionality reduction using PCA
# Adjust the number of PCA components to 49 (a perfect square)
pca_components = 49
pca = PCA(n_components=pca_components)
reduced_feature_matrix = pca.fit_transform(feature_matrix)

# Reshape for Convolutional Neural Network
# Calculate the size to reshape into a square
side_length = int(np.sqrt(pca_components))
reshaped_feature_matrix = np.reshape(reduced_feature_matrix, (reduced_feature_matrix.shape[0], side_length, side_length))

# Save the reshaped feature matrix
reshaped_save_path = "K:/Thesis/featureMatrix/4d_matrix.npy"
np.save(reshaped_save_path, reshaped_feature_matrix)

print(f"Shape of Reshaped Feature Matrix: {reshaped_feature_matrix.shape} \n\n")

# Shape of Reshaped Feature Matrix: (100, 7, 7)