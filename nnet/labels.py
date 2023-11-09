import os
import numpy as np

def load_dataset(data_folder):
    dataset = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".wav"):
            # Extract labels from the file name
            labels = filename.split(".")[1].split(", ")  # Assuming labels are separated by commas and spaces
            dataset.append(labels)
    return dataset

def create_label_matrix(dataset):
    # Step 1: Identify unique labels
    unique_labels = set(label for example_labels in dataset for label in example_labels)

    # Step 2: Create a mapping between labels and integer indices
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    # Step 3: Create a binary matrix (label matrix)
    label_matrix = np.zeros((len(dataset), len(unique_labels)), dtype=int)

    for example_idx, example_labels in enumerate(dataset):
        for label in example_labels:
            # Set the corresponding entry in the label matrix to 1
            label_matrix[example_idx, label_to_index[label]] = 1

    return label_matrix, label_to_index

# Example usage
data_folder = "K:/Thesis/labeled_dataset"
dataset = load_dataset(data_folder)

# Create the label matrix and label-to-index mapping
label_matrix, label_to_index = create_label_matrix(dataset)

# Print the label matrix and label-to-index mapping
print("Label Matrix:")
print(label_matrix)

print("\nLabel-to-Index Mapping:")
print(label_to_index)
