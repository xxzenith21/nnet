import os
import numpy as np

def load_dataset(data_folder, label_mapping):
    dataset = []
    labels_set = set()
    
    for filename in os.listdir(data_folder):
        if filename.endswith(".wav"):
            # Extract labels from the file name
            labels = filename.split(".")[1].split(", ")  # Assuming labels are separated by commas and spaces
            dataset.append(labels)
            
            # Update the set of unique labels
            labels_set.update(labels)

    # Create a list of unique labels and sort them
    unique_labels = sorted(list(labels_set))
    
    # Create a mapping from labels to indices
    label_to_index = {label: i for i, label in enumerate(unique_labels)}

    # Create a label matrix
    label_matrix = np.zeros((len(dataset), len(unique_labels)))

    # Fill in the label matrix with one-hot encoding
    for i, labels in enumerate(dataset):
        for label in labels:
            label_index = label_to_index[label]
            label_matrix[i, label_index] = 1

    return label_matrix, label_to_index

# Example usage
data_folder = "K:/Thesis/labeled_dataset"
label_mapping_file = "K:/Thesis/labelMapping/label_mapping.npy"

# Load the dataset and get the label matrix and label-to-index mapping
label_matrix, label_to_index = load_dataset(data_folder, label_mapping_file)

# Save the label-to-index mapping for later use
np.save(label_mapping_file, label_to_index)

# Print the label matrix
print("Label Matrix:")
print(label_matrix)

num_unique_labels = np.sum(label_matrix, axis=1)
num_unique_sets = np.count_nonzero(num_unique_labels)

print(f"Number of unique sets of labels: {num_unique_sets}")