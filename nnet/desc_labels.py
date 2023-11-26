import os
import numpy as np

def load_dataset(data_folder):
    dataset = []
    labels_set = set()
    
    for filename in os.listdir(data_folder):
        if filename.endswith(".wav"):
            labels = filename.split('.')[1].split(',')
            labels = [label.strip() for label in labels]
            dataset.append(labels)
            labels_set.update(labels)

    unique_labels = sorted(list(labels_set))
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    label_matrix = np.zeros((len(dataset), len(unique_labels)))

    for i, labels in enumerate(dataset):
        for label in labels:
            label_index = label_to_index[label]
            label_matrix[i, label_index] = 1

    return label_matrix, label_to_index

# Example usage
data_folder = "K:/Thesis/labeled_dataset"
label_matrix_file = "K:/Thesis/labelMapping/label_matrix.npy"
label_mapping_file = "K:/Thesis/labelMapping/label_to_index.npy"

# Load dataset and save label matrix and mapping
label_matrix, label_to_index = load_dataset(data_folder)
np.save(label_matrix_file, label_matrix)
np.save(label_mapping_file, label_to_index)

print(f"Shape of Label Matrix: {label_matrix.shape}")
print(f"Label-to-Index Mapping: {label_to_index}")
