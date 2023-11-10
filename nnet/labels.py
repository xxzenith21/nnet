import os
import numpy as np
import matplotlib.pyplot as plt

def load_spectrogram(file_path):
    spectrogram = plt.imread(file_path)
    spectrogram = spectrogram / spectrogram.max()
    return spectrogram

def prepare_data(input_folder, label_file_path):
    # Load and process spectrograms
    X = []
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith("_spectrogram.png"):
            file_path = os.path.join(input_folder, filename)
            spectrogram = load_spectrogram(file_path)
            spectrogram = spectrogram[:, :, None]  # Add a channel dimension
            X.append(spectrogram)
    X = np.stack(X, axis=0)

    # Load label matrix
    y = np.load(label_file_path, allow_pickle=True)

    return X, y

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
spectrogram_folder = "K:/Thesis/spectro"
label_mapping_file = "K:/Thesis/labelMapping/label_mapping.npy"

# Load the dataset and get the label matrix and label-to-index mapping
label_matrix, label_to_index = load_dataset(data_folder)
np.save(label_mapping_file, label_to_index)  # Save the label-to-index mapping

# Load and prepare spectrogram data
X_train, y_train = prepare_data(spectrogram_folder, label_mapping_file)

# Print the label matrix and label-to-index mapping
print("Label Matrix:")
print(label_matrix)
print("\nLabel to Index Mapping:")
print(label_to_index)

# Print shapes to verify
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
