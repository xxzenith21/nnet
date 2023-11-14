import os
import numpy as np
import matplotlib.pyplot as plt

matrix_folder = "K:/Thesis/labelMapping"
FRAME_SIZE = 2048
HOP_SIZE = 512

# Function to clear the contents of a folder
def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

# Clear the contents of the output folder
clear_folder(matrix_folder)

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
label_matrix_file = "K:/Thesis/labelMapping/label_matrix.npy"
label_mapping_file = "K:/Thesis/labelMapping/label_to_index.npy"

# Load dataset and save label matrix and mapping
label_matrix, label_to_index = load_dataset(data_folder)
np.save(label_matrix_file, label_matrix)
np.save(label_mapping_file, label_to_index)

# Load and prepare spectrogram data
X_train, y_train = prepare_data(spectrogram_folder, label_matrix_file)

# Reshape X_train to remove the extra dimension
X_train = X_train[:, :, :, :, 0]  # New shape: (100, 1000, 2500, 1)

# Print shapes to verify
print(f"Shape of Label Matrix: {X_train.shape}")
print(f"Shape of Label-to-Index Mapping: {y_train.shape} \n\n")


# Shape of X_train: (100, 1000, 2500, 1)
# Shape of y_train: (100, 96)