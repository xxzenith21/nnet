import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Define the folder paths and constants
label_matrix_file = "K:/Thesis/synth_settings/settings_label_matrix/settings_label_matrix.npy"
label_mapping_file = "K:/Thesis/synth_settings/settings_label_matrix/settings_label_to_index.npy"
csv_file_path = "K:/Thesis/synth_settings/synth_settings_dataset/labels.csv"

# Function to load a spectrogram from a file
def load_spectrogram(file_path):
    spectrogram = plt.imread(file_path)
    spectrogram = spectrogram / spectrogram.max()
    return spectrogram

# Function to prepare the data
def prepare_data(input_folder, label_file_path):
    X = []
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith("_spectrogram.png"):
            file_path = os.path.join(input_folder, filename)
            spectrogram = load_spectrogram(file_path)
            spectrogram = spectrogram[:, :, None]  # Add a channel dimension
            X.append(spectrogram)
    X = np.stack(X, axis=0)
    y = np.load(label_file_path, allow_pickle=True)
    return X, y

# Function to parse individual parameter string
def parse_parameters(param_string):
    params = {}
    for param in param_string.split(","):
        key, value = param.split("=")
        params[key.strip()] = float(value.strip()) if is_float(value.strip()) else value.strip()
    return params

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def load_and_process_labels(csv_file_path, continuous_columns, categorical_columns):
    labels_df = pd.read_csv(csv_file_path)

    # List to store processed rows
    processed_rows = []

    # Parse each row and add to processed_rows
    for _, row in labels_df.iterrows():
        parsed_params = parse_parameters(row['Parameters'])
        processed_rows.append(parsed_params)

    # Convert list of dicts to DataFrame
    processed_df = pd.DataFrame(processed_rows)

    # Process continuous and categorical data
    continuous_data = processed_df[continuous_columns].fillna(0).values
    scaler = MinMaxScaler()
    continuous_normalized = scaler.fit_transform(continuous_data)

    categorical_data = processed_df[categorical_columns].fillna('None').values
    encoder = OneHotEncoder()
    categorical_encoded = encoder.fit_transform(categorical_data).toarray()

    # Combine continuous and categorical data into a label matrix
    label_matrix = np.hstack((continuous_normalized, categorical_encoded))

    # Create label-to-index mapping for categorical data
    label_to_index_mapping = {}
    for i, category in enumerate(encoder.categories_):
        for j, label in enumerate(category):
            label_to_index_mapping[label] = (i, j)

    return label_matrix, label_to_index_mapping

# Define the names of the continuous and categorical columns based on your CSV file
continuous_columns = [
    'tune', 'transposition', 'delay_feedback', 'delay_mix', 'delay_time',
    'distortion_drive', 'distortion_mix', 'reverb_size', 'reverb_damp',
    'reverb_mix', 'attack', 'decay', 'sustain', 'release',
    'filter_drive', 'filter_envelope_depth', 'filter_freq', 'filter_key_track'
]

categorical_columns = [
    'oscillator_type', 'distortion_type', 'lfo_freq', 'delay_freq'
]

# Correctly call the function with these arguments
label_matrix, label_to_index = load_and_process_labels(
    csv_file_path, continuous_columns, categorical_columns
)

# Save the label matrix and label-to-index mapping
np.save(label_matrix_file, label_matrix)
np.save(label_mapping_file, label_to_index)

# # Print shapes to verify
# print(f"Shape of Label Matrix: {label_matrix.shape}")
# print("Label-to-Index Mapping:", label_to_index)





