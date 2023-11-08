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

def extract_features(file_name):
    # Implement logic to extract features from the file name
    # You might want to split the file name, handle different features, etc.
    # For simplicity, let's assume a basic case where the file name is split by '.'
    return file_name.split('.')

def label_from_file_name(file_name):
    # Implement logic to extract the label from the file name
    # This depends on your naming convention. For now, let's assume the label is the last part of the file name
    return file_name.split('.')[-1]

# Example usage
data_folder = "K:/Thesis/labeled_dataset"
dataset = load_dataset(data_folder)

