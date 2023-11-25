import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

mapping_file = 'K:/Thesis/labelMapping/label_to_index.npy'  
label_mapping = np.load(mapping_file, allow_pickle=True).item()
print("Contents:", label_mapping)
output = [1, 2, 3, 4, 5]

for number in output:
    corresponding_label = next((label for label, index in label_mapping.items() if index == number), "Unknown Label")
    print(f"Number {number} corresponds to '{corresponding_label}'")