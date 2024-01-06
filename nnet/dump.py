import numpy as np
import random
import os
import re

# Load label to index mapping from .npy file
file_path = 'K:/Thesis/labelMapping/label_to_index.npy'
label_to_index_mapping = np.load(file_path, allow_pickle=True).item()

# Set the range and the number of random numbers you want
lower_limit = 0
upper_limit = 247
num_numbers = 10

# Generate a set of unique random numbers
random_numbers = random.sample(range(lower_limit, upper_limit + 1), num_numbers)

# Map the random numbers to their corresponding labels using label to index mapping
mapped_labels = [list(label_to_index_mapping.keys())[list(label_to_index_mapping.values()).index(num)] for num in random_numbers]

# Print the result
print("Random Numbers:", random_numbers)

directory_path = r'K:/Thesis/try'
files = os.listdir(directory_path)

for i, file in enumerate(files):
    # Extract labels from the filename (excluding file number and extension)
    file_name_without_extension = os.path.splitext(file)[0]
    extracted_labels = re.split(r'\.\s*', file_name_without_extension, maxsplit=1)[1].split(', ')
    
    # Map the extracted labels to numerical indices
    numerical_indices = [label_to_index_mapping[label] for label in extracted_labels]

# Calculate the percentage similarity
common_elements = set(random_numbers) & set(numerical_indices)
percentage_similarity = (len(common_elements) / len(random_numbers)) * 100

print("Extracted Labels: ", extracted_labels)
print("Chromosome:", random_numbers)
print("Ground Truth (Numerical): ", numerical_indices)
print(f"Percentage Similarity: {percentage_similarity}%")
