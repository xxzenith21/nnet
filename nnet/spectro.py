import os
import librosa
import numpy as np
from PIL import Image

print("Running...")
# Placeholder function for generating spectrogram
def generate_spectrogram(audio_path, save_path):
    # Add your code for generating spectrogram here
    pass

# Function to extract features from a single spectrogram
def extract_features_from_spectrogram(spectro_path):
    # Load the spectrogram image
    spectro_image = Image.open(spectro_path)

    # Convert the image to a NumPy array
    spectro_array = np.array(spectro_image)

    # Extract Chroma features
    chroma_features = librosa.feature.chroma_stft(y=spectro_array.mean(axis=-1), sr=sr)

    # Extract Spectral Contrast features
    spectral_contrast = librosa.feature.spectral_contrast(y=spectro_array.mean(axis=-1), sr=sr)

    # Extract Spectral Centroid, Bandwidth, and Rolloff
    spectral_centroid = librosa.feature.spectral_centroid(y=spectro_array.mean(axis=-1), sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=spectro_array.mean(axis=-1), sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=spectro_array.mean(axis=-1), sr=sr)

    # Flatten the features for simplicity
    flattened_features = np.concatenate([chroma_features.flatten(),
                                         spectral_contrast.flatten(),
                                         spectral_centroid.flatten(),
                                         spectral_bandwidth.flatten(),
                                         rolloff.flatten()])

    return flattened_features

# Function to generate and save MFCC features
def generate_and_save_mfcc_features(audio_path, save_folder):
    y, sr = librosa.load(audio_path)
    mfcc_features = librosa.feature.mfcc(y=y, sr=sr)
    np.save(os.path.join(save_folder, f"{os.path.splitext(os.path.basename(audio_path))[0]}_mfcc.npy"), mfcc_features)

# Function to generate spectrograms and save MFCC features for all audio files in a folder
def process_audio_folder(folder_path, save_folder):
    # Create the save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            audio_path = os.path.join(folder_path, filename)
            save_path = os.path.join(save_folder, f"{os.path.splitext(filename)[0]}_spectrogram.png")

            generate_spectrogram(audio_path, save_path)
            generate_and_save_mfcc_features(audio_path, save_folder)

# Function to extract features from saved spectrograms
def extract_features_from_spectrograms(spectro_folder):
    features = []

    # Iterate through all files in the folder
    for filename in os.listdir(spectro_folder):
        if filename.endswith("_spectrogram.png"):
            spectro_path = os.path.join(spectro_folder, filename)

            # Extract features from the spectrogram
            feature_vector = extract_features_from_spectrogram(spectro_path)

            # Append the features and the label to the list
            features.append((feature_vector, get_label_from_filename(filename)))

    return features

# Function to extract label from filename
def get_label_from_filename(filename):
    # Extract label from the filename (you can customize this based on your filenames)
    label = filename.split(".")[1]
    return label

# Example usage:
audio_folder = 'K:/Thesis/labeled_dataset'
save_folder = 'K:/Thesis/spectro'
process_audio_folder(audio_folder, save_folder)

generate_spectrogram(audio_folder, save_folder)

# Extract features from saved spectrograms
extracted_features = extract_features_from_spectrograms(save_folder)

# Print the extracted features
for feature_vector, label in extracted_features:
    print(f"Label: {label}, Features: {feature_vector}")

print(extracted_features)
print("Finished.")
