import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

input_folder = "K:/Thesis/labeled_dataset"
output_folder = "K:/Thesis/spectro"
features_folder = "K:/Thesis/features"
matrix_folder = "K:/Thesis/featureMatrix"
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
clear_folder(output_folder)
clear_folder(features_folder)
clear_folder(matrix_folder)

def generate_and_save_spectrogram(audio_file, output_folder, frame_size=2048, hop_size=512):
    audio, sr = librosa.load(audio_file)
    
    S_audio = librosa.stft(audio, n_fft=frame_size, hop_length=hop_size)
    Y_audio = np.abs(S_audio) ** 2
    Y_log_audio = librosa.power_to_db(Y_audio)
    
    # Get the filename without extension
    filename = os.path.splitext(os.path.basename(audio_file))[0]
    
    # Save the spectrogram to the output folder
    output_path = os.path.join(output_folder, f"{filename}_spectrogram.png")
    plot_spectrogram(Y_log_audio, sr, hop_size, y_axis="log", save_path=output_path)
    plt.close()  # Close the figure after saving

def plot_spectrogram(Y, sr, hop_length, y_axis="linear", save_path=None):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y, 
                             sr=sr, 
                             hop_length=hop_length, 
                             x_axis="time", 
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def extract_features_from_spectrogram(spectro_path, sr):
    # Load the spectrogram image
    spectro_image = plt.imread(spectro_path)

    # Convert the image to a NumPy array
    spectro_array = np.array(spectro_image)

    # Extract Chroma features
    chroma_features = librosa.feature.chroma_stft(y=spectro_array.mean(axis=-1), sr=sr)

    # Extract MFCC features
    mfcc_features = librosa.feature.mfcc(y=spectro_array.mean(axis=-1), sr=sr, n_mfcc=13)

    # Extract Spectral Contrast features
    spectral_contrast = librosa.feature.spectral_contrast(y=spectro_array.mean(axis=-1), sr=sr)

    # Extract Spectral Centroid, Bandwidth, and Rolloff
    spectral_centroid = librosa.feature.spectral_centroid(y=spectro_array.mean(axis=-1), sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=spectro_array.mean(axis=-1), sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=spectro_array.mean(axis=-1), sr=sr)
    
    # Extract pitch range
    pitches, magnitudes = librosa.core.piptrack(y=spectro_array.mean(axis=-1), sr=sr)
    pitch_range = np.max(pitches, axis=0) - np.min(pitches, axis=0)

    # Ensure all features have compatible dimensions
    features = [chroma_features, mfcc_features, spectral_contrast, spectral_centroid, spectral_bandwidth, rolloff]

    # Flatten the features for simplicity
    flattened_features = np.concatenate([feature.flatten() for feature in features])

    # Add pitch range to the feature vector
    flattened_features = np.concatenate([flattened_features, pitch_range.flatten()])

    return flattened_features

# Process all audio files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".wav"):
        audio_path = os.path.join(input_folder, filename)
        generate_and_save_spectrogram(audio_path, output_folder, FRAME_SIZE, HOP_SIZE)

# Initialize an empty list to store feature vectors
feature_vectors = []

# Process all spectrograms in the output folder
for filename in os.listdir(output_folder):
    if filename.endswith("_spectrogram.png"):
        spectro_path = os.path.join(output_folder, filename)
        
        # Extract features from the spectrogram
        feature_vector = extract_features_from_spectrogram(spectro_path, librosa.get_samplerate(audio_path))
        
        # Create a valid filename for saving the features
        valid_filename = ''.join(e for e in filename if e.isalnum() or e in ['_', '-'])
        features_path = os.path.join(features_folder, f"{os.path.splitext(valid_filename)[0]}_features.npy")
        
        np.save(features_path, feature_vector)
        
        # Append the feature vector to the list
        feature_vectors.append(feature_vector)

# Convert the list of feature vectors to a NumPy array
feature_matrix = np.array(feature_vectors)

# Save the feature matrix
save_path = "K:/Thesis/featureMatrix/feature_matrix.npy"
np.save(save_path, feature_matrix)

feature_matrix = np.load("K:/Thesis/featureMatrix/feature_matrix.npy")
print("Shape of the Feature Matrix:", feature_matrix.shape)
print("Features extracted and saved successfully.\n\n")

# Shape of the Feature Matrix: (100, 180125)