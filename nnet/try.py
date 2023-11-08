import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

input_folder = "K:/Thesis/labeled_dataset"
output_folder = "K:/Thesis/spectro"
FRAME_SIZE = 2048
HOP_SIZE = 512

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

# Process all audio files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".wav"):
        audio_path = os.path.join(input_folder, filename)
        generate_and_save_spectrogram(audio_path, output_folder, FRAME_SIZE, HOP_SIZE)

print("Spectrograms generated and saved successfully.")
