import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt

audio_file = "k:/Thesis/labeled_dataset"
audio, sr = librosa.load(audio_file)

FRAME_SIZE = 2048
HOP_SIZE = 512

S_scale = librosa.stft(audio, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
S_scale.shape
type(S_scale[0][0])

Y_scale = np.abs(S_scale)