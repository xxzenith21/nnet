import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

feature_matrix = np.load("K:/Thesis/synth_settings/settings_label_matrix/settings_label_matrix.npy")
print("Shape of the Feature Matrix:", feature_matrix.shape)