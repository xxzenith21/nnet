import os
import numpy as np
import pandas as pd
import librosa.display
from matplotlib import pyplot as plt
# import spectro
# import matrix_convert
# import labels

# NEURAL NETWORK MAIN
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    x_clipped = np.clip(x, -709, 709)  # Clip values to avoid overflow
    return 1 / (1 + np.exp(-x_clipped))

def binary_crossentropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def back_prop(conv_layer, fc_layer, X, Y, conv_output, fc_output):
    m = Y.shape[0]

    # Gradient of loss w.r.t the output of the fully connected layer
    dA2 = fc_output - Y
    dZ2 = dA2 * sigmoid_derivative(fc_output)
    dW2 = np.dot(conv_output.reshape(m, -1).T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # Backpropagation through the convolutional layer would go here
    # Placeholder for convolutional layer gradients
    dW1, db1 = np.zeros_like(conv_layer.weights), np.zeros_like(conv_layer.bias)

    return dW1, db1, dW2, db2

def update_weights(layer, dw, db, learning_rate):
    layer.weights -= learning_rate * dw
    layer.bias -= learning_rate * db

def gradient_descent(X, Y, alpha, epochs, conv_layer, fc_layer):
    for epoch in range(epochs):
        conv_output = conv_layer.forward(X)
        fc_output = fc_layer.forward(conv_output.reshape(conv_output.shape[0], -1))

        loss = binary_crossentropy(Y, fc_output)
        dW1, db1, dW2, db2 = back_prop(conv_layer, fc_layer, X, Y, conv_output, fc_output)

        update_weights(conv_layer, dW1, db1, alpha)
        update_weights(fc_layer, dW2, db2, alpha)

        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")


class Conv2DLayer:
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding):
        self.weights = np.random.randn(output_channels, input_channels, kernel_size, kernel_size) * 0.1
        self.bias = np.zeros((1, output_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, X):
        # Add padding
        X_padded = np.pad(X, [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)], mode='constant')
        
        batch_size, input_channels, input_height, input_width = X_padded.shape
        kernel_height, kernel_width = self.weights.shape[2], self.weights.shape[3]
        output_height = (input_height - kernel_height) // self.stride + 1
        output_width = (input_width - kernel_width) // self.stride + 1

        # Output tensor
        output = np.zeros((batch_size, self.weights.shape[0], output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + kernel_height
                w_start = j * self.stride
                w_end = w_start + kernel_width
                X_slice = X_padded[:, :, h_start:h_end, w_start:w_end]
                for k in range(self.weights.shape[0]):
                    output[:, k, i, j] = np.sum(X_slice * self.weights[k, :, :, :], axis=(1, 2, 3)) + self.bias[0, k]
        return relu(output)
    
    def save_model(self, file_path):
        # Save model parameters using numpy's save function
        np.savez(file_path, weights=self.weights, bias=self.bias, stride=self.stride, padding=self.padding)

    def load_model(self, file_path):
        # Load model parameters using numpy's load function
        loaded_params = np.load(file_path)
        self.weights = loaded_params['weights']
        self.bias = loaded_params['bias']
        self.stride = loaded_params['stride']
        self.padding = loaded_params['padding']

class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros((1, output_size))

    def forward(self, X):
        X_flattened = X.reshape(X.shape[0], -1)
        output = np.dot(X_flattened, self.weights) + self.bias
        return sigmoid(output)
    
    def save_model(self, file_path):
        # Save model parameters using numpy's save function
        np.savez(file_path, weights=self.weights, bias=self.bias)

    def load_model(self, file_path):
        # Load model parameters using numpy's load function
        loaded_params = np.load(file_path)
        self.weights = loaded_params['weights']
        self.bias = loaded_params['bias']

def visualize_feature_maps(output, num_filters):
    # Set up the subplot dimensions
    cols = rows = int(np.ceil(np.sqrt(num_filters)))

    # Create a figure to plot the feature maps
    plt.figure(figsize=(15, 15))

    # Plot each feature map
    for i in range(num_filters):
        ax = plt.subplot(cols, rows, i + 1)
        plt.imshow(output[0, i, :, :], cmap='gray')  # Assuming grayscale feature maps
        plt.axis('off')

    # Display the feature maps
    plt.show()

# Function to get the output features from the convolutional layer
def get_audio_features(conv_layer, X):
    conv_output = conv_layer.forward(X)
    # Flatten the output to make it compatible with the second model
    flattened_output = conv_output.reshape(conv_output.shape[0], -1)
    return flattened_output
    
# Load the feature matrix (Shape: 100 samples, 7x7 features each)
feature_matrix = np.load("K:/Thesis/featureMatrix/4d_matrix.npy")

# Expand dimensions of feature_matrix to add channel dimension
feature_matrix = np.expand_dims(feature_matrix, axis=1)  # Shape: (100, 1, 7, 7)

# Shape: (100, 1000, 2500, 1)
label_matrix = np.load("K:/Thesis/labelMapping/label_matrix.npy", allow_pickle=True)

# Neural network architecture
conv_layer = Conv2DLayer(input_channels=1, output_channels=16, kernel_size=3, stride=2, padding=1)
fc_layer = FullyConnectedLayer(input_size=16 * 4 * 4, output_size=96)  # Adjust output size to match the number of labels (96)

# Forward pass through the network
conv_output = conv_layer.forward(feature_matrix)
fc_output = fc_layer.forward(conv_output.reshape(conv_output.shape[0], -1))

print(f"Convolutional Layer Output Shape: {conv_output.shape}")
print(f"Fully Connected Layer Output Shape: {fc_output.shape}")

audio_features = get_audio_features(conv_layer, feature_matrix)
print("Audio Features Shape:", audio_features.shape)
np.save("K:/Thesis/audio_features/audio_features.npy", audio_features)

learning_rate = 0.01
epochs = 100  # Adjust as necessary
gradient_descent(feature_matrix, label_matrix, learning_rate, epochs, conv_layer, fc_layer)

# Visualization of the feature maps
visualize_feature_maps(conv_output, num_filters=16)  # Assuming 16 filters in conv_layer

conv_layer.save_model("K:/Thesis/models/conv_model.npz")
fc_layer.save_model("K:/Thesis/models/fc_model.npz")