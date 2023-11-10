import numpy as np
import matplotlib.pyplot as plt
import os

class Conv2DLayer:
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding):
        self.weights = np.random.randn(output_channels, input_channels, kernel_size, kernel_size)
        self.bias = np.zeros((1, output_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, X):
        batch_size, input_channels, input_height, input_width = X.shape
        kernel_size = self.weights.shape[2]
        output_channels = self.weights.shape[0]

        # Compute output dimensions
        output_height = (input_height - kernel_size + 2 * self.padding) // self.stride + 1
        output_width = (input_width - kernel_size + 2 * self.padding) // self.stride + 1

        # Initialize the output tensor
        self.conv_output = np.zeros((batch_size, output_channels, output_height, output_width))

        # Apply convolution operation
        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + kernel_size
                w_start = j * self.stride
                w_end = w_start + kernel_size

                # Extract the receptive field
                receptive_field = X[:, :, h_start:h_end, w_start:w_end]

                # Perform element-wise multiplication and sum over the kernel dimensions and input channels
                # The shape of conv_sum should be (batch_size, output_channels)
                conv_sum = np.sum(receptive_field[:, None, :, :] * self.weights, axis=(2, 3, 1))

                # Add bias
                # Bias is broadcasted automatically along the batch size dimension
                self.conv_output[:, :, i, j] = conv_sum + self.bias[0, :]

        return self.conv_output

    
class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros((1, output_size))

    def forward(self, X):
        self.input = X
        self.fc_output = np.dot(X, self.weights) + self.bias
        return self.sigmoid(self.fc_output)  # Apply sigmoid activation function

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

# Function to Load a Single Spectrogram
def load_spectrogram(file_path):
    spectrogram = plt.imread(file_path)
    spectrogram = spectrogram / spectrogram.max()
    # Add a channel dimension if it's grayscale
    if len(spectrogram.shape) == 2:
        spectrogram = spectrogram[:, :, np.newaxis]  # Adds a single channel
    return spectrogram

def prepare_data(spectrogram_folder, label_matrix_file):
    X = []
    for filename in sorted(os.listdir(spectrogram_folder)):
        if filename.endswith("_spectrogram.png"):
            file_path = os.path.join(spectrogram_folder, filename)
            spectrogram = load_spectrogram(file_path)
            X.append(spectrogram)
    X = np.stack(X, axis=0)  # This creates a 4D tensor

    # Load label matrix
    y = np.load(label_matrix_file, allow_pickle=True)

    return X, y

# Paths to Folders
spectrogram_folder = "K:/Thesis/spectro"
label_matrix_file = "K:/Thesis/labelMapping/label_mapping.npy"

# Load and Prepare Data
X_train, y_train = prepare_data(spectrogram_folder, label_matrix_file)

# Initialize the convolutional layer
conv_layer = Conv2DLayer(input_channels=1, output_channels=16, kernel_size=3, stride=1, padding=1)

# Forward pass through the convolutional layer
conv_output = conv_layer.forward(X_train)

# Compute the size for the fully connected layer's input
fc_input_size = np.prod(conv_output.shape[1:])

# Initialize the fully connected layer
fc_layer = FullyConnectedLayer(input_size=fc_input_size, output_size=10)

# Flatten the output of the convolutional layer
flattened_output = conv_output.reshape(X_train.shape[0], -1)

# Forward pass through the fully connected layer
fc_output = fc_layer.forward(flattened_output)

# Print shapes to verify
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")