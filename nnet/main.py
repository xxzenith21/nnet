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
        # Assume X is a 4D tensor (batch_size, input_channels, height, width)
        batch_size, input_channels, input_height, input_width = X.shape
        kernel_size = self.weights.shape[2]
        output_channels = self.weights.shape[0]

        # Compute output dimensions
        output_height = (input_height - kernel_size + 2 * self.padding) // self.stride + 1
        output_width = (input_width - kernel_size + 2 * self.padding) // self.stride + 1

        # Apply convolution operation
        self.input = X
        self.conv_output = np.zeros((batch_size, output_channels, output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + kernel_size
                w_start = j * self.stride
                w_end = w_start + kernel_size

                receptive_field = X[:, :, h_start:h_end, w_start:w_end]
                self.conv_output[:, :, i, j] = np.sum(receptive_field * self.weights, axis=(1, 2, 3)) + self.bias

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
    return spectrogram

def prepare_data(input_folder, label_file_path):
    X = []
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith("_spectrogram.png"):
            file_path = os.path.join(input_folder, filename)
            spectrogram = load_spectrogram(file_path)
            spectrogram = spectrogram[:, :, None]  # Add a channel dimension
            X.append(spectrogram)

    X = np.stack(X, axis=0)

    # Load label matrix
    y = np.load(label_file_path, allow_pickle=True)
    
    # Print shapes for debugging
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")

    # Ensure the number of samples matches between features and labels
    assert X.shape[0] == y.shape[0], "Mismatch in the number of samples between features and labels."

    return X, y





# Path to the label matrix file
label_file_path = "K:/Thesis/labelMapping/label_mapping.npy"

# Load and Prepare Data
X_train, y_train = prepare_data("K:/Thesis/spectro", label_file_path)

# Print shapes to verify
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
# Initialize the convolutional layer
# Modify input_channels based on your spectrogram's channel (1 for grayscale, 3 for RGB)
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

# Now fc_output contains the sigmoid-activated outputs for each label