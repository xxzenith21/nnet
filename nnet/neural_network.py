import numpy as np

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
        return output

class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros((1, output_size))

    def forward(self, X):
        X_flattened = X.reshape(X.shape[0], -1)
        output = np.dot(X_flattened, self.weights) + self.bias
        return output
    
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

# Output shapes
print(f"Shape of Convolutional Output: {conv_output.shape}")
print(f"Shape of Fully Connected Layer Output: {fc_output.shape}")