import numpy as np

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
                self.conv_output[:, :, i, j] = np.sum(receptive_field * self.weights, axis=(2, 3, 4)) + self.bias

        return self.conv_output

class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros((1, output_size))

    def forward(self, X):
        self.input = X
        self.fc_output = np.dot(X, self.weights) + self.bias
        return self.fc_output

# Example usage
# Assuming X_train is your feature matrix and y_train is your label matrix
# X_train and y_train should be NumPy arrays

# Create some dummy data for demonstration
X_train = np.random.rand(100, 3, 10, 10)  # Replace this with your actual feature matrix
y_train = np.random.randint(0, 2, size=(100, 1))  # Replace this with your actual label matrix

# Initialize the convolutional layer
conv_layer = Conv2DLayer(input_channels=3, output_channels=16, kernel_size=3, stride=1, padding=1)

# Forward pass through the convolutional layer
conv_output = conv_layer.forward(X_train)

# Initialize the fully connected layer
fc_layer = FullyConnectedLayer(input_size=16 * 10 * 10, output_size=1)

# Flatten the output of the convolutional layer
flattened_output = conv_output.reshape(X_train.shape[0], -1)

# Forward pass through the fully connected layer
fc_output = fc_layer.forward(flattened_output)

# Now you can use fc_output for making predictions or further training
