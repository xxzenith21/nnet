import numpy as np

# Activation function (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Mean Squared Error (MSE) loss
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Initialize weights
def initialize_weights(input_size, output_size):
    return np.random.randn(input_size, output_size) * 0.01

# Forward pass
def forward_pass(X, weights):
    return sigmoid(np.dot(X, weights))

# Backward pass (gradient descent)
def backward_pass(X, y_true, y_pred, weights, learning_rate):
    error = y_true - y_pred
    delta = error * sigmoid_derivative(y_pred)
    weights += learning_rate * np.dot(X.T, delta)
    return weights

# Training the neural network
def train_neural_network(X_train, y_train, epochs, learning_rate):
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]

    # Initialize weights
    weights = initialize_weights(input_size, output_size)

    for epoch in range(epochs):
        # Forward pass
        y_pred = forward_pass(X_train, weights)

        # Calculate loss
        loss = mse_loss(y_train, y_pred)

        # Backward pass (update weights)
        weights = backward_pass(X_train, y_train, y_pred, weights, learning_rate)

        # Print loss for every 100 epochs
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    return weights

# Example usage
# Assuming X_train is your feature matrix and y_train is your label matrix
# X_train and y_train should be NumPy arrays

# Create some dummy data for demonstration
X_train = np.random.rand(100, 10)  # Replace this with your actual feature matrix
y_train = np.random.randint(0, 2, size=(100, 1))  # Replace this with your actual label matrix

# Train the neural network
trained_weights = train_neural_network(X_train, y_train, epochs=1000, learning_rate=0.01)

# Test the neural network on new data
# Perform a forward pass with the trained weights
# You can then use the output for making predictions
