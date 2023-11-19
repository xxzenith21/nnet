import numpy as np

# Neural Network Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -709, 709)))

def binary_crossentropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

# Fully Connected Layer Class
class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros((1, output_size))

    def forward(self, X):
        return sigmoid(np.dot(X, self.weights) + self.bias)

# Back Propagation and Gradient Descent Functions
def back_prop(fc_layer, X, Y, fc_output):
    m = Y.shape[0]
    dA2 = fc_output - Y
    dZ2 = dA2 * sigmoid_derivative(fc_output)
    dW2 = np.dot(X.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    return dW2, db2

def update_weights(layer, dw, db, learning_rate):
    layer.weights -= learning_rate * dw
    layer.bias -= learning_rate * db

def gradient_descent(X, Y, alpha, epochs, fc_layer):
    for epoch in range(epochs):
        fc_output = fc_layer.forward(X)
        loss = binary_crossentropy(Y, fc_output)
        dW2, db2 = back_prop(fc_layer, X, Y, fc_output)
        update_weights(fc_layer, dW2, db2, alpha)
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

# Load Feature Matrix and Label Matrix
feature_matrix = np.load("K:/Thesis/synth_settings/settings_features_matrix/settings_4d_matrix.npy")
label_matrix = np.load("K:/Thesis/synth_settings/settings_label_matrix/settings_label_matrix.npy", allow_pickle=True)

# Flatten the feature matrix if it's not already 2D
flattened_feature_matrix = feature_matrix.reshape(feature_matrix.shape[0], -1)

# Check Shapes of Matrices
print("Flattened Feature Matrix Shape:", flattened_feature_matrix.shape)
print("Label Matrix Shape:", label_matrix.shape)

# Neural Network Architecture
input_size = flattened_feature_matrix.shape[1]  # Number of features in flattened matrix
output_size = label_matrix.shape[1]  # Number of unique synthesizer settings
fc_layer = FullyConnectedLayer(input_size, output_size)

# Training Parameters
learning_rate = 0.01
epochs = 100

# Train the Network
gradient_descent(flattened_feature_matrix, label_matrix, learning_rate, epochs, fc_layer)