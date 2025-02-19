import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
# import settings_labels
# import settings_convert

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

# Load the audio features generated by the first model
audio_features = np.load("K:/Thesis/audio_features/audio_features.npy")

# Check Shapes of Matrices
print("Audio Features Shape:", audio_features.shape)
print("Label Matrix Shape:", label_matrix.shape)

# Neural Network Architecture
input_size = audio_features.shape[1]  # Number of features in the audio features
output_size = label_matrix.shape[1]   # Number of unique synthesizer settings
fc_layer = FullyConnectedLayer(input_size, output_size)

# Training Parameters
learning_rate = 0.01
epochs = 100

# Train the Network using the audio features and label matrix
gradient_descent(audio_features, label_matrix, learning_rate, epochs, fc_layer)


# Function to convert neural network output to synthesizer settings
def nn_output_to_synthesizer_settings(nn_output):
    expected_length = 38
    if len(nn_output) < expected_length:  # Replace expected_length with the actual expected length
        raise ValueError("nn_output is too short or empty")
    
    # Categorical Settings
    oscillator_index = np.argmax(nn_output[0:4])
    oscillator_types = ['Sine', 'Saw', 'Square', 'Triangle']
    oscillator = oscillator_types[oscillator_index]

    lfo_freq_index = np.argmax(nn_output[4:11])
    lfo_freq_values = ['1/1', '1/2', '1/4', '1/8', '1/16', '1/32', '1/64']
    lfo_freq = lfo_freq_values[lfo_freq_index]

    delay_freq_index = np.argmax(nn_output[11:18])
    delay_freq_values = ['1/1', '1/2', '1/4', '1/8', '1/16', '1/32', '1/64']
    delay_freq = delay_freq_values[delay_freq_index]

    distortion_type_index = np.argmax(nn_output[18:20])
    distortion_types = ['Soft Clipping', 'Hard Clipping']
    distortion_type = distortion_types[distortion_type_index]

    # Continuous Settings (assuming the original range and applying scaling if needed)
    tune = nn_output[20] * 900 + 100  # Scaling output back to range 100-1000
    transposition = nn_output[21] * 10  # Original range 0-10

    reverb_feedback = nn_output[22]  # Assuming no scaling is needed
    reverb_dampening = nn_output[23]
    reverb_mix = nn_output[24]

    attack = nn_output[25]
    decay = nn_output[26]
    sustain = nn_output[27]
    release = nn_output[28]

    distortion_drive = nn_output[29]
    distortion_mix = nn_output[30]

    filter_drive = nn_output[31]
    filter_envelope_depth = nn_output[32]
    filter_key_track = nn_output[33]

    delay_feedback = nn_output[34]
    delay_mix = nn_output[35]

    # Construct the settings dictionary
    settings = {
        'Oscillator': oscillator,
        'LFO Frequency': lfo_freq,
        'Delay Frequency': delay_freq,
        'Distortion Type': distortion_type,
        'Tune': tune,
        'Transposition': transposition,
        'Reverb Feedback': reverb_feedback,
        'Reverb Dampening': reverb_dampening,
        'Reverb Mix': reverb_mix,
        'Attack': attack,
        'Decay': decay,
        'Sustain': sustain,
        'Release': release,
        'Distortion Drive': distortion_drive,
        'Distortion Mix': distortion_mix,
        'Filter Drive': filter_drive,
        'Filter Envelope Depth': filter_envelope_depth,
        'Filter Key Track': filter_key_track,
        'Delay Feedback': delay_feedback,
        'Delay Mix': delay_mix
    }

    return settings

def process_user_input(user_input):
    # Split the user input text by commas and strip spaces
    labels = [label.strip() for label in user_input.split(',')]
    return labels

def adjust_settings_based_on_input(user_description, settings):
    user_description_lower = user_description.lower()

    # Adjust settings based on keywords in user description
    if 'reverb' not in user_description_lower:
        settings['Reverb Feedback'] = 0  # Default value when 'reverb' not mentioned
        settings['Reverb Dampening'] = 0  # Default value
        settings['Reverb Mix'] = 0  # Default value

    if 'distortion' not in user_description_lower:
        settings['Distortion Type'] = 'None'  # No distortion
        settings['Distortion Drive'] = 0  # Default value
        settings['Distortion Mix'] = 0  # Default value

    if 'bass' in user_description_lower:
        settings['Filter Drive'] = 0.7  # Example value for bass
        settings['Filter Envelope Depth'] = 0.5  # Example value

    if 'digital' in user_description_lower:
        settings['Oscillator'] = 'Saw'  # Example choice for digital sound
        settings['Tune'] = 500  # Mid-range tuning for digital sound

    if 'analog' in user_description_lower:
        settings['Oscillator'] = 'Triangle'  # Example choice for analog sound
        settings['Tune'] = 300  # Lower tuning for analog sound

    if 'low passed' in user_description_lower:
        settings['Filter Key Track'] = 0.3  # Example value for low-pass filter

    if 'sine wave' in user_description_lower:
        settings['Oscillator'] = 'Sine'  # Sine wave oscillator

    if 'square wave' in user_description_lower:
        settings['Oscillator'] = 'Square'  # Square wave oscillator

    # Mono Low Frequency Oscillator - Frequency
    if any(freq in user_description_lower for freq in ['1/1', '1/2', '1/4', '1/8', '1/16', '1/32', '1/64']):
        settings['LFO Frequency'] = [freq for freq in ['1/1', '1/2', '1/4', '1/8', '1/16', '1/32', '1/64'] if freq in user_description_lower][0]

    # Delay - Frequency
    if any(freq in user_description_lower for freq in ['1/1', '1/2', '1/4', '1/8', '1/16', '1/32', '1/64']):
        settings['Delay Frequency'] = [freq for freq in ['1/1', '1/2', '1/4', '1/8', '1/16', '1/32', '1/64'] if freq in user_description_lower][0]

    return settings

def get_user_input():
    user_input = input("Enter a description of the sound you want to generate: ")
    return user_input

user_description = get_user_input()

def predict_synthesizer_settings(user_description):
    processed_input = process_user_input(user_description)

    # Convert processed_input to audio features using the first model
    # This step depends on how you integrate the first model
    # For now, using a placeholder random vector
    new_audio_features = np.random.rand(1, input_size)  # Replace with actual audio features

    # Predict output using the neural network
    predicted_output = fc_layer.forward(new_audio_features)[0]

    # Convert NN output to synthesizer settings
    predicted_settings = nn_output_to_synthesizer_settings(predicted_output)

    # Adjust settings based on user input
    adjusted_settings = adjust_settings_based_on_input(user_description, predicted_settings)

    print("Predicted Synthesizer Settings:", adjusted_settings)

# Example execution with the preset description
predict_synthesizer_settings(user_description)






# Split your data into training, validation, and testing sets
feature_matrix_train, feature_matrix_temp, label_matrix_train, label_matrix_temp = train_test_split(feature_matrix, label_matrix, test_size=0.2, random_state=42)

feature_matrix_val, feature_matrix_test, label_matrix_val, label_matrix_test = train_test_split(feature_matrix_temp, label_matrix_temp, test_size=0.5, random_state=42)

# Determine the number of features in your input data
input_size = feature_matrix_val.shape[1] * feature_matrix_val.shape[2]

# Update the initialization of your FullyConnectedLayer
fc_layer = FullyConnectedLayer(input_size=input_size, output_size=output_size)


# Validation
fc_output_val = fc_layer.forward(feature_matrix_val.reshape(feature_matrix_val.shape[0], -1))

# Calculate validation loss
validation_loss = binary_crossentropy(label_matrix_val, fc_output_val)
print(f"Validation Loss: {validation_loss}")

# Testing
fc_output_test = fc_layer.forward(feature_matrix_test.reshape(feature_matrix_test.shape[0], -1))

# Calculate testing loss
test_loss = binary_crossentropy(label_matrix_test, fc_output_test)
print(f"Testing Loss: {test_loss}")