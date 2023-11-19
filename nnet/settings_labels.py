import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Continuous parameters and their ranges for normalization
continuous_params = [
    'tune', 'transposition', 'reverb_feedback', 'reverb_dampening', 'reverb_mix',
    'attack', 'decay', 'sustain', 'release', 'distortion_drive', 'distortion_mix',
    'filter_drive', 'filter_envelope_depth', 'filter_key_track', 'delay_feedback', 'delay_mix'
]

continuous_ranges = {
    'tune': (100, 1000),  # Example range, adjust as needed
    'transposition': (0, 10),  # Example range, adjust as needed
    'reverb_feedback': (0, 1),  # Assuming a normalized range
    'reverb_dampening': (0, 1),  # Assuming a normalized range
    'reverb_mix': (0, 1),  # Assuming a normalized range
    'attack': (0.001, 1),  # Time in seconds, adjust as needed
    'decay': (0.001, 1),  # Time in seconds, adjust as needed
    'sustain': (0, 1),  # Assuming a normalized range
    'release': (0.001, 1),  # Time in seconds, adjust as needed
    'distortion_drive': (0, 1),  # Assuming a normalized range
    'distortion_mix': (0, 1),  # Assuming a normalized range
    'filter_drive': (0, 1),  # Assuming a normalized range
    'filter_envelope_depth': (0, 1),  # Assuming a normalized range
    'filter_key_track': (0, 1),  # Assuming a normalized range
    'delay_feedback': (0, 1),  # Assuming a normalized range
    'delay_mix': (0, 1)  # Assuming a normalized range
}

# Categorical parameters and their categories
categorical_params = {'oscillator_type': ['Sine', 'Saw', 'Square', 'Triangle'], 'lfo_freq': ['1/1', '1/2', '1/4', '1/8', '1/16', '1/32', '1/64'], 'delay_freq': ['1/1', '1/2', '1/4', '1/8', '1/16', '1/32', '1/64'], 'distortion_type': ['Soft Clipping', 'Hard Clipping']}

def parse_and_encode(label_str):
    label_dict = dict(item.split("=") for item in label_str.split(", "))
    
    continuous_values = [float(label_dict[param]) for param in continuous_params]
    categorical_values = {param: label_dict[param] for param in categorical_params.keys()}

    # Normalize continuous values
    scaler = MinMaxScaler()
    scaler.fit([[continuous_ranges[param][0], continuous_ranges[param][1]] for param in continuous_params])
    normalized_values = scaler.transform([continuous_values])[0]

    # One-hot encode categorical values
    encoded_categorical = []
    encoder = OneHotEncoder()
    for param, categories in categorical_params.items():
        encoder.fit(np.array(categories).reshape(-1, 1))
        encoded = encoder.transform(np.array([categorical_values[param]]).reshape(-1, 1)).toarray()
        encoded_categorical.extend(encoded[0])

    # Combine into a single vector
    return np.concatenate((normalized_values, encoded_categorical))
