import numpy as np

class SynthesizerConfig:
    def __init__(self):
        # Categorical parameters with default values
        self.oscillator_waveform = 'sine'
        self.lfo_frequency = '1/1'
        self.delay_frequency = '1/1'
        self.distortion_type = 'soft clipping'

        # Continuous parameters with default values (assuming normalized for simplicity)
        self.oscillator_tune = 0.5  # Assuming 0 is min, 1 is max
        self.oscillator_transposition = 0.5  # Assuming 0 is min, 1 is max
        self.oscillator_tune = 0.5
        self.reverb_feedback = 0.5
        self.reverb_dampening = 0.5
        self.reverb_mix = 0.5
        self.amplitude_attack = 0.5
        self.amplitude_decay = 0.5
        self.amplitude_sustain = 0.5
        self.amplitude_release = 0.5
        self.distortion_drive = 0.5
        self.distortion_mix = 0.5
        self.filter_drive = 0.5
        self.filter_envelope_depth = 0.5
        self.filter_key_track = 0.5
        self.delay_feedback = 0.5
        self.delay_mix = 0.5

    def to_feature_vector(self):
        # Convert categorical parameters to one-hot vectors
        waveform_vector = self.one_hot(['sine', 'saw', 'square', 'triangle'], self.oscillator_waveform)
        lfo_freq_vector = self.one_hot(['1/1', '1/2', '1/4', '1/8', '1/16', '1/32', '1/64'], self.lfo_frequency)
        delay_freq_vector = self.one_hot(['1/1', '1/2', '1/4', '1/8', '1/16', '1/32', '1/64'], self.delay_frequency)
        distortion_type_vector = self.one_hot(['soft clipping', 'hard clipping'], self.distortion_type)

        # Combine one-hot vectors and normalized continuous parameters into a single feature vector
        feature_vector = np.concatenate([
            waveform_vector,
            lfo_freq_vector,
            delay_freq_vector,
            distortion_type_vector,
            [
                self.oscillator_tune, 
                self.oscillator_transposition,
                self.reverb_feedback,
                self.reverb_dampening,
                self.reverb_mix,
                self.amplitude_attack,
                self.amplitude_decay,
                self.amplitude_sustain,
                self.amplitude_release,
                self.distortion_drive,
                self.distortion_mix,
                self.filter_drive,
                self.filter_envelope_depth,
                self.filter_key_track,
                self.delay_feedback,
                self.delay_mix
            ]
        ])

        return feature_vector

    @staticmethod
    def one_hot(options, value):
        """Create a one-hot encoded vector based on the given options and the selected value."""
        return np.array([1 if option == value else 0 for option in options])

# Example usage:
synth_config = SynthesizerConfig()
synth_config.oscillator_waveform = 'saw'

feature_vector = synth_config.to_feature_vector()
print(feature_vector)  # This will print out the feature vector representation of the settings
