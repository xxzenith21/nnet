import os
import csv
import random
from pyo import *

dataset_output = "K:/Thesis/synth_settings/synth_settings_dataset"

# Function to clear the contents of a folder
def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

# Clear the contents of the output folder
clear_folder(dataset_output)


def generate_synthesizer_parameters():
    # Random selection of parameters
    params = {
        "oscillator_type": random.choice(['Sine', 'Saw', 'Square', 'Triangle', ]),
        "tune": random.uniform(0, 880),  # Oscillator tuning range
        "transposition": random.uniform(-12, 12),  # Oscillator transposition range

        "lfo_freq": random.choice([1/1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64]),  # LFO frequencies

        "delay_freq": random.choice([1/1, 1/2, 1.4, 1/8, 1/16, 1/32, 1/64]),  # Delay frequencies
        "delay_feedback": random.uniform(0, 101),  # Delay feedback coefficient
        "delay_mix": random.uniform(1, 101),  # Delay mix level
        "delay_time": random.uniform(0.01, 0.5),

        "distortion_type": random.choice(['Soft Clipping', 'Hard Clipping']),
        "distortion_drive": random.uniform(1, 101),  # Distortion drive amount
        "distortion_mix": random.uniform(1, 101),  # Distortion mix level

        "reverb_size": random.uniform(1, 101),  # Reverb room size
        "reverb_damp": random.uniform(1, 101),  # Reverb dampening
        "reverb_mix": random.uniform(1, 101),  # Reverb mix level

        "attack": random.uniform(1, 101),  # ADSR attack range
        "decay": random.uniform(1, 101),  # ADSR decay range
        "sustain": random.uniform(1, 101),  # ADSR sustain level
        "release": random.uniform(1, 101),  # ADSR release range

        "filter_drive": random.uniform(1, 101),  # Filter drive amount
        "filter_envelope_depth": random.uniform(1, 101),  # Filter envelope depth
        "filter_freq": random.uniform(500, 5000),  # Filter cutoff frequency in Hz
        "filter_key_track": random.uniform(1, 101)  # Filter key track
    }

    # Constrained ADSR parameters
    total_adsr_time = 2  # Total time for ADSR in seconds
    attack = random.uniform(0, total_adsr_time / 4)
    decay = random.uniform(0, (total_adsr_time - attack) / 3)
    sustain_level = random.uniform(0, 1)  # Sustain is a level, not a time
    release = random.uniform(0, total_adsr_time - attack - decay)

    params.update({
        "attack": attack,
        "decay": decay,
        "sustain": sustain_level,
        "release": release
    })

    return params

def synthesize_sound(params, file_path, server):
    s = Server().boot()
    s.start()

    # Oscillator selection
    if params['oscillator_type'] == 'Sine':
        osc = Sine(freq=params['tune'], mul=0.3)
    elif params['oscillator_type'] == 'Saw':
        osc = SuperSaw(freq=params['tune'], detune=params['transposition'], mul=0.3)
    elif params['oscillator_type'] == 'Square':
        square_table = SquareTable()
        osc = Osc(table=square_table, freq=params['tune'], mul=0.3)
    elif params['oscillator_type'] == 'Triangle':
        triangle_table = TriangleTable()
        osc = Osc(table=triangle_table, freq=params['tune'], mul=0.3)
    else:
        raise ValueError("Invalid oscillator type")

    # ADSR Envelope
    env = Adsr(attack=params['attack'], decay=params['decay'], sustain=params['sustain'], release=params['release'], dur=2, mul=0.5)
    osc.mul = env

    # Low Frequency Oscillator (LFO)
    lfo = Sine(freq=params['lfo_freq']).range(-0.5, 0.5)
    osc.freq = osc.freq + lfo

    # Delay
    delay = Delay(osc, delay=params['delay_time'], feedback=params['delay_feedback'], maxdelay=1)

    # Distortion
    if params['distortion_type'] == 'Soft Clipping':
        dist = Disto(delay, drive=params['distortion_drive'], slope=0.5)
    elif params['distortion_type'] == 'Hard Clipping':
        dist = Disto(delay, drive=params['distortion_drive'], slope=0.5)  # Initialize with Disto before Clip
        dist = Clip(dist, min=-.5, max=.5)
    else:
        dist = delay  # Default to delay if no distortion type is specified

    # Reverb
    reverb = Freeverb(dist, size=params['reverb_size'], damp=params['reverb_damp'], bal=params['reverb_mix'])

    # Filter
    filt = ButLP(reverb, freq=params['filter_freq'])

    # Create Record object to start recording
    rec = Record(filt, filename=file_path, fileformat=0, sampletype=0)

     # Start the envelope and let the sound play
    env.play()

    # Start server recording
    server.recstart()

    # Adjust the duration to ensure the total sound duration does not exceed 2 seconds
    sound_duration = min(2, params['release'] + 0.5)  # Adjust the additional time if needed
    time.sleep(sound_duration)

    # Stop server recording
    server.recstop()

    # Stop the Record object
    rec.stop()

def create_dataset(num_samples, dataset_output):
    s = Server().boot()
    s.start()

    if not os.path.exists(dataset_output):
        os.makedirs(dataset_output)

    labels_file = os.path.join(dataset_output, 'labels.csv')

    with open(labels_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Parameters'])  # Header

        for i in range(num_samples):
            params = generate_synthesizer_parameters()
            file_name = f"sound_{i:04d}.wav"  # Unique file name
            file_path = os.path.join(dataset_output, file_name)

            synthesize_sound(params, file_path, s)

            # Convert params dict to a string format for CSV
            params_str = ', '.join(f"{k}={v}" for k, v in params.items())
            writer.writerow([file_name, params_str])

    s.stop()

if __name__ == "__main__":
    create_dataset(700, dataset_output)