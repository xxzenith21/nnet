import os
import csv
import random
from pyo import *

dataset_output = "K:/Thesis/synth_settings_dataset"

def generate_synthesizer_parameters():
    # Random selection of parameters
    params = {
        "oscillator_type": random.choice(['Sine', 'Saw', 'Square', 'Triangle']),
        "tune": random.uniform(220, 880),  # Oscillator tuning range
        "transposition": random.uniform(-12, 12),  # Oscillator transposition range

        "lfo_freq": random.choice([1, 0.5, 0.25, 0.125, 1/16, 1/32, 1/64]),  # LFO frequencies

        "delay_freq": random.choice([1, 0.5, 0.25, 0.125, 1/16, 1/32, 1/64]),  # Delay frequencies
        "delay_feedback": random.uniform(0.0, 0.9),  # Delay feedback coefficient
        "delay_mix": random.uniform(0.1, 0.9),  # Delay mix level
        "delay_time": random.uniform(0.01, 0.5),

        "distortion_type": random.choice(['Soft Clipping', 'Hard Clipping']),
        "distortion_drive": random.uniform(0.1, 1),  # Distortion drive amount
        "distortion_mix": random.uniform(0.1, 0.9),  # Distortion mix level

        "reverb_feedback": random.uniform(0.1, 0.9),  # Reverb feedback
        "reverb_dampening": random.uniform(0.1, 0.5),  # Reverb dampening
        "reverb_mix": random.uniform(0.1, 0.9),  # Reverb mix level

        "attack": random.uniform(0.01, 1),  # ADSR attack range
        "decay": random.uniform(0.01, 1),  # ADSR decay range
        "sustain": random.uniform(0.1, 1),  # ADSR sustain level
        "release": random.uniform(0.01, 1),  # ADSR release range

        "filter_drive": random.uniform(0.1, 1),  # Filter drive amount
        "filter_envelope_depth": random.uniform(0.1, 1),  # Filter envelope depth
        "filter_key_track": random.uniform(0.1, 1)  # Filter key track
    }
    return params

def synthesize_sound(params, file_path):
    s = Server().boot()
    s.start()

    # Oscillator selection
    if params['oscillator_type'] == 'Sine':
        osc = Sine(freq=params['tune'], mul=0.3)
    elif params['oscillator_type'] == 'Saw':
        osc = SuperSaw(freq=params['tune'], detune=params['transposition'], mul=0.3)
    elif params['oscillator_type'] == 'Square':
        osc = SquareTable().lookup(SigTo(params['tune'], params['transposition']))
    elif params['oscillator_type'] == 'Triangle':
        osc = TriangleTable().lookup(SigTo(params['tune'], params['transposition']))
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
        dist = Clip(dist, min=-.5, max=.5)

    # Reverb
    reverb = Freeverb(dist, size=params['reverb_size'], damp=params['reverb_damp'], bal=params['reverb_mix'])

    # Filter
    filt = ButLP(reverb, freq=params['filter_freq'])

    # Output the final processed sound
    filt.out()

    # Start ADSR envelope
    env.play()

    # Record the final sound
    rec = Record(filt, filename=file_path, fileformat=0, sampletype=0)
    rec.record()

    # Wait for the sound to fully play
    server.recstart()  # Start recording
    wait(params['release'] + 1)
    server.recstop()   # Stop recording

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
    create_dataset(1000, dataset_output)