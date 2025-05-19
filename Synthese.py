import streamlit as st
import numpy as np
from scipy.signal import lfilter, freqz
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import io

# Global sampling rate
SAMPLE_RATE = 22050

# Function to generate waveforms with overlapping
def generate_waveform(wave_type, frequencies, durations, start_times):
    # Calculate the total duration of the waveform
    total_duration = max([start + duration for start, duration in zip(start_times, durations)])
    waveform = np.zeros(int(SAMPLE_RATE * total_duration))  # Initialize the waveform array

    # Generate the waveform for each frequency, duration, and start time
    for frequency, duration, start_time in zip(frequencies, durations, start_times):
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)  # Time vector for the note
        if wave_type == 'Sine':
            wave = np.sin(2 * np.pi * frequency * t)  # Generate sine wave
        elif wave_type == 'Triangle':
            wave = 2 * np.abs(2 * (t * frequency % 1) - 1) - 1  # Generate triangle wave
        elif wave_type == 'Sawtooth':
            wave = 2 * (t * frequency % 1) - 1  # Generate sawtooth wave
        elif wave_type == 'Square':
            wave = np.sign(np.sin(2 * np.pi * frequency * t))  # Generate square wave
        else:
            wave = np.zeros_like(t)  # Default to silence if wave type is unknown

        # Add the generated wave to the waveform at the correct start time
        start_sample = int(SAMPLE_RATE * start_time)
        waveform[start_sample:start_sample + len(wave)] += wave

    return waveform

# Simple LFO
def apply_lfo(signal, rate, depth, wave_type='Sine'):
    # Generate the LFO waveform
    lfo = 1 + depth * generate_waveform(wave_type, [rate], [len(signal) / SAMPLE_RATE], [0])
    # Modulate the signal amplitude using the LFO
    return signal * lfo

# Function to calculate biquad filter coefficients
def biquad(cutoff, q, filter_type='low'):
    # Calculate normalized frequency and alpha for the filter
    omega = 2 * np.pi * cutoff / SAMPLE_RATE
    alpha = np.sin(omega) / (2 * q)

    # Calculate coefficients for low-pass or high-pass filter
    if filter_type == 'low':
        b0 = (1 - np.cos(omega)) / 2
        b1 = 1 - np.cos(omega)
        b2 = (1 - np.cos(omega)) / 2
        a0 = 1 + alpha
        a1 = -2 * np.cos(omega)
        a2 = 1 - alpha
    elif filter_type == 'high':
        b0 = (1 + np.cos(omega)) / 2
        b1 = -(1 + np.cos(omega))
        b2 = (1 + np.cos(omega)) / 2
        a0 = 1 + alpha
        a1 = -2 * np.cos(omega)
        a2 = 1 - alpha

    # Normalize coefficients
    b = [b0 / (a0 * 10), b1 / (a0 * 10), b2 / (a0 * 10)]
    a = [1, a1 / a0, a2 / a0]
    return b, a

# Function to plot poles and zeros of the filter
def plot_poles_zeros(b, a):
    # Calculate poles and zeros of the filter
    z = np.roots(b)
    p = np.roots(a)

    # Plot the poles and zeros on the complex plane
    from matplotlib import patches
    fig, ax = plt.subplots()
    unit_circle = patches.Circle((0, 0), radius=1, fill=False, color='black', ls='dotted')
    ax.add_patch(unit_circle)
    ax.plot(np.real(z), np.imag(z), 'go', label='Zeros')  # Plot zeros
    ax.plot(np.real(p), np.imag(p), 'rx', label='Poles')  # Plot poles
    ax.set_xlim((-1.5, 1.5))
    ax.set_ylim((-1.5, 1.5))
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_title("Filter Poles and Zeros")
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.legend()
    ax.grid()
    return fig

# Apply ADSR envelope
def apply_adsr(signal, attack, decay, sustain, release):
    # Calculate the total duration of the signal in seconds
    total_duration = len(signal) / SAMPLE_RATE

    # Ensure A + D + R does not exceed the total duration
    if attack + decay + release > total_duration:
        release = max(0, total_duration - (attack + decay))  # Adjust release to fit within the total duration

    # Calculate the number of samples for each ADSR phase
    attack_samples = int(SAMPLE_RATE * attack)
    decay_samples = int(SAMPLE_RATE * decay)
    release_samples = int(SAMPLE_RATE * release)

    # Initialize the envelope
    envelope = np.zeros(len(signal))

    # Generate the ADSR envelope
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)  # Attack phase
    decay_end = attack_samples + decay_samples
    envelope[attack_samples:decay_end] = np.linspace(1, sustain, decay_samples)  # Decay phase
    envelope[decay_end:len(signal) - release_samples] = sustain  # Sustain phase
    envelope[len(signal) - release_samples:] = np.linspace(sustain, 0, release_samples)  # Release phase

    # Apply the envelope to the signal
    return signal * envelope

# Function to convert a numpy signal to a WAV file
def numpy_to_wav(signal):
    signal = np.int16(signal / np.max(np.abs(signal)) * 32767)
    buf = io.BytesIO()
    wav.write(buf, SAMPLE_RATE, signal)
    buf.seek(0)
    return buf

# Function to apply an ADSR envelope and an LFO to the filter cutoff
def apply_combined_adsr_lfo_to_cutoff(cutoff, lfo_rate, lfo_depth, lfo_wave_type, attack, decay, sustain, release, total_duration):
    length = int(SAMPLE_RATE * total_duration)
    # Create a constant signal at 1 for modulation
    base = np.ones(length)
    # Apply ADSR to the constant signal
    envelope = apply_adsr(base, attack, decay, sustain, release)
    # Apply LFO to the envelope
    lfo_enveloped = apply_lfo(envelope, lfo_rate, lfo_depth, lfo_wave_type)
    # Multiply by the cutoff value
    return cutoff * lfo_enveloped

# Function to apply a static filter without LFO
def apply_static_biquad_filter(signal, cutoff, filter_type='low', filter_q=1.0):
    b, a = biquad(cutoff, filter_q, filter_type)
    return lfilter(b, a, signal)

# Apply biquad filter with modulated cutoff
def apply_dynamic_biquad_filter(signal, cutoff_lfo, filter_type='low', filter_q=1.0):
    filtered_signal = np.zeros_like(signal)
    b, a = biquad(cutoff_lfo[0], filter_q, filter_type)  # Initialize filter coefficients
    zi = np.zeros(max(len(b), len(a)) - 1)  # Initialize the filter state
    for i in range(len(signal)):
        if i > 0 and cutoff_lfo[i] != cutoff_lfo[i-1]:  # Update filter coefficients if cutoff changes
            b, a = biquad(cutoff_lfo[i], filter_q, filter_type)
            zi = lfilter(b, a, [0], zi=zi)[1]  # Reset the filter state when coefficients change
        filtered_signal[i], zi = lfilter(b, a, [signal[i]], zi=zi)  # Apply the filter with the current state
    return filtered_signal

# Function to apply an Echo effect
def apply_echo(signal, delay, decay):
    # Calculate the number of samples for the delay
    delay_samples = int(SAMPLE_RATE * delay)
    # Initialize the echo signal with additional space for the delay
    echo_signal = np.zeros(len(signal) + delay_samples)
    echo_signal[:len(signal)] = signal  # Copy the original signal

    # Add the delayed signal with decay
    for i in range(len(signal)):
        if i + delay_samples < len(echo_signal):
            echo_signal[i + delay_samples] += signal[i] * decay

    # Mix the original signal with the echo signal
    mixed_signal = echo_signal[:len(signal)] + signal * (1 - decay)

    # Return the mixed signal
    return mixed_signal

# Function to apply a Flanger effect
def apply_flanger(signal, rate, depth):
    # Initialize the flanged signal
    flanged_signal = np.zeros_like(signal)
    # Calculate the maximum delay in samples
    max_delay_samples = int(SAMPLE_RATE * depth)
    # Generate the LFO for the flanger
    lfo = (np.sin(2 * np.pi * rate * np.arange(len(signal)) / SAMPLE_RATE) + 1) / 2  # LFO oscillates between 0 and 1

    # Apply the flanger effect
    for i in range(len(signal)):
        delay_samples = int(lfo[i] * max_delay_samples)  # Calculate the delay for the current sample
        if i - delay_samples >= 0:
            flanged_signal[i] = signal[i] + signal[i - delay_samples]  # Add delayed signal
        else:
            flanged_signal[i] = signal[i]  # No delay for the first samples

    return flanged_signal

# Function to apply all transformations
def apply_transformations(signal, lfo_rate, lfo_depth, lfo_wave_type, combined_cutoff, filter_type, filter_q, attack, decay, sustain, release, echo_delay, echo_decay, flanger_rate, flanger_depth):
    lfo_signal = apply_lfo(signal, lfo_rate, lfo_depth, lfo_wave_type)
    filtered_signal = apply_dynamic_biquad_filter(lfo_signal, combined_cutoff, filter_type, filter_q)
    adsr_signal = apply_adsr(filtered_signal, attack, decay, sustain, release)
    echo_signal = apply_echo(adsr_signal, echo_delay, echo_decay)
    flanger_signal = apply_flanger(echo_signal, flanger_rate, flanger_depth)
    return flanger_signal

# Function to create centered subheaders with expandable descriptions
def centered_subheader_with_help(title, help_text):
    st.markdown(f"<h2 style='text-align: center;'>{title}</h2>", unsafe_allow_html=True)
    if help_text:
        with st.expander("ℹ️ Description"):
            st.markdown(help_text)

# Streamlit Application
st.title("🎛️ Subtractive Synthesizer")
st.info("Adjust the parameters and click 'Play Sound' to listen to your sound creation.")

# Frequencies and durations of notes for the beginning of "Für Elise"
fur_elise_frequencies = [659.25,622.25,659.25,622.25,659.25,493.88,587.33,523.25,440.0,261.63,329.63,440.0,493.88,329.63,415.3,493.88,523.25,220.0,329.63,164.81,329.63,415.30,220.0,329.63,440,0]
fur_elise_durations = [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.6,0.2,0.2,0.2,0.6,0.2,0.2,0.2,0.6,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
fur_elise_start_times = [0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,2.2,2.4,2.6,2.8,3.4,3.6,3.8,4.0,1.6,1.8,2.8,3.0,3.2,4.0,4.2,4.4]


# Available notes and their frequencies in Hz
available_notes = """
Available notes and their frequencies in Hz:
A0: 27.50, A#0/Bb0: 29.14, B0: 30.87,
C1: 32.70, C#1/Db1: 34.65, D1: 36.71, D#1/Eb1: 38.89, E1: 41.20, F1: 43.65, F#1/Gb1: 46.25, G1: 49.00, G#1/Ab1: 51.91, A1: 55.00, A#1/Bb1: 58.27, B1: 61.74,
C2: 65.41, C#2/Db2: 69.30, D2: 73.42, D#2/Eb2: 77.78, E2: 82.41, F2: 87.31, F#2/Gb2: 92.50, G2: 98.00, G#2/Ab2: 103.83, A2: 110.00, A#2/Bb2: 116.54, B2: 123.47,
C3: 130.81, C#3/Db3: 138.59, D3: 146.83, D#3/Eb3: 155.56, E3: 164.81, F3: 174.61, F#3/Gb3: 185.00, G3: 196.00, G#3/Ab3: 207.65, A3: 220.00, A#3/Bb3: 233.08, B3: 246.94,
C4: 261.63, C#4/Db4: 277.18, D4: 293.66, D#4/Eb4: 311.13, E4: 329.63, F4: 349.23, F#4/Gb4: 369.99, G4: 392.00, G#4/Ab4: 415.30, A4: 440.00, A#4/Bb4: 466.16, B4: 493.88,
C5: 523.25, C#5/Db5: 554.37, D5: 587.33, D#5/Eb5: 622.25, E5: 659.25, F5: 698.46, F#5/Gb5: 739.99, G5: 783.99, G#5/Ab5: 830.61, A5: 880.00, A#5/Bb5: 932.33, B5: 987.77,
C6: 1046.50, C#6/Db6: 1108.73, D6: 1174.66, D#6/Eb6: 1244.51, E6: 1318.51, F6: 1396.91, F#6/Gb6: 1479.98, G6: 1567.98, G#6/Ab6: 1661.22, A6: 1760.00, A#6/Bb6: 1864.66, B6: 1975.53,
C7: 2093.00, C#7/Db7: 2217.46, D7: 2349.32, D#7/Eb7: 2489.02, E7: 2637.02, F7: 2793.83, F#7/Gb7: 2959.96, G7: 3135.96, G#7/Ab7: 3322.44, A7: 3520.00, A#7/Bb7: 3729.31, B7: 3951.07,
C8: 4186.01
"""

biquad_filter = """
    Apply a biquad filter to the input signal.

    This function calculates the coefficients based on the desired filter
    characteristics and applies the filter to the input signal.

    The biquad filter is a second-order recursive linear filter that uses
    the following difference equation:

    y[n] = b0 * x[n] + b1 * x[n-1] + b2 * x[n-2] - a1 * y[n-1] - a2 * y[n-2]

    Where:
    - x[n] is the current input sample
    - x[n-1] and x[n-2] are the previous input samples
    - y[n] is the current output sample
    - y[n-1] and y[n-2] are the previous output samples
    - b0, b1, b2 are the feedforward coefficients
    - a1, a2 are the feedback coefficients

    The coefficients (b0, b1, b2, a1, a2) are calculated based on the desired
    filter characteristics such as cutoff frequency, Q factor, and gain. These
    characteristics define the filter's behavior, including its frequency
    response and stability.

    The coefficient calculation involves the following steps:
    1. Determine the normalized frequency (ω0) based on the cutoff frequency
    and sampling frequency.
    2. Calculate intermediate variables (α, cos(ω0), sin(ω0)) using
    trigonometric functions.
    3. Calculate the coefficients using standard biquad formulas for the
    specific filter type (low-pass, high-pass, band-pass, etc.).

    For a low-pass filter:
    b0 = (1 - cos(ω0)) / 2
    b1 = 1 - cos(ω0)
    b2 = (1 - cos(ω0)) / 2
    a0 = 1 + α
    a1 = -2 * cos(ω0)
    a2 = 1 - α

    For a high-pass filter:
    b0 = (1 + cos(ω0)) / 2
    b1 = -(1 + cos(ω0))
    b2 = (1 + cos(ω0)) / 2
    a0 = 1 + α
    a1 = -2 * cos(ω0)
    a2 = 1 - α
    
    This filter is commonly used in audio processing, equalization, and other
    signal processing applications due to its efficiency and versatility.
    
    Note: Since b0 = b2, and b1 = +2*b0 for low-pass and b1 = -2*b0 for
    high-pass, there will be a double zero at -1 for low-pass and a double
    zero at +1 for high-pass.

    """

# VCO Section
centered_subheader_with_help("🎚️ Voltage-Controlled Oscillator (VCO)", "The voltage-controlled oscillator (VCO) generates basic waveforms.")
col1, col2 = st.columns(2)
with col1:
    wave_type = st.selectbox("Wave Type", ["Square", "Triangle", "Sawtooth", "Sine"], help="Select the type of waveform to generate.")
    notes = st.text_area("Notes (frequencies in Hz, separated by commas)", ",".join(map(str, fur_elise_frequencies)), help=f"Enter a sequence of frequencies separated by commas.\n{available_notes}")
    durations = st.text_area("Durations (in seconds, separated by commas)", ",".join(map(str, fur_elise_durations)), help="Enter a sequence of durations separated by commas.")
    start_times = st.text_area("Start Times (in seconds, separated by commas)", ",".join(map(str, fur_elise_start_times)), help="Enter the start times of the notes, separated by commas.")
    frequencies = [float(freq) for freq in notes.split(',') if freq.strip()]
    durations = [float(dur) for dur in durations.split(',') if dur.strip()]
    start_times = [float(start) for start in start_times.split(',') if start.strip()]
    total_duration = max([start + duration for start, duration in zip(start_times, durations)])
with col2:
    # Generate only the first note for images
    first_waveform = generate_waveform(wave_type, frequencies[:1], [durations[0] * 5], start_times[:1])
    t = np.arange(len(first_waveform)) / SAMPLE_RATE
    fig, ax = plt.subplots()
    ax.plot(t[:1000], first_waveform[:1000], color='blue')
    ax.set_title("Waveform Preview (First Note)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)

    # Compute and display the Fourier transform for the first note
    fft_waveform = np.fft.fft(first_waveform)
    fft_freqs = np.fft.fftfreq(len(fft_waveform), 1 / SAMPLE_RATE)
    fig, ax = plt.subplots()
    ax.plot(fft_freqs[:len(fft_freqs)//2], 20 * np.log10(np.abs(fft_waveform)[:len(fft_waveform)//2]), color='purple', label="FFT")
    ax.set_xlim([0, 4000])
    ax.set_title("Fourier Transform (First Note)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (dB)")
    ax.legend()
    st.pyplot(fig)

st.audio(numpy_to_wav(first_waveform), format="audio/wav", start_time=0)

# Tremolo Section
centered_subheader_with_help("🔄 Tremolo LFO", "The LFO (Low Frequency Oscillator) of the tremolo modulates the signal amplitude.")
col3, col4 = st.columns(2)
with col3:
    lfo_wave_type = st.selectbox("LFO Wave Type", ["Sine", "Triangle", "Sawtooth", "Square"], key="lfo_wave_type", help="Select the waveform type for the LFO.")
    lfo_rate = st.slider("LFO Frequency (Hz)", 0.1, 20.0, 5.0, key="lfo_rate", help="Set the LFO frequency in Hertz.")
    lfo_depth = st.slider("LFO Depth", 0.0, 1.0, 0.5, key="lfo_depth", help="Set the LFO depth.")
with col4:
    # Compute only the LFO amplitude
    duration = len(first_waveform) / SAMPLE_RATE
    t_lfo = np.linspace(0, duration, len(first_waveform), endpoint=False)
    lfo_amplitude = 1 + lfo_depth * generate_waveform(lfo_wave_type, [lfo_rate], [duration], [0])

    # Display the LFO amplitude over time
    fig, ax = plt.subplots()
    ax.plot(t_lfo[:20000], lfo_amplitude[:20000], color='green', label="LFO Amplitude")
    ax.set_title("Signal Max Amplitude Modulated by LFO (First Note)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal Max Amplitude")
    st.pyplot(fig)

tremolo_signal = apply_lfo(first_waveform, lfo_rate, lfo_depth, lfo_wave_type)
st.audio(numpy_to_wav(tremolo_signal), format="audio/wav", start_time=0)

# Filter Section
centered_subheader_with_help("🎛️ Static Filter", "The filter modifies the frequency spectrum of the signal." + biquad_filter)
col5, col6 = st.columns(2)
with col5:
    type_filter = st.selectbox("Filter Type", ["low", "high"], help="Select the filter type (low-pass or high-pass).")
    cutoff = st.slider("Average Cutoff Frequency (Hz)", 20, 4000, 1000, help="Set the filter cutoff frequency in Hertz.")
    filter_q = st.slider("Resonance (Q)", 0.5, 10.0, 1.0, help="Set the filter resonance.")
with col6:
    # Display the Bode plot of the biquad filter
    b, a = biquad(cutoff, filter_q, type_filter)
    w, h = freqz(b, a, worN=8000)
    fig, ax = plt.subplots()
    ax.plot(0.5 * SAMPLE_RATE * w / np.pi, 20 * np.log10(np.abs(h)), 'b')  # Convert to dB
    ax.set_xlim([0, 6000])
    ax.set_ylim([-60, 0])
    ax.set_title("Biquad Filter Frequency Response")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain (dB)")
    ax.grid()
    st.pyplot(fig)

    # Display the poles and zeros of the biquad filter
    fig_pz = plot_poles_zeros(b, a)
    st.pyplot(fig_pz)

    # Display the filtered waveform without cutoff modulation by the LFO
    filtered_signal_static = apply_static_biquad_filter(first_waveform, cutoff, type_filter, filter_q)
    t_filtered_static = np.arange(len(filtered_signal_static)) / SAMPLE_RATE
    fig, ax = plt.subplots()
    ax.plot(t_filtered_static[1000:2000], filtered_signal_static[1000:2000], color='cyan')
    ax.set_title("Preview of Filtered Signal (First Note, Without LFO)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)

st.audio(numpy_to_wav(filtered_signal_static), format="audio/wav", start_time=0)

# Compute the minimum note duration
min_note_duration = min(durations)

# Filter LFO and ADSR Section
centered_subheader_with_help("🔄 Dynamic filter", "The LFO and ADSR envelope of the filter modulate the filter cutoff frequency.")
col7, col8 = st.columns(2)
with col7:
    filter_lfo_wave_type = st.selectbox("Filter LFO Wave Type", ["Sine", "Triangle", "Sawtooth", "Square"], key="filter_lfo_wave_type", help="Select the waveform type for the filter cutoff LFO.")
    filter_lfo_rate = st.slider("Filter LFO Frequency (Hz)", 0.1, 10.0, 2.0, key="filter_lfo_rate", help="Set the filter cutoff LFO frequency in Hertz.")
    filter_lfo_depth = st.slider("Filter LFO Depth", 0.0, 1.0, 0.3, key="filter_lfo_depth", help="Set the filter cutoff LFO depth.")
    filter_adsr_attack = st.slider("Filter Attack (s)", 0.01, min_note_duration / 2, min_note_duration * 0.1, key="filter_adsr_attack", help="Set the attack duration in seconds.")
    filter_adsr_decay = st.slider("Filter Decay (s)", 0.01, min_note_duration / 2, min_note_duration * 0.1, key="filter_adsr_decay", help="Set the decay duration in seconds.")
    filter_adsr_sustain = st.slider("Filter Sustain (level)", 0.0, 1.0, 0.7, key="filter_adsr_sustain", help="Set the filter cutoff sustain level.")
    filter_adsr_release = st.slider("Filter Release (s)", 0.01, min_note_duration / 2, min_note_duration * 0.1, key="filter_adsr_release", help="Set the filter cutoff release duration.")
    
    # Generate combined cutoff for the graph (original note duration)
    original_duration = durations[0]  # Use the original duration of the first note
    combined_cutoff_graph = apply_combined_adsr_lfo_to_cutoff(
        cutoff, filter_lfo_rate, filter_lfo_depth, filter_lfo_wave_type,
        filter_adsr_attack, filter_adsr_decay, filter_adsr_sustain, filter_adsr_release,
        original_duration
    )
with col8:
    # Plot the combined cutoff for the graph
    t_lfo_filter_graph = np.linspace(0, original_duration, len(combined_cutoff_graph), endpoint=False)
    fig, ax = plt.subplots()
    ax.plot(t_lfo_filter_graph, combined_cutoff_graph, color='orange')
    ax.set_title("Preview of Filter LFO and ADSR (First Note)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cutoff Frequency (Hz)")
    ax.legend()
    st.pyplot(fig)

    # Adjust combined cutoff for audio playback (multiplied by 5)
    combined_cutoff_audio = apply_combined_adsr_lfo_to_cutoff(
        cutoff, filter_lfo_rate, filter_lfo_depth, filter_lfo_wave_type,
        filter_adsr_attack * 5, filter_adsr_decay * 5, filter_adsr_sustain, filter_adsr_release * 5,
        len(first_waveform) / SAMPLE_RATE
    )
    filtered_signal_lfo_adsr = apply_dynamic_biquad_filter(
        first_waveform, combined_cutoff_audio, filter_type=type_filter, filter_q=filter_q
    )
    
    # Display the spectrogram of the filtered signal
    fig, ax = plt.subplots()
    Pxx, freqs, bins, im = ax.specgram(filtered_signal_lfo_adsr, NFFT=1024, Fs=SAMPLE_RATE, noverlap=512, cmap='viridis')
    st.subheader("Spectrogram of Dynamically Filtered Signal (for the First Note of the Melody)", help="The spectrogram shows the evolution of signal frequencies over time. Colors represent frequency amplitude (in dB).")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim([0, 4000])
    fig.colorbar(im, ax=ax, label="Amplitude (dB)")
    st.pyplot(fig)
    
    # Plot the time-domain graph of the dynamically filtered signal
    t_filtered_signal = np.arange(len(filtered_signal_lfo_adsr)) / SAMPLE_RATE
    fig, ax = plt.subplots()
    ax.plot(t_filtered_signal[:10000], filtered_signal_lfo_adsr[:10000], color='orange')
    ax.set_title("Preview of Dynamically Filtered Signal with LFO and ADSR (First Note)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)

st.audio(numpy_to_wav(filtered_signal_lfo_adsr), format="audio/wav", start_time=0)

# ADSR Section
centered_subheader_with_help("🎯 ADSR Envelope", "The ADSR envelope (Attack, Decay, Sustain, Release) modulates the signal amplitude over time.")
col9, col10 = st.columns(2)
with col9:
    attack = st.slider("Attack (s)", 0.01, min_note_duration / 2, min_note_duration * 0.1, key="adsr_attack", help="Set the attack duration in seconds.")
    decay = st.slider("Decay (s)", 0.01, min_note_duration / 2, min_note_duration * 0.1, key="adsr_decay", help="Set the decay duration in seconds.")
    sustain = st.slider("Sustain (level)", 0.0, 1.0, 0.7, key="adsr_sustain", help="Set the sustain level.")
    release = st.slider("Release (s)", 0.01, min_note_duration / 2, min_note_duration * 0.1, key="adsr_release", help="Set the release duration in seconds.")
with col10:
    # Generate the ADSR envelope for the graph (original note duration)
    original_duration = durations[0]  # Use the original duration of the first note
    t_graph = np.linspace(0, original_duration, int(SAMPLE_RATE * original_duration), endpoint=False)
    adsr_envelope = apply_adsr(np.ones_like(t_graph), attack, decay, sustain, release)
    fig, ax = plt.subplots()
    ax.plot(t_graph, adsr_envelope, color='red')
    ax.set_title("Preview of ADSR Envelope (First Note)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)

    # Adjust ADSR parameters for audio playback (multiplied by 5)
    adjusted_attack = attack * 5
    adjusted_decay = decay * 5
    adjusted_release = release * 5
    adsr_signal_audio = apply_adsr(first_waveform, adjusted_attack, adjusted_decay, sustain, adjusted_release)
    
st.audio(numpy_to_wav(adsr_signal_audio), format="audio/wav", start_time=0)

# Echo Section
centered_subheader_with_help(
    "🔊 Echo Effect", 
    "Add an echo effect to the signal.\n\n"
    "This function adds an echo effect to the input signal.\n"
    "- `delay` : Echo delay time in seconds.\n"
    "- `decay` : Echo attenuation (multiplicative factor).\n\n"
    "The echo signal is added to the original signal with a time offset."
)
col11, col12 = st.columns(2)
with col11:
    echo_delay = st.slider("Delay Time (s)", 0.01, 1.0, 0.2, key="echo_delay", help="Set the echo delay time in seconds.")
    echo_decay = st.slider("Echo Decay", 0.0, 1.0, 0.0, key="echo_decay", help="Set the echo attenuation level.")
with col12:
    # Adjust the original waveform to match the duration of the first note
    first_note_duration_samples = int(SAMPLE_RATE * durations[0])
    first_waveform_trimmed = first_waveform[:first_note_duration_samples]

    # Generate the echo signal
    delay_samples = int(SAMPLE_RATE * echo_delay)
    echo_signal = np.zeros(len(first_waveform_trimmed) + delay_samples)
    echo_signal[:len(first_waveform_trimmed)] = first_waveform_trimmed
    echo_signal[delay_samples:delay_samples + len(first_waveform_trimmed)] += first_waveform_trimmed * echo_decay

    # Time vector for plotting
    t_echo = np.arange(len(echo_signal)) / SAMPLE_RATE

    # Plot the original signal and the echo
    fig, ax = plt.subplots()
    ax.plot(t_echo[:len(first_waveform_trimmed)], first_waveform_trimmed, color='blue', label="Original Signal")
    ax.plot(t_echo[delay_samples:delay_samples + len(first_waveform_trimmed)], first_waveform_trimmed * echo_decay, color='red', label="Echo Signal")
    ax.set_title("Preview of Signal with Echo (First Note)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim([0, durations[0] + echo_delay + 0.1])  # Add a small margin for better visualization
    ax.grid()
    ax.legend()
    st.pyplot(fig)

# Add audio playback for the echo effect
echo_audio_duration = durations[0] + echo_delay
st.audio(numpy_to_wav(echo_signal[:int(SAMPLE_RATE * echo_audio_duration)]), format="audio/wav", start_time=0)

# Flanger Section
centered_subheader_with_help(
    "🔄 Flanger Effect", 
    "Add a flanger effect to the signal.\n\n"
    "This function adds a flanger effect to the input signal.\n"
    "- `frequency` : LFO frequency (low-frequency oscillator) in Hertz.\n"
    "- `depth` : Maximum delay depth in seconds.\n\n"
    "The flanger uses an LFO to dynamically modulate the time delay "
    "applied to the signal, creating a sweeping effect."
)
col13, col14 = st.columns(2)
with col13:
    flanger_rate = st.slider("Flanger LFO Frequency (Hz)", 0.1, 5.0, 0.5, key="flanger_rate", help="Set the LFO frequency for the flanger.")
    flanger_depth = st.slider("Flanger Depth (s)", 0.0, 0.05, 0.0, key="flanger_depth", help="Set the flanger depth in seconds.")
with col14:
    flanger_signal = apply_flanger(first_waveform, flanger_rate, flanger_depth)
    t_flanger = np.arange(len(flanger_signal)) / SAMPLE_RATE
    fig, ax = plt.subplots()
    ax.plot(t_flanger[:2000], flanger_signal[:2000], color='green')
    ax.set_title("Preview of Signal with Flanger (First Note)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)

st.audio(numpy_to_wav(flanger_signal), format="audio/wav", start_time=0)

# Complete Melody Section
centered_subheader_with_help("🎵 Play Complete Melody", "")

# Calculate the maximum required length for the melody signal
max_duration = max([start + duration for start, duration in zip(start_times, durations)])
melody_signal = np.zeros(int(SAMPLE_RATE * max_duration))

for freq, dur, start in zip(frequencies, durations, start_times):
        # Generate the waveform for the note
        note_signal = generate_waveform(wave_type, [freq], [dur], [0])

        # Apply transformations to the note
        note_cutoff = apply_combined_adsr_lfo_to_cutoff(
            cutoff, filter_lfo_rate, filter_lfo_depth, filter_lfo_wave_type,
            filter_adsr_attack, filter_adsr_decay, filter_adsr_sustain, filter_adsr_release, dur
        )
        transformed_note = apply_transformations(
            note_signal, lfo_rate, lfo_depth, lfo_wave_type, note_cutoff,
            type_filter, filter_q, attack, decay, sustain, release,
            echo_delay, echo_decay, flanger_rate, flanger_depth
        )

        # Generate the echo for the note
        echo_signal = np.zeros_like(melody_signal)
        echo_start_sample = int(SAMPLE_RATE * (start + echo_delay))
        echo_end_sample = echo_start_sample + len(transformed_note)

        if echo_end_sample <= len(melody_signal):
            echo_signal[echo_start_sample:echo_end_sample] = transformed_note * echo_decay

        # Add the transformed note and its echo to the final signal
        start_sample = int(SAMPLE_RATE * start)
        end_sample = start_sample + len(transformed_note)

        if end_sample <= len(melody_signal):
            melody_signal[start_sample:end_sample] += transformed_note

        melody_signal += echo_signal

# Trim the signal to remove unnecessary silence at the end
non_zero_indices = np.nonzero(melody_signal)[0]
if len(non_zero_indices) > 0:
    melody_signal = melody_signal[:non_zero_indices[-1] + 1]

# Normalize the final signal
melody_signal /= np.max(np.abs(melody_signal))

# Generate and play the audio file
audio_file = numpy_to_wav(melody_signal)
st.audio(audio_file, format="audio/wav")
