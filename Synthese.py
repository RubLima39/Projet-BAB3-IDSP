import streamlit as st
import numpy as np
from scipy.signal import lfilter, freqz
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import io

# Fonction pour générer des formes d'onde
def generate_waveform(wave_type, frequencies, durations, sample_rate=44100):
    waveform = np.array([])
    for frequency, duration in zip(frequencies, durations):
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        if wave_type == 'Sinus':
            wave = np.sin(2 * np.pi * frequency * t)
        elif wave_type == 'Triangle':
            wave = 2 * np.abs(2 * (t * frequency % 1) - 1) - 1
        elif wave_type == 'Dent de scie':
            wave = 2 * (t * frequency % 1) - 1
        elif wave_type == 'Carré':
            wave = np.sign(np.sin(2 * np.pi * frequency * t))
        else:
            wave = np.zeros_like(t)
        waveform = np.concatenate((waveform, wave))
    return waveform

# LFO simple
def apply_lfo(signal, rate, depth, wave_type='Sinus', sample_rate=44100):
    t = np.linspace(0, len(signal) / sample_rate, len(signal), endpoint=False)
    lfo = 1 + depth * generate_waveform(wave_type, [rate], [len(signal) / sample_rate], sample_rate)
    return signal * lfo

# Fonction pour calculer les coefficients du filtre biquad
def biquad(cutoff, q, sample_rate=44100, filter_type='low'):
    omega = 2 * np.pi * cutoff / sample_rate
    alpha = np.sin(omega) / (2 * q)

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
    # Rem: comme b0 et b2 sont les mêmes pour low et high, et b1 = 2*b0 pour le passe-bas et -2*b0 pour le passe-haut, 
    # on a un zéro double en -1 pour le passe-bas et un zéro double en +1 pour le passe-haut.
    
    # Normalisation des coefficients pour que le niveau stable soit inférieur à 1
    b = [b0 / (a0 * 10), b1 / (a0 * 10), b2 / (a0 * 10)]
    a = [1, a1 / a0, a2 / a0]
    return b, a

# Fonction pour afficher les pôles et zéros du filtre
def plot_poles_zeros(b, a):
    from matplotlib import patches

    # Calculer les pôles et zéros
    z = np.roots(b)
    p = np.roots(a)

    # Tracer les pôles et zéros
    fig, ax = plt.subplots()
    unit_circle = patches.Circle((0, 0), radius=1, fill=False, color='black', ls='dotted')
    ax.add_patch(unit_circle)
    ax.plot(np.real(z), np.imag(z), 'go', label='Zéros')
    ax.plot(np.real(p), np.imag(p), 'rx', label='Pôles')
    ax.set_xlim((-1.5, 1.5))
    ax.set_ylim((-1.5, 1.5))
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_title("Pôles et Zéros du Filtre")
    ax.set_xlabel("Partie Réelle")
    ax.set_ylabel("Partie Imaginaire")
    ax.legend()
    ax.grid()
    return fig

# Application de l'enveloppe ADSR
def apply_adsr(signal, sample_rate, attack, decay, sustain, release):
    length = len(signal)
    t = np.linspace(0, length / sample_rate, length, endpoint=False)

    attack_samples = int(sample_rate * attack)
    decay_samples = int(sample_rate * decay)
    release_samples = int(sample_rate * release)

    envelope = np.zeros(length)

    # Attack
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    # Decay
    decay_end = attack_samples + decay_samples
    envelope[attack_samples:decay_end] = np.linspace(1, sustain, decay_samples)
    # Sustain
    envelope[decay_end:length - release_samples] = sustain
    # Release
    envelope[length - release_samples:] = np.linspace(sustain, 0, release_samples)

    return signal * envelope

# Fonction pour convertir un signal numpy en fichier WAV
def numpy_to_wav(signal, sample_rate=44100):
    signal = np.int16(signal / np.max(np.abs(signal)) * 32767)
    buf = io.BytesIO()
    wav.write(buf, sample_rate, signal)
    buf.seek(0)
    return buf

# Fonction pour appliquer une enveloppe ADSR et un LFO sur le cutoff du filtre
def apply_combined_adsr_lfo_to_cutoff(cutoff, lfo_rate, lfo_depth, lfo_wave_type, sample_rate, attack, decay, sustain, release, total_duration):
    length = int(sample_rate * total_duration)
    t = np.linspace(0, total_duration, length, endpoint=False)

    # Generate LFO
    lfo = 1 + lfo_depth * generate_waveform(lfo_wave_type, [lfo_rate], [total_duration], sample_rate)

    # Generate ADSR envelope
    attack_samples = int(sample_rate * attack)
    decay_samples = int(sample_rate * decay)
    release_samples = int(sample_rate * release)

    envelope = np.zeros(length)

    # Attack
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    # Decay
    decay_end = attack_samples + decay_samples
    envelope[attack_samples:decay_end] = np.linspace(1, sustain, decay_samples)
    # Sustain
    envelope[decay_end:length - release_samples] = sustain
    # Release
    envelope[length - release_samples:] = np.linspace(sustain, 0, release_samples)

    # Combine LFO and ADSR
    combined = lfo * envelope

    return cutoff * combined

# Fonction pour appliquer un filtre statique sans LFO
def apply_static_biquad_filter(signal, cutoff, sample_rate=44100, filter_type='low', filter_q=1.0):
    b, a = biquad(cutoff, filter_q, sample_rate, filter_type)
    return lfilter(b, a, signal)

# Appliquer le filtre biquad avec LFO modulé et résonance
def apply_dynamic_biquad_filter(signal, cutoff_lfo, sample_rate=44100, filter_type='low', filter_q=1.0):
    filtered_signal = np.zeros_like(signal)
    b, a = biquad(cutoff_lfo[0], filter_q, sample_rate, filter_type)  # Initialize filter coefficients
    zi = np.zeros(max(len(b), len(a)) - 1)  # Initialize the filter state
    for i in range(len(signal)):
        if i > 0 and cutoff_lfo[i] != cutoff_lfo[i-1]:  # Update filter coefficients if cutoff changes
            b, a = biquad(cutoff_lfo[i], filter_q, sample_rate, filter_type)
            zi = lfilter(b, a, [0], zi=zi)[1]  # Reset the filter state when coefficients change
        filtered_signal[i], zi = lfilter(b, a, [signal[i]], zi=zi)  # Apply the filter with the current state
    return filtered_signal

# Fonction pour appliquer toutes les transformations
def apply_transformations(signal, sample_rate, lfo_rate, lfo_depth, lfo_wave_type, combined_cutoff, filter_type, filter_q, attack, decay, sustain, release):
    lfo_signal = apply_lfo(signal, lfo_rate, lfo_depth, lfo_wave_type, sample_rate)
    filtered_signal = apply_dynamic_biquad_filter(lfo_signal, combined_cutoff, sample_rate, filter_type, filter_q)
    adsr_signal = apply_adsr(filtered_signal, sample_rate, attack, decay, sustain, release)
    return adsr_signal

# Application Streamlit
st.title("🎛️ Synthétiseur Soustractif")
st.info("Ajustez les paramètres et cliquez sur 'Jouer le son' pour écouter votre création sonore.")

# Fréquences et durées des notes pour le début de "Für Elise"
fur_elise_frequencies = [659.25, 622.25, 659.25, 622.25, 659.25, 493.88, 587.33, 523.25, 440.00]
fur_elise_durations = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.6, 0.2, 0.2, 0.2, 0.6]

# Notes disponibles et leurs fréquences en Hz
notes_disponibles = """
Notes disponibles et leurs fréquences en Hz:
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

filtre_biquad = """
    Appliquer un filtre biquad au signal d'entrée.

    Cette fonction calcule les coefficients en fonction des caractéristiques souhaitées du filtre et applique le filtre au signal d'entrée.

    Le filtre biquad est un filtre linéaire récursif de second ordre qui utilise l'équation aux différences suivante :

    y[n] = b0 * x[n] + b1 * x[n-1] + b2 * x[n-2] - a1 * y[n-1] - a2 * y[n-2]

    Où :
    - x[n] est l'échantillon d'entrée actuel
    - x[n-1] et x[n-2] sont les échantillons d'entrée précédents
    - y[n] est l'échantillon de sortie actuel
    - y[n-1] et y[n-2] sont les échantillons de sortie précédents
    - b0, b1, b2 sont les coefficients de l'élément direct
    - a1, a2 sont les coefficients de rétroaction

    Les coefficients (b0, b1, b2, a1, a2) sont calculés en fonction des caractéristiques souhaitées du filtre telles que la fréquence de coupure, le facteur Q et le gain. Ces caractéristiques définissent le comportement du filtre, y compris sa réponse en fréquence et sa stabilité.

    Le calcul des coefficients implique les étapes suivantes :
    1. Déterminer la fréquence normalisée (ω0) en fonction de la fréquence de coupure et de la fréquence d'échantillonnage.
    2. Calculer les variables intermédiaires (α, cos(ω0), sin(ω0)) en utilisant des fonctions trigonométriques.
    3. Calculer les coefficients en utilisant les formules standard du biquad pour le type de filtre spécifique (passe-bas, passe-haut, passe-bande, etc.).

    Pour un filtre passe-bas :
    b0 = (1 - cos(ω0)) / 2
    b1 = 1 - cos(ω0)
    b2 = (1 - cos(ω0)) / 2
    a0 = 1 + α
    a1 = -2 * cos(ω0)
    a2 = 1 - α

    Pour un filtre passe-haut :
    b0 = (1 + cos(ω0)) / 2
    b1 = -(1 + cos(ω0))
    b2 = (1 + cos(ω0)) / 2
    a0 = 1 + α
    a1 = -2 * cos(ω0)
    a2 = 1 - α

    Ce filtre est couramment utilisé dans le traitement audio, l'égalisation et d'autres applications de traitement du signal en raison de son efficacité et de sa polyvalence.
    """

# Section VCO
col1, col2 = st.columns(2)
with col1:
    st.subheader("🎚️ Oscillateur à Commande en Tension (VCO)", help="L'oscillateur à commande en tension (VCO) génère des formes d'onde de base.")
    wave_type = st.selectbox("Type d'onde", ["Carré", "Triangle", "Dent de scie", "Sinus"], help="Sélectionnez le type d'onde à générer.")
    notes = st.text_area("Notes (fréquences en Hz, séparées par des virgules)", ",".join(map(str, fur_elise_frequencies)), help=f"Entrez une suite de fréquences séparées par des virgules.\n{notes_disponibles}")
    durations = st.text_area("Durées (en secondes, séparées par des virgules)", ",".join(map(str, fur_elise_durations)), help="Entrez une suite de durées séparées par des virgules.")
    frequencies = [float(freq) for freq in notes.split(',')]
    durations = [float(dur) for dur in durations.split(',')]
    total_duration = sum(durations)
with col2:
    waveform = generate_waveform(wave_type, frequencies, durations)
    t = np.arange(len(waveform)) / 44100
    fig, ax = plt.subplots()
    ax.plot(t[:1000], waveform[:1000], color='blue')
    ax.set_title("Aperçu de la forme d'onde")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)

    # Calcul et affichage de la transformée de Fourier
    fft_waveform = np.fft.fft(waveform)
    fft_freqs = np.fft.fftfreq(len(fft_waveform), 1 / 44100)
    fig, ax = plt.subplots()
    ax.plot(fft_freqs[:len(fft_freqs)//2], 20 * np.log10(np.abs(fft_waveform)[:len(fft_waveform)//2]), color='purple', label="FFT")
    ax.set_xlim([0, 4000])
    ax.set_title("Transformée de Fourier")
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("Amplitude (dB)")
    ax.legend()
    st.pyplot(fig)

# Section trémolo
col3, col4 = st.columns(2)
with col3:
    st.subheader("🔄 LFO du Trémolo", help="Le LFO (Low Frequency Oscillator) du trémolo module l'amplitude du signal.")
    lfo_wave_type = st.selectbox("Type d'onde LFO", ["Sinus", "Triangle", "Dent de scie", "Carré"], key="lfo_wave_type", help="Sélectionnez le type d'onde pour le LFO.")
    lfo_rate = st.slider("Fréquence LFO (Hz)", 0.1, 20.0, 5.0, help="Définissez la fréquence du LFO en Hertz.")
    lfo_depth = st.slider("Profondeur LFO", 0.0, 1.0, 0.5, help="Définissez la profondeur du LFO.")
with col4:
    lfo_signal = apply_lfo(waveform, lfo_rate, lfo_depth, lfo_wave_type)
    t_lfo = np.arange(len(lfo_signal)) / 44100
    fig, ax = plt.subplots()
    ax.plot(t_lfo[:20000], lfo_signal[:20000], color='green')
    ax.set_title("Aperçu du LFO")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)

# Section Filtre
col5, col6 = st.columns(2)
with col5:
    st.subheader("🎛️ Filtre", help="Le filtre permet de modifier le spectre de fréquence du signal."+filtre_biquad)
    type_filter = st.selectbox("Type de filtre", ["low", "high"], help="Sélectionnez le type de filtre (passe-bas ou passe-haut).")
    cutoff = st.slider("Fréquence de coupure moyenne (Hz)", 20, 4000, 1000, help="Définissez la fréquence de coupure du filtre en Hertz.")
    filter_q = st.slider("Résonance (Q)", 0.5, 10.0, 1.0, help="Définissez la résonance du filtre.")
with col6:
    # Affichage de la courbe de Bode du filtre biquad
    b, a = biquad(cutoff, filter_q, 44100, type_filter)
    w, h = freqz(b, a, worN=8000)
    fig, ax = plt.subplots()
    ax.plot(0.5 * 44100 * w / np.pi, 20 * np.log10(np.abs(h)), 'b')  # Convertir en dB
    ax.set_xlim([0, 6000])
    ax.set_ylim([-60, 0])
    ax.set_title("Réponse en fréquence du filtre biquad")
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("Gain (dB)")
    ax.grid()
    st.pyplot(fig)

    # Affichage des pôles et zéros du filtre biquad
    fig_pz = plot_poles_zeros(b, a)
    st.pyplot(fig_pz)

    # Affichage de l'onde filtrée sans modulation du cutoff par le LFO
    filtered_signal_static = apply_static_biquad_filter(waveform, cutoff, 44100, type_filter, filter_q)
    t_filtered_static = np.arange(len(filtered_signal_static)) / 44100
    fig, ax = plt.subplots()
    ax.plot(t_filtered_static[1000:2000], filtered_signal_static[1000:2000], color='cyan')
    ax.set_title("Aperçu du signal filtré (Sans LFO)")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)

# Section LFO et ADSR du Filtre
col7, col8 = st.columns(2)
with col7:
    st.subheader("🔄 LFO et ADSR sur la fréquence de coupure (cutoff) du filtre", help="Le LFO et l'enveloppe ADSR du filtre modulent la fréquence de coupure du filtre.")
    filter_lfo_wave_type = st.selectbox("Type d'onde LFO Filtre", ["Sinus", "Triangle", "Dent de scie", "Carré"], key="filter_lfo_wave_type", help="Sélectionnez le type d'onde pour le LFO du cutoff.")
    filter_lfo_rate = st.slider("Fréquence LFO Filtre (Hz)", 0.1, 10.0, 2.0, help="Définissez la fréquence du LFO du cutoff en Hertz.")
    filter_lfo_depth = st.slider("Profondeur LFO Filtre", 0.0, 1.0, 0.3, help="Définissez la profondeur du LFO du cutoff.")
    filter_adsr_attack = st.slider("Attack (s) Filtre", 0.01, 2.0, 0.1, help="Définissez la durée de l'attaque du cutoff en secondes.")
    filter_adsr_decay = st.slider("Decay (s) Filtre", 0.01, 2.0, 0.1, help="Définissez la durée de la décroissance du cutoff en secondes.")
    filter_adsr_sustain = st.slider("Sustain (niveau) Filtre", 0.0, 1.0, 0.7, help="Définissez le niveau de maintien du cutoff.")
    filter_adsr_release = st.slider("Release (s) Filtre", 0.01, 2.0, 0.2, help="Définissez la durée de la relâche du cutoff en secondes.")
    combined_cutoff = apply_combined_adsr_lfo_to_cutoff(cutoff, filter_lfo_rate, filter_lfo_depth, filter_lfo_wave_type, 44100, filter_adsr_attack, filter_adsr_decay, filter_adsr_sustain, filter_adsr_release, total_duration)
with col8:
    fig, ax = plt.subplots()
    t_lfo_filter = np.arange(len(combined_cutoff)) / 44100
    ax.plot(t_lfo_filter, combined_cutoff, color='orange')
    ax.set_title("Aperçu du LFO et ADSR du Filtre")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Fréquence de coupure (Hz)")
    ax.legend()
    st.pyplot(fig)
    filtered_signal_lfo_adsr = apply_dynamic_biquad_filter(waveform, combined_cutoff, sample_rate=44100, filter_type=type_filter, filter_q=filter_q)
    fig, ax = plt.subplots()
    ax.plot(t[2000:10000], filtered_signal_lfo_adsr[2000:10000], color='orange')
    ax.set_title("Aperçu du signal filtré dynamiquement avec LFO et ADSR")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)

with col8:
    # Affichage de la transformée de Fourier du signal filtré
    fft_filtered_signal = np.fft.fft(filtered_signal_lfo_adsr)
    fft_filtered_freqs = np.fft.fftfreq(len(fft_filtered_signal), 1 / 44100)
    fig, ax = plt.subplots()
    ax.plot(fft_filtered_freqs[:len(fft_filtered_freqs)//2], 20 * np.log10(np.abs(fft_filtered_signal)[:len(fft_filtered_signal)//2]), color='orange')
    ax.set_xlim([0, 4000])
    ax.set_title("Transformée de Fourier du Signal Filtré avec LFO et ADSR")
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("Amplitude (dB)")
    ax.legend()
    st.pyplot(fig)

# Section ADSR
col9, col10 = st.columns(2)
with col9:
    st.subheader("🎯 Enveloppe ADSR", help="L'enveloppe ADSR (Attack, Decay, Sustain, Release) module l'amplitude du signal au fil du temps.")
    attack = st.slider("Attack (s)", 0.01, 2.0, 0.1, help="Définissez la durée de l'attaque en secondes.")
    decay = st.slider("Decay (s)", 0.01, 2.0, 0.1, help="Définissez la durée de la décroissance en secondes.")
    sustain = st.slider("Sustain (niveau)", 0.0, 1.0, 0.7, help="Définissez le niveau de maintien.")
    release = st.slider("Release (s)", 0.01, 2.0, 0.2, help="Définissez la durée de la relâche en secondes.")
with col10:
    adsr_envelope = apply_adsr(np.ones_like(t), 44100, attack, decay, sustain, release)
    fig, ax = plt.subplots()
    ax.plot(t[:600000], adsr_envelope[:600000], color='red')
    ax.set_title("Aperçu de l'enveloppe ADSR")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)

# Appliquer toutes les transformations sur le signal initial
transformed_signal = apply_transformations(waveform, 44100, lfo_rate, lfo_depth, lfo_wave_type, combined_cutoff, type_filter, filter_q, attack, decay, sustain, release)

# Génération et lecture du signal final
audio_file = numpy_to_wav(transformed_signal)

st.subheader("🎧 Jouer le son")
st.audio(audio_file, format="audio/wav")
