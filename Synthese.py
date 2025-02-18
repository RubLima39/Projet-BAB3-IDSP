import streamlit as st
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import io

# Fonction pour générer des formes d'onde
def generate_waveform(wave_type, frequency, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    if wave_type == 'Sinus':
        return np.sin(2 * np.pi * frequency * t)
    elif wave_type == 'Triangle':
        return 2 * np.abs(2 * (t * frequency % 1) - 1) - 1
    elif wave_type == 'Dent de scie':
        return 2 * (t * frequency % 1) - 1
    elif wave_type == 'Carré':
        return np.sign(np.sin(2 * np.pi * frequency * t))
    else:
        return np.zeros_like(t)

# LFO simple
def apply_lfo(signal, rate, depth, sample_rate=44100):
    t = np.linspace(0, len(signal) / sample_rate, len(signal), endpoint=False)
    lfo = 1 + depth * np.sin(2 * np.pi * rate * t)
    return signal * lfo

# Filtre basique
def butter_filter(signal, cutoff, sample_rate=44100, filter_type='low', order=5):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    return lfilter(b, a, signal)

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
    # Convertir le signal numpy en un format compatible WAV
    signal = np.int16(signal / np.max(np.abs(signal)) * 32767)  # Normalisation
    buf = io.BytesIO()
    wav.write(buf, sample_rate, signal)
    buf.seek(0)
    return buf

# Application Streamlit
st.title("Synthétiseur Subtractif Basique")

# Paramètres du VCO
wave_type = st.selectbox("Type d'onde", ["Sinus", "Triangle", "Dent de scie", "Carré"])
frequency = st.slider("Fréquence (Hz)", 20, 2000, 440)
duration = st.slider("Durée (s)", 1, 5, 2)

# Affichage graphique de la forme d'onde
waveform = generate_waveform(wave_type, frequency, duration)
fig, ax = plt.subplots()
ax.plot(waveform[:1000])
ax.set_title("Aperçu de la forme d'onde")
st.pyplot(fig)

# Paramètres des LFO
lfo_rate = st.slider("Fréquence LFO (Hz)", 0.1, 20.0, 5.0)
lfo_depth = st.slider("Profondeur LFO", 0.0, 1.0, 0.5)

# Affichage graphique du LFO
lfo_signal = apply_lfo(waveform, lfo_rate, lfo_depth)
fig, ax = plt.subplots()
ax.plot(lfo_signal[:10000])
ax.set_title("Aperçu du LFO")
st.pyplot(fig)

# Paramètres du filtre
filter_type = st.selectbox("Type de filtre", ["low", "high"])
cutoff = st.slider("Fréquence de coupure (Hz)", 20, 2000, 1000)

# Paramètres de l'enveloppe ADSR
attack = st.slider("Attack (s)", 0.01, 2.0, 0.1)
decay = st.slider("Decay (s)", 0.01, 2.0, 0.1)
sustain = st.slider("Sustain (niveau)", 0.0, 1.0, 0.7)
release = st.slider("Release (s)", 0.01, 2.0, 0.2)

# Affichage graphique de l'enveloppe ADSR
t = np.linspace(0, duration, int(44100 * duration), endpoint=False)
adsr_envelope = apply_adsr(np.ones_like(t), 44100, attack, decay, sustain, release)
fig, ax = plt.subplots()
ax.plot(t[:1000000], adsr_envelope[:1000000])
ax.set_title("Aperçu de l'enveloppe ADSR")
st.pyplot(fig)

# Générer le signal complet
filtered_signal = butter_filter(lfo_signal, cutoff, filter_type=filter_type)
adsr_signal = apply_adsr(filtered_signal, 44100, attack, decay, sustain, release)

# Convertir le signal en un fichier audio
audio_file = numpy_to_wav(adsr_signal)

# Lecture du son si demandé
if st.button("Jouer le son"):
    st.audio(audio_file, format="audio/wav")

st.write("Ajuste les paramètres et clique sur 'Jouer le son' pour écouter.")
