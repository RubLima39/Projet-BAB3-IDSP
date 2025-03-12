import streamlit as st
import numpy as np
from scipy.signal import lfilter, freqz
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
def apply_lfo(signal, rate, depth, wave_type='Sinus', sample_rate=44100):
    t = np.linspace(0, len(signal) / sample_rate, len(signal), endpoint=False)
    lfo = 1 + depth * generate_waveform(wave_type, rate, len(signal) / sample_rate, sample_rate)
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

    # Normalisation des coefficients pour que le niveau stable soit inférieur à 1
    b = [b0 / (a0 * 10), b1 / (a0 * 10), b2 / (a0 * 10)]
    a = [1, a1 / a0, a2 / a0]
    return b, a

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

# Second LFO pour moduler le filtre
def apply_filter_lfo(cutoff, lfo_rate, lfo_depth, wave_type='Sinus', duration=1, sample_rate=44100):
    # Créer le LFO pour moduler la fréquence de coupure
    lfo = 1 + lfo_depth * generate_waveform(wave_type, lfo_rate, duration, sample_rate)
    # Appliquer le LFO à chaque échantillon pour obtenir une série dynamique de valeurs de coupure
    return cutoff * lfo

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
def apply_transformations(signal, sample_rate, lfo_rate, lfo_depth, lfo_wave_type, filter_lfo, filter_type, filter_q, attack, decay, sustain, release):
    lfo_signal = apply_lfo(signal, lfo_rate, lfo_depth, lfo_wave_type, sample_rate)
    filtered_signal = apply_dynamic_biquad_filter(lfo_signal, filter_lfo, sample_rate, filter_type, filter_q)
    adsr_signal = apply_adsr(filtered_signal, sample_rate, attack, decay, sustain, release)
    return adsr_signal

# Application Streamlit
st.title("🎛️ Synthétiseur Soustractif")
st.info("Ajustez les paramètres et cliquez sur 'Jouer le son' pour écouter votre création sonore.")

# Section VCO
col1, col2 = st.columns(2)
with col1:
    st.subheader("🎚️ Oscillateur à Commande de Tension (VCO)", help="L'oscillateur à commande de tension (VCO) génère des formes d'onde de base.")
    wave_type = st.selectbox("Type d'onde", ["Sinus", "Triangle", "Dent de scie", "Carré"], help="Sélectionnez le type d'onde à générer.")
    frequency = st.slider("Fréquence (Hz)", 20, 5000, 440, help="Définissez la fréquence de l'onde en Hertz.")
    duration = st.slider("Durée (s)", 1, 6, 2, help="Définissez la durée de l'onde en secondes.")
with col2:
    waveform = generate_waveform(wave_type, frequency, duration)
    t = np.arange(len(waveform)) / 44100
    fig, ax = plt.subplots()
    ax.plot(t[:1000], waveform[:1000], color='blue', label="Forme d'onde")
    ax.set_title("Aperçu de la forme d'onde")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)

    # Calcul et affichage de la transformée de Fourier
    fft_waveform = np.fft.fft(waveform)
    fft_freqs = np.fft.fftfreq(len(fft_waveform), 1 / 44100)
    fig, ax = plt.subplots()
    ax.plot(fft_freqs[:len(fft_freqs)//2], np.abs(fft_waveform)[:len(fft_waveform)//2], color='purple', label="FFT")
    ax.set_xlim([0, 20000])  # Limiter l'axe des x à 20 000 Hz
    ax.set_title("Transformée de Fourier")
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("Amplitude (arbitraire)")
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
    ax.plot(t_lfo[:20000], lfo_signal[:20000], color='green', label='LFO')
    ax.set_title("Aperçu du LFO")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)

# Section Filtre
col5, col6 = st.columns(2)
with col5:
    st.subheader("🎛️ Filtre", help="Le filtre permet de modifier le spectre de fréquence du signal.")
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
    # Affichage de l'onde filtrée sans modulation du cutoff par le LFO
    filtered_signal_static = apply_static_biquad_filter(waveform, cutoff, 44100, type_filter, filter_q)
    t_filtered_static = np.arange(len(filtered_signal_static)) / 44100
    fig, ax = plt.subplots()
    ax.plot(t_filtered_static[:1000], filtered_signal_static[:1000], color='cyan', label="Signal Filtré (Sans LFO)")
    ax.set_title("Aperçu du signal filtré (Sans LFO)")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)

# Section LFO du Filtre
col7, col8 = st.columns(2)
with col7:
    st.subheader("🔄 LFO du Filtre", help="Le LFO du filtre module la fréquence de coupure du filtre.")
    filter_lfo_wave_type = st.selectbox("Type d'onde LFO Filtre", ["Sinus", "Triangle", "Dent de scie", "Carré"], key="filter_lfo_wave_type", help="Sélectionnez le type d'onde pour le LFO du filtre.")
    filter_lfo_rate = st.slider("Fréquence LFO Filtre (Hz)", 0.1, 10.0, 2.0, help="Définissez la fréquence du LFO du filtre en Hertz.")
    filter_lfo_depth = st.slider("Profondeur LFO Filtre", 0.0, 1.0, 0.3, help="Définissez la profondeur du LFO du filtre.")
    filter_lfo = apply_filter_lfo(cutoff, filter_lfo_rate, filter_lfo_depth, filter_lfo_wave_type, duration)
with col8:
    fig, ax = plt.subplots()
    t_lfo_filter = np.arange(len(filter_lfo)) / 44100
    ax.plot(t[:50000], filter_lfo[:50000], color='orange', label='LFO Filtre')
    ax.set_title("Aperçu du LFO du Filtre")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Fréquence de coupure (Hz)")
    ax.legend()
    st.pyplot(fig)
    filtered_signal_lfo = apply_dynamic_biquad_filter(waveform, filter_lfo, sample_rate=44100, filter_type=type_filter, filter_q=1.0)
    fig, ax = plt.subplots()
    ax.plot(t[:50000], filtered_signal_lfo[:50000], color='orange', label='LFO Filtre')
    ax.set_title("Aperçu du signal filtré dynamiquement")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
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
    ax.plot(t[:300000], adsr_envelope[:300000], color='red', label='ADSR')
    ax.set_title("Aperçu de l'enveloppe ADSR")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)

# Appliquer toutes les transformations sur le signal initialnommés. Voici la correction :
transformed_signal = apply_transformations(waveform, 44100, lfo_rate, lfo_depth, lfo_wave_type, filter_lfo, type_filter, filter_q, attack, decay, sustain, release)

with col8:
    # Affichage de la transformée de Fourier du signal filtré
    fft_filtered_signal = np.fft.fft(transformed_signal)
    fft_filtered_freqs = np.fft.fftfreq(len(fft_filtered_signal), 1 / 44100)
    fig, ax = plt.subplots()
    ax.plot(fft_filtered_freqs[:len(fft_filtered_freqs)//2], np.abs(fft_filtered_signal)[:len(fft_filtered_signal)//2], color='orange', label="FFT Filtré")
    ax.set_xlim([0, 20000])  # Limiter l'axe des x à 20 000 Hz
    ax.set_title("Transformée de Fourier du Signal Filtré")
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("Amplitude (arbitraire)")
    ax.legend()
    st.pyplot(fig)

# Génération et lecture du signal final
audio_file = numpy_to_wav(transformed_signal)

st.subheader("🎧 Jouer le son")
st.audio(audio_file, format="audio/wav")
