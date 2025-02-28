import streamlit as st
import numpy as np
from scipy.signal import butter, lfilter, freqz
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
    signal = np.int16(signal / np.max(np.abs(signal)) * 32767)
    buf = io.BytesIO()
    wav.write(buf, sample_rate, signal)
    buf.seek(0)
    return buf

# Second LFO pour moduler le filtre
def apply_filter_lfo(cutoff, lfo_rate, lfo_depth, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Créer le LFO pour moduler la fréquence de coupure
    lfo = 1 + lfo_depth * np.sin(2 * np.pi * lfo_rate * t)
    # Appliquer le LFO à chaque échantillon pour obtenir une série dynamique de valeurs de coupure
    return cutoff * lfo

# Fonction pour appliquer le filtre avec LFO modulé sur la fréquence de coupure
def apply_filter_after_oversampling(signal, filter_lfo, filter_type, sample_rate=44100, oversample_factor=10):
    # Initialiser le tableau du signal filtré
    filtered_signal = np.zeros_like(signal)
    
    # Appliquer le filtre à chaque tranche du signal
    for i in range(0, len(signal), oversample_factor):
        # Calculer la fréquence de coupure pour cette tranche
        current_cutoff = filter_lfo[i]  # Prendre la fréquence de coupure correspondante pour cette tranche
        
        # Appliquer le filtre à la tranche avec la fréquence de coupure modifiée
        filtered_signal[i:i + oversample_factor] = butter_filter(
            signal[i:i + oversample_factor], current_cutoff, sample_rate, filter_type)
    
    return filtered_signal

# Application Streamlit
st.title("🎛️ Synthétiseur Subtractif")

# Section VCO
col1, col2 = st.columns(2)
with col1:
    st.subheader("🎚️ Oscillateur à Commande de Tension (VCO)")
    wave_type = st.selectbox("Type d'onde", ["Sinus", "Triangle", "Dent de scie", "Carré"])
    frequency = st.slider("Fréquence (Hz)", 20, 2000, 440)
    duration = st.slider("Durée (s)", 1, 6, 2)
with col2:
    waveform = generate_waveform(wave_type, frequency, duration)
    fig, ax = plt.subplots()
    ax.plot(waveform[:1000], color='blue', label="Forme d'onde")
    ax.set_xlim([0, 1000]) 
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
    st.subheader("🔄 LFO du Trémolo")
    lfo_rate = st.slider("Fréquence LFO (Hz)", 0.1, 20.0, 5.0)
    lfo_depth = st.slider("Profondeur LFO", 0.0, 1.0, 0.5)
with col4:
    lfo_signal = apply_lfo(waveform, lfo_rate, lfo_depth)
    fig, ax = plt.subplots()
    ax.plot(lfo_signal[:20000], color='green', label='LFO')
    ax.set_xlim([0, 5000]) 
    ax.set_title("Aperçu du LFO")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)

# Section Filtre
col5, col6 = st.columns(2)
with col5:
    st.subheader("🎛️ Filtre")
    filter_type = st.selectbox("Type de filtre", ["low", "high"])
    cutoff = st.slider("Fréquence de coupure moyenne (Hz)", 20, 2000, 1000)
with col6:
    # Affichage de la courbe de Bode du filtre
    st.subheader("📉 Courbe de Bode du Filtre")
    b, a = butter(5, cutoff / (0.5 * 44100), btype=filter_type, analog=False)
    w, h = freqz(b, a, worN=8000)
    fig, ax = plt.subplots()
    ax.plot(0.5 * 44100 * w / np.pi, np.abs(h), 'b')
    ax.set_xlim([0, 6000])  # Limiter l'axe des x à 10 000 Hz
    ax.set_title("Réponse en fréquence du filtre")
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("Gain")
    ax.grid()
    st.pyplot(fig)

# Section LFO du Filtre
col7, col8 = st.columns(2)
with col7:
    st.subheader("🔄 LFO du Filtre")
    filter_lfo_rate = st.slider("Fréquence LFO Filtre (Hz)", 0.1, 10.0, 2.0)
    filter_lfo_depth = st.slider("Profondeur LFO Filtre", 0.0, 1.0, 0.3)
    filter_lfo = apply_filter_lfo(cutoff, filter_lfo_rate, filter_lfo_depth, duration)
with col8:
    fig, ax = plt.subplots()
    ax.plot(filter_lfo[:50000], color='orange', label='LFO Filtre')
    ax.set_xlim([0, 50000]) 
    ax.set_title("Aperçu du LFO du Filtre")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Fréquence de coupure (Hz)")
    ax.legend()
    st.pyplot(fig)

# Appliquer le filtre après oversampling
filtered_signal = apply_filter_after_oversampling(lfo_signal, filter_lfo, filter_type)

# Affichage de la transformée de Fourier du signal filtré
st.subheader("Transformée de Fourier du Signal Filtré")
fft_filtered_signal = np.fft.fft(filtered_signal)
fft_filtered_freqs = np.fft.fftfreq(len(fft_filtered_signal), 1 / 44100)
fig, ax = plt.subplots()
ax.plot(fft_filtered_freqs[:len(fft_filtered_freqs)//2], np.abs(fft_filtered_signal)[:len(fft_filtered_signal)//2], color='orange', label="FFT Filtré")
ax.set_xlim([0, 20000])  # Limiter l'axe des x à 20 000 Hz
ax.set_title("Transformée de Fourier du Signal Filtré")
ax.set_xlabel("Fréquence (Hz)")
ax.set_ylabel("Amplitude (arbitraire)")
ax.legend()
st.pyplot(fig)

# Section ADSR
col9, col10 = st.columns(2)
with col9:
    st.subheader("🎯 Enveloppe ADSR")
    attack = st.slider("Attack (s)", 0.01, 2.0, 0.1)
    decay = st.slider("Decay (s)", 0.01, 2.0, 0.1)
    sustain = st.slider("Sustain (niveau)", 0.0, 1.0, 0.7)
    release = st.slider("Release (s)", 0.01, 2.0, 0.2)
with col10:
    t = np.linspace(0, duration, int(44100 * duration), endpoint=False)
    adsr_envelope = apply_adsr(np.ones_like(t), 44100, attack, decay, sustain, release)
    fig, ax = plt.subplots()
    ax.plot(t[:300000], adsr_envelope[:300000], color='red', label='ADSR')
    ax.set_title("Aperçu de l'enveloppe ADSR")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)

# Génération et lecture du signal final
adsr_signal = apply_adsr(filtered_signal, 44100, attack, decay, sustain, release)
audio_file = numpy_to_wav(adsr_signal)

if st.button("▶️ Jouer le son"):
    st.audio(audio_file, format="audio/wav")

st.info("🎧 Ajustez les paramètres et cliquez sur 'Jouer le son' pour écouter votre création sonore.")
