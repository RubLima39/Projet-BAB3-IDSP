import numpy as np
import sounddevice as sd
import streamlit as st
import keyboard
import threading

# Fréquences des notes
notes = {
    "C": 261.63, "C#": 277.18, "D": 293.66, "D#": 311.13, "E": 329.63,
    "F": 349.23, "F#": 369.99, "G": 392.00, "G#": 415.30, "A": 440.00,
    "A#": 466.16, "B": 493.88, "C2": 523.25
}

# Mapping clavier -> note
key_to_note = {
    "a": "C", "w": "C#", "z": "D", "x": "D#", "e": "E",
    "q": "F", "r": "F#", "s": "G", "t": "G#", "d": "A",
    "y": "A#", "f": "B", "g": "C2"
}

# Liste des notes actuellement jouées
active_notes = set()
stream = None  # Flux audio

# Générer l'onde sonore
def generate_wave(frequencies, duration=1, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = sum(0.5 * np.sin(2 * np.pi * f * t) for f in frequencies) / len(frequencies)
    return wave.astype(np.float32)

# Fonction qui génère du son en continu
def audio_callback(outdata, frames, time, status):
    if status:
        print(status)
    if active_notes:
        wave = generate_wave([notes[n] for n in active_notes], duration=frames / 44100)
        outdata[:] = np.expand_dims(wave, axis=1)
    else:
        outdata.fill(0)  # Silence si aucune note jouée

# Démarrer la sortie audio
def start_audio_stream():
    global stream
    if stream is None:
        stream = sd.OutputStream(callback=audio_callback, samplerate=44100, channels=1)
        stream.start()

# Gérer la pression et le relâchement des touches
def listen_keyboard():
    while True:
        for key, note in key_to_note.items():
            if keyboard.is_pressed(key):
                active_notes.add(note)
            else:
                active_notes.discard(note)

# Interface Streamlit
st.title("🎹 Piano Virtuel - Moog Box")
st.write("Maintiens une touche du clavier ou clique sur une touche pour jouer en continu.")

# Interface visuelle du piano
st.markdown("""
<style>
.piano {
    display: flex;
    justify-content: center;
}
.white-key, .black-key {
    border: 2px solid black;
    text-align: center;
    font-size: 18px;
    cursor: pointer;
    user-select: none;
}
.white-key {
    width: 60px;
    height: 200px;
    background: white;
    color: black;
}
.black-key {
    width: 40px;
    height: 120px;
    background: black;
    color: white;
    margin-left: -20px;
    margin-right: -20px;
    z-index: 1;
}
</style>

<div class="piano">
    <div class="white-key" onclick="play('C')">A (C)</div>
    <div class="black-key" onclick="play('C#')">W (C#)</div>
    <div class="white-key" onclick="play('D')">Z (D)</div>
    <div class="black-key" onclick="play('D#')">X (D#)</div>
    <div class="white-key" onclick="play('E')">E (E)</div>
    <div class="white-key" onclick="play('F')">Q (F)</div>
    <div class="black-key" onclick="play('F#')">R (F#)</div>
    <div class="white-key" onclick="play('G')">S (G)</div>
    <div class="black-key" onclick="play('G#')">T (G#)</div>
    <div class="white-key" onclick="play('A')">D (A)</div>
    <div class="black-key" onclick="play('A#')">Y (A#)</div>
    <div class="white-key" onclick="play('B')">F (B)</div>
    <div class="white-key" onclick="play('C2')">G (C2)</div>
</div>
""", unsafe_allow_html=True)

# Démarrer l'écoute clavier et le son
if st.button("Activer le clavier 🎶"):
    st.success("Maintiens une touche pour jouer en continu !")
    start_audio_stream()
    threading.Thread(target=listen_keyboard, daemon=True).start()
