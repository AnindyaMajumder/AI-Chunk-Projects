
import sounddevice as sd
import numpy as np
import queue
import threading
import tempfile
import os
from tts import voice as tts_voice
from translator import translator
from transcribe import transcribe_audio

samplerate = 16000
channels = 1
duration = 5  # seconds to record

def record_audio(filename, duration=duration, samplerate=samplerate, channels=channels):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='int16')
    sd.wait()
    # Save as WAV
    import wave
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16 bits
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())
    print(f"Audio saved to {filename}")


if __name__ == "__main__":
    target_language = input("Enter target language (e.g., French, Spanish): ")
    
    # Step 1: Record audio
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
        audio_path = tmpfile.name
    record_audio(audio_path)

    # Step 2: Transcribe
    print("Transcribing...")
    text = transcribe_audio(audio_path)
    print(f"Transcribed: {text}")

    # Step 3: Translate
    translated_text = translator(text, target_language)
    print(f"Translated: {translated_text}")

    # Step 4: TTS
    print("Synthesizing speech...")
    tts_voice(translated_text)
    print("Audio output saved as output_audio.wav")

    # Cleanup
    os.remove(audio_path)
