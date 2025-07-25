from dotenv import load_dotenv
load_dotenv()
from tts import gen_dub
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from translator import translate

SAMPLE_RATE = 16000
DURATION = 5  # seconds per chunk

def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

def transcribe_audio(audio, sample_rate=SAMPLE_RATE):
    # Save to temporary WAV file
    import tempfile, soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sample_rate)
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, info = model.transcribe(tmp.name)
        text = " ".join([seg.text for seg in segments])
    return text.strip()

def live_translate(language="French"):
    print(f"Speak into your microphone. Translating to {language}. Press Ctrl+C to exit.")
    while True:
        try:
            audio = record_audio()
            if np.max(np.abs(audio)) < 0.01:
                print("No speech detected. Try again.")
                continue
            text = transcribe_audio(audio)
            if text:
                print("Original:", text)
                translated = translate(text, language)
                print(f"Translated ({language}):", translated)
                gen_dub(translated)
            else:
                print("No transcription detected.")
        except KeyboardInterrupt:
            print("Exiting...")
            break

if __name__ == "__main__":
    import sys
    lang = sys.argv[1] if len(sys.argv) > 1 else "French"
    live_translate(lang)