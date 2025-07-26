from openai import OpenAI
from elevenlabs.client import ElevenLabs
import os
import tempfile
import sounddevice as sd
import wave
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-transcribe", 
            file=audio_file, 
            response_format="text"
        )
    print(transcription)
    return transcription

def translator(text, target_language):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a translation assistant. Only translate the text, do not include any additional information."},
            {"role": "user", "content": f"Translate the following text to {target_language}: {text}"}
        ]
    )
    translated_text = response.choices[0].message.content.strip()
    print(f"Translated text: {translated_text}\n")
    return translated_text

elevenlabs = ElevenLabs(
  api_key=os.getenv("ELEVENLABS_API_KEY"),
)

def tts_voice(text: str):
    audio_gen = elevenlabs.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    # Save audio to file
    with open("output_audio.wav", "wb") as f:
        for chunk in audio_gen:
            f.write(chunk)
            
samplerate = 16000
channels = 1
duration = 25 # seconds to record
def record_audio(filename, duration=duration, samplerate=samplerate, channels=channels):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='int16')
    sd.wait()
    # Save as WAV
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