from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import os

load_dotenv()

elevenlabs = ElevenLabs(
  api_key=os.getenv("ELEVENLABS_API_KEY"),
)

def voice(text: str):
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
    # play(audio)
    