from elevenlabs.client import ElevenLabs
from elevenlabs import play, stream
import os

client = ElevenLabs(
  api_key=os.getenv("ELEVENLABS_API_KEY"),
)

def gen_dub(text):
    print("Generating audio...")
    audio = client.generate(
        text=text,
        voice="JBFqnCBsd6RMkjVDRZzb", # Insert voice model here!
        model="eleven_multilingual_v2"
    )
    play(audio)