from elevenlabs import generate, play
import os

def gen_dub(text):
    print("Generating audio...")
    audio = generate(
        text=text,
        voice="JBFqnCBsd6RMkjVDRZzb", # Insert voice model here!
        model="eleven_multilingual_v2",
        api_key=os.getenv("ELEVENLABS_API_KEY")
    )
    play(audio)