from openai import OpenAI

def transcribe_audio(file_path):
    client = OpenAI()
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-transcribe", 
            file=audio_file, 
            response_format="text"
        )
    print(transcription)
    return transcription