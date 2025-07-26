from openai import OpenAI

def transcribe_audio(file_path):
    client = OpenAI()
    audio_file = open(file_path, "rb")

    transcription = client.audio.transcriptions.create(
        model="gpt-4o-transcribe", 
        file=audio_file, 
        response_format="text"
    )

    print(transcription)
    return transcription