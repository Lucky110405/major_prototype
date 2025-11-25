import openai

class AudioTranscriber:
    def __init__(self, api_key):
        openai.api_key = api_key

    def transcribe(self, audio_path):
        with open(audio_path, "rb") as f:
            result = openai.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        return result["text"]
