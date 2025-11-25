# import openai

# class AudioTranscriber:
#     def __init__(self, api_key):
#         openai.api_key = api_key

#     def transcribe(self, audio_path):
#         with open(audio_path, "rb") as f:
#             result = openai.audio.transcriptions.create(
#                 model="whisper-1",
#                 file=f
#             )
#         return result["text"]


from typing import Dict, Any, cast 
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

class AudioTranscriber:
    def __init__(self, model_name: str = "openai/whisper-small"):
        """
        Initialize the Audio Transcriber with a local Whisper model.
        
        Args:
            model_name: HuggingFace Whisper model (e.g., 'openai/whisper-small' for speed, 'openai/whisper-base' for accuracy).
        """
        self.transcriber = pipeline("automatic-speech-recognition", model=model_name)
        logger.info(f"Loaded Whisper model: {model_name}")

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to the audio file (supports WAV, MP3, etc.).
        
        Returns:
            Transcribed text.
        """
        try:
            result = cast(Dict[str, Any], self.transcriber(audio_path))
            text = result["text"]
            logger.info(f"Transcribed audio: {audio_path}")
            return text
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return ""