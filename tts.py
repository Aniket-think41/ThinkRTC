import os
from openai import OpenAI
import asyncio
import logging
logging.basicConfig(level=logging.INFO)

class TTSManager:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.should_stop = asyncio.Event()

    async def get_complete_audio(self, text: str) -> bytes:
        """Generate complete audio response for a text segment"""
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="shimmer",
                input=text,
                response_format="mp3"
            )
            return response.content
            
        except Exception as e:
            logging.error(f"Error in TTS: {e}")
            return b''