# stt.py
import logging
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions, DeepgramClientOptions
from pathlib import Path
import asyncio
from functools import partial
import json

logging.basicConfig(level=logging.INFO)

API_KEY = "b348ee9fd0bcfaa7ef63b81bca9e74116d82db31"

config = DeepgramClientOptions(
    verbose=logging.WARN,
    options={"keepalive": "true"}
)

deepgram = DeepgramClient(API_KEY, config)

TEXT_DIR = Path("./data/text")
TEXT_DIR.mkdir(parents=True, exist_ok=True)

class AudioTranscriber:
    def __init__(self, on_transcript_callback):
        self.on_transcript_callback = on_transcript_callback
        self.dg_connection = None
        self.loop = asyncio.get_event_loop()
        self.websocket = None

    async def init_connection(self):
        self.dg_connection = deepgram.listen.live.v("1")

        self.dg_connection.on(LiveTranscriptionEvents.Open, self.on_open)
        self.dg_connection.on(LiveTranscriptionEvents.Transcript, self.on_message)
        self.dg_connection.on(LiveTranscriptionEvents.Close, self.on_close)
        self.dg_connection.on(LiveTranscriptionEvents.Error, self.on_error)

        options = LiveOptions(
            model="nova-2",
            language="en-US",
            interim_results=True,
            punctuate=True
        )

        if not self.dg_connection.start(options):
            logging.error("Failed to start connection")
            raise RuntimeError("Failed to start connection")

    def on_open(self, event, *args, **kwargs):
        logging.info(f"Connection opened: {event}")

    async def send_transcript_to_frontend(self, transcript, is_final=False):
        """Send transcript to frontend in real-time"""
        if self.websocket:
            message = {
                "type": "transcript",
                "text": transcript,
                "is_final": is_final
            }
            logging.info(f"Sending transcript to frontend: {message}")
            await self.websocket.send_text(json.dumps(message))

    def on_message(self, *args, **kwargs):
        result = kwargs.get('result')
        try:
            if hasattr(result, 'channel') and hasattr(result.channel, 'alternatives'):
                alternatives = result.channel.alternatives
                if alternatives and hasattr(alternatives[0], 'transcript'):
                    transcript = alternatives[0].transcript
                    if transcript:
                        is_final = hasattr(result, 'speech_final') and result.speech_final
                        
                        # Log the transcript
                        logging.info(f"Received transcript: {transcript} ({'final' if is_final else 'interim'})")
                        
                        # Send transcript to frontend immediately
                        future = asyncio.run_coroutine_threadsafe(
                            self.send_transcript_to_frontend(transcript, is_final),
                            self.loop
                        )
                        future.add_done_callback(self._handle_callback_result)
                        
                        # Only process final transcripts for LLM response
                        if is_final and transcript.strip():
                            future = asyncio.run_coroutine_threadsafe(
                                self.on_transcript_callback(transcript),
                                self.loop
                            )
                            future.add_done_callback(self._handle_callback_result)
                            
        except Exception as e:
            logging.error(f"Error processing message: {e}")
            
    def _handle_callback_result(self, future):
        try:
            future.result()
        except Exception as e:
            logging.error(f"Error in transcript callback: {e}")

    def on_close(self, close, *args, **kwargs):
        logging.info(f"Connection closed: {close}")

    def on_error(self, error, *args, **kwargs):
        logging.error(f"Error occurred: {error}")

    async def process_audio(self, audio_data):
        if self.dg_connection:
            self.dg_connection.send(audio_data)

    async def close(self):
        if self.dg_connection:
            try:
                logging.info("Stopping Deepgram connection...")
                await self.dg_connection.close()
                logging.info("Connection closed successfully.")
            except Exception as e:
                logging.error(f"Error stopping the connection: {e}")
            finally:
                self.dg_connection = None