from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import aiofiles
from stt import AudioTranscriber
import asyncio
from starlette.responses import HTMLResponse
import logging
import uvicorn
from llm import get_llm_response
import wave
import simpleaudio as sa
from openai import OpenAI
import os
from tts import TTSManager

# Initialize FastAPI app and OpenAI client
app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up logging
logging.basicConfig(level=logging.INFO)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for storing streams
DATA_DIR = Path("./data")
VIDEO_DIR = DATA_DIR / "video"
AUDIO_DIR = DATA_DIR / "audio"
TEXT_DIR = DATA_DIR / "text"

for dir in [VIDEO_DIR, AUDIO_DIR, TEXT_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

transcribers = {}

async def on_transcript_callback(transcript: str, websocket: WebSocket, tts_manager: TTSManager):
    logging.info(f"Transcribed: {transcript}")
    AudioTranscriber.send_transcript_to_frontend(transcript, True)
    
    try:
        print("\nLLM Response:")
        print("-" * 50)
        buffer = []  
        
        async for response_chunk in get_llm_response(transcript):
            if response_chunk:
                await websocket.send_text(response_chunk)
                buffer.append(response_chunk)
                
                # Process TTS when we have a complete sentence or enough text
                if any(char in response_chunk for char in '.!?') or len(''.join(buffer)) > 100:
                    text_to_speak = ''.join(buffer)
                    buffer = []  # Clear buffer
                    
                    # Get complete audio response for the sentence
                    audio_data = await tts_manager.get_complete_audio(text_to_speak)
                    if audio_data:
                        # Send the complete audio segment
                        await websocket.send_bytes(bytes([3]) + audio_data)

        # Process any remaining text in buffer
        if buffer:
            text_to_speak = ''.join(buffer)
            audio_data = await tts_manager.get_complete_audio(text_to_speak)
            if audio_data:
                await websocket.send_bytes(bytes([3]) + audio_data)

    except Exception as e:
        logging.error(f"Error in LLM processing: {e}")
        await websocket.send_text(f"Error: {str(e)}")
    logging.info(f"Transcribed: {transcript}")
    
    
@app.get("/")
async def get():
    return HTMLResponse(content=open("index.html").read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Create TTS manager for this connection
    tts_manager = TTSManager()
    
    # Create callback with websocket and TTS manager
    callback = lambda transcript: on_transcript_callback(transcript, websocket, tts_manager)
    
    # Create a new transcriber instance for this connection
    transcriber = AudioTranscriber(callback)
    transcriber.websocket = websocket 
    await transcriber.init_connection()

    # Store the transcriber instance
    connection_id = id(websocket)
    transcribers[connection_id] = transcriber

    try:
        async for message in websocket.iter_bytes():
            data_type = message[0]
            data = message[1:]

            if data_type == 0:  # Text
                async with aiofiles.open(TEXT_DIR / "chat.txt", mode="ab") as file:
                    await file.write(data + b'\n')

            elif data_type == 1:  # Video
                async with aiofiles.open(VIDEO_DIR / "stream.h264", mode="ab") as file:
                    await file.write(data)

            elif data_type == 2:  # Audio
                async with aiofiles.open(AUDIO_DIR / "stream.aac", mode="ab") as file:
                    await file.write(data)
                await transcriber.process_audio(data)
                

    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")
        tts_manager.stop_streaming()  # Stop any ongoing TTS streams
    except Exception as e:
        logging.error(f"Error in websocket connection: {e}")
    finally:
        if connection_id in transcribers:
            await transcribers[connection_id].close()
            del transcribers[connection_id]

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)