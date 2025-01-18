import torch
import torchaudio
from torchaudio.transforms import Resample
import pyaudio
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

class VoiceActivityDetector:
    def __init__(self, threshold=0.5, sample_rate=8000):
        self.threshold = threshold
        self.sample_rate = sample_rate

        # Load Silero VAD model
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        self.model.eval()

        # PyAudio setup
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,  # Use a higher sample rate for better quality, but resample to 8000 Hz
            input=True,
            frames_per_buffer=1024
        )

        # Resampler if the sample rate is not 8000 Hz
        self.resampler = Resample(orig_freq=16000, new_freq=self.sample_rate)
        self.silence_duration = 0  # Track silence duration
        self.silence_threshold = 2  # 2 seconds of silence


    # ... existing code ...

    def process_audio(self, audio_data=None):
        if audio_data is not None:
            # Log the size of the audio data for debugging
            logging.info(f"Received audio data size: {len(audio_data)} bytes")
            
            if len(audio_data) % 2 != 0:
                logging.error("Audio data size is not a multiple of 2, indicating an issue with the data.")
                return  # Skip processing if the size is not valid

            audio_tensor = self._convert_audio_to_tensor(audio_data)
            
            # Resample the audio to 8000 Hz if necessary
            audio_tensor = self.resampler(audio_tensor)

            # Log the shape of the tensor after resampling
            logging.info(f"Audio tensor shape after resampling: {audio_tensor.shape}")

            # Ensure the tensor has the correct shape for the model
            if audio_tensor.shape[-1] != 256:  # For 8000 Hz
                logging.error(f"Tensor shape is incorrect: {audio_tensor.shape}. Expected shape ending with 256.")
                return  # Skip if the shape is not correct

            # Perform voice activity detection
            with torch.no_grad():
                predictions = self.model(audio_tensor, self.sample_rate)

            if predictions.item() > self.threshold:
                print("Voice detected!")
                self.silence_duration = 0  # Reset silence duration
            else:
                self.silence_duration += 0.1  # Increment silence duration
                if self.silence_duration >= self.silence_threshold:
                    print("Silence detected!")
                    self.silence_duration = 0  # Reset after printing
                    
    def _convert_audio_to_tensor(self, audio_data):
    # Check the size of the audio data
        if len(audio_data) % 2 != 0:
            raise ValueError("Audio data size is not a multiple of 2, indicating an issue with the data.")

        # Convert the audio data to a numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
        return audio_tensor
    def cleanup(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

if __name__ == "__main__":
    vad = VoiceActivityDetector()
    vad.process_audio()
