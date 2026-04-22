"""Speech-to-Text services."""
import base64
from abc import ABC, abstractmethod
from loguru import logger


class STTProvider(ABC):
    """Abstract base class for STT providers."""

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes to text."""
        pass


class DeepgramSTT(STTProvider):
    """Deepgram Speech-to-Text provider."""

    def __init__(self, api_key: str, model: str = "nova-2", language: str = "en"):
        """Initialize Deepgram STT."""
        from deepgram import DeepgramClient

        self.client = DeepgramClient(api_key=api_key)
        self.model = model
        self.language = language

    async def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes using Deepgram."""
        import asyncio

        try:
            logger.debug(f"Transcribing {len(audio_bytes)} bytes of audio with Deepgram")
            response = await asyncio.to_thread(self._transcribe_sync, audio_bytes)
            
            # Extract transcript from response
            # Response format: response.results.channels[0].alternatives[0].transcript
            if (
                response.results
                and response.results.channels
                and len(response.results.channels) > 0
                and response.results.channels[0].alternatives
                and len(response.results.channels[0].alternatives) > 0
            ):
                transcript = response.results.channels[0].alternatives[0].transcript
                logger.info(f"Transcribed: {transcript}")
                return transcript
            else:
                logger.warning("No transcription result from Deepgram")
                return ""
        except Exception as e:
            logger.error(f"Deepgram transcription error: {e}")
            raise

    def _transcribe_sync(self, audio_bytes: bytes):
        """Synchronous transcription using Deepgram SDK v6.1.1."""
        # Use Deepgram v1 API for prerecorded transcription
        # API: transcribe_file(source, options)
        # where source is the audio data and options are keyword arguments
        
        response = self.client.listen.v1.media.transcribe_file(
            request=audio_bytes,
            model=self.model,
            language=self.language,
            smart_format=True,
        )
        
        return response
