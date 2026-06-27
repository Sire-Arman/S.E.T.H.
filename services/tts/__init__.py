"""TTS services module."""
from .kokoro_tts import KokoroTTS, AVAILABLE_VOICES as KOKORO_VOICES, SAMPLE_RATE
from .cartesia_tts import CartesiaTTS, AVAILABLE_VOICES as CARTESIA_VOICES
from .smallest_tts import SmallestTTS, AVAILABLE_VOICES as SMALLEST_VOICES

__all__ = [
    "KokoroTTS",
    "CartesiaTTS",
    "SmallestTTS",
    "KOKORO_VOICES",
    "CARTESIA_VOICES",
    "SMALLEST_VOICES",
    "SAMPLE_RATE",
]
