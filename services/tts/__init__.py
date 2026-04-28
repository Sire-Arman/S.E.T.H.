"""TTS services module."""
from .kokoro_tts import KokoroTTS, AVAILABLE_VOICES as KOKORO_VOICES, SAMPLE_RATE
from .cartesia_tts import CartesiaTTS, AVAILABLE_VOICES as CARTESIA_VOICES

__all__ = [
    "KokoroTTS",
    "CartesiaTTS",
    "KOKORO_VOICES",
    "CARTESIA_VOICES",
    "SAMPLE_RATE",
]
