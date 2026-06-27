# -*- coding: utf-8 -*-
"""
services/tts/smallest_tts.py
=============================
Smallest.ai Lightning TTS service for the Pipecat AI pipeline.

Provides SmallestTTS, a drop-in alternative to KokoroTTS / CartesiaTTS that
uses the Smallest.ai Waves REST API.  The public interface mirrors the other
TTS services so server.py can swap providers via a single config change.

Install:
    pip install requests   (already a transitive dep, but listed for clarity)

Environment:
    SMALLEST_API_KEY  – your Smallest.ai API key (set in .env)

Available voices (subset):
    emily, meher, sarah, olivia, aria, kavya, mia, luna, zara
    See https://app.smallest.ai for the full voice catalog.

API endpoint: https://api.smallest.ai/waves/v1/tts
"""

import io
import os
import time
import wave
import asyncio
from typing import AsyncIterator

import numpy as np
import requests
from loguru import logger

# ---------------------------------------------------------------------------
# Constants  (match KokoroTTS / CartesiaTTS so the rest of the pipeline is unchanged)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 24_000  # output sample rate in Hz

# Curated subset of confirmed Smallest.ai voices
# Standard pool (lightning_v3.1): olivia, rachel, lauren, hannah, chloe
# Pro pool (lightning_v3.1_pro): kaitlyn, meher, sophie, savannah
AVAILABLE_VOICES: dict[str, str] = {
    # Standard pool — lightning_v3.1
    "olivia":   "olivia",    # US Female (recommended default)
    "rachel":   "rachel",    # US Female
    "lauren":   "lauren",    # US Female
    "hannah":   "hannah",    # US Female
    "chloe":    "chloe",     # Australian Female
    # Pro pool — lightning_v3.1_pro
    "kaitlyn":  "kaitlyn",   # US Female, natural American accent
    "meher":    "meher",     # Indian Female, English + Hindi
    "sophie":   "sophie",    # UK Female, British accent
    "savannah": "savannah",  # US Female
}

DEFAULT_VOICE_ID = "olivia"
DEFAULT_MODEL    = "lightning_v3.1"

_API_URL = "https://api.smallest.ai/waves/v1/tts"


# ---------------------------------------------------------------------------
# SmallestTTS Service
# ---------------------------------------------------------------------------

class SmallestTTS:
    """
    Smallest.ai Lightning Text-to-Speech service.

    Usage
    -----
    Basic (blocking):
        tts = SmallestTTS(api_key="sk_...")
        tts.speak("Hello, how can I help you today?")

    Pipeline (non-blocking, streams from LLM):
        async for sentence in llm_store.invoke_stream(messages):
            tts.speak(sentence)

    Full async pipeline (recommended for production):
        await tts.speak_stream(llm_store.invoke_stream(messages))
    """

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str = DEFAULT_VOICE_ID,
        model_id: str = DEFAULT_MODEL,
        speed: float = 1.0,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        """
        Initialize SmallestTTS.

        Parameters
        ----------
        api_key:     Smallest.ai API key.  Falls back to SMALLEST_API_KEY env var.
        voice_id:    Voice identifier (see AVAILABLE_VOICES or app.smallest.ai).
        model_id:    Model to use (default: "lightning_v3.1_pro").
        speed:       Speech speed multiplier (0.5–2.0, 1.0 = normal). Note: the
                     Smallest.ai API does not expose a speed param directly; this
                     is stored for potential future use.
        sample_rate: Output sample rate (8000, 16000, 24000, or 44100 Hz).
        """
        self._api_key = api_key or os.getenv("SMALLEST_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Smallest.ai API key is required. Set SMALLEST_API_KEY in .env "
                "or pass api_key= to SmallestTTS()."
            )

        self.voice_id    = voice_id
        self.model_id    = model_id
        self.speed       = speed
        self.sample_rate = sample_rate

        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "audio/wav",
        })

        logger.info(
            f"[SmallestTTS] Initialized  (model={model_id}, "
            f"voice={voice_id}, sample_rate={sample_rate})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pcm_bytes_to_float32(self, pcm_bytes: bytes) -> np.ndarray:
        """Convert raw PCM s16le bytes to a float32 numpy array in [-1, 1]."""
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
        return pcm.astype(np.float32) / 32768.0

    def _wav_bytes_to_float32(self, wav_bytes: bytes) -> np.ndarray:
        """Read WAV bytes and return a float32 numpy array."""
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            n_frames = wf.getnframes()
            raw_pcm  = wf.readframes(n_frames)
        return self._pcm_bytes_to_float32(raw_pcm)

    def _float32_to_wav_bytes(self, audio: np.ndarray) -> bytes:
        """Wrap a float32 numpy array in a WAV container (PCM s16le)."""
        pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)       # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Core synthesis
    # ------------------------------------------------------------------

    def synthesize_wav_bytes(self, text: str) -> bytes:
        """
        Synthesize *text* and return raw WAV bytes.

        Calls the Smallest.ai REST API synchronously and returns the WAV
        audio data, suitable for streaming directly to the browser.
        """
        if not text or not text.strip():
            return b""

        payload = {
            "text":          text.strip(),
            "voice_id":      self.voice_id,
            "model":         self.model_id,
            "sample_rate":   self.sample_rate,
            "output_format": "wav",
        }

        try:
            resp = self._session.post(_API_URL, json=payload, timeout=30)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            logger.error(
                f"[SmallestTTS] HTTP error {exc.response.status_code}: "
                f"{exc.response.text[:300]}"
            )
            return b""
        except requests.exceptions.RequestException as exc:
            logger.error(f"[SmallestTTS] Request error: {exc}")
            return b""

        return resp.content

    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize *text* and return a float32 numpy array at `sample_rate` Hz.
        """
        if not text or not text.strip():
            return np.array([], dtype=np.float32)

        wav_bytes = self.synthesize_wav_bytes(text)
        if not wav_bytes:
            return np.array([], dtype=np.float32)

        audio = self._wav_bytes_to_float32(wav_bytes)

        # Pad with 0.25s of silence (matches KokoroTTS / CartesiaTTS behavior)
        silence = np.zeros(int(self.sample_rate * 0.25), dtype=np.float32)
        return np.concatenate([audio, silence])

    # ------------------------------------------------------------------
    # Playback helpers
    # ------------------------------------------------------------------

    def play(self, audio: np.ndarray) -> None:
        """Play a float32 audio array (blocking until done)."""
        if len(audio) == 0:
            return
        import sounddevice as sd
        sd.play(audio, samplerate=self.sample_rate)
        sd.wait()

    def speak(self, text: str) -> None:
        """
        Synthesize *text* and play it immediately (blocking).
        """
        t0 = time.perf_counter()
        audio = self.synthesize(text)
        synth_s = time.perf_counter() - t0

        dur = len(audio) / self.sample_rate if len(audio) > 0 else 0
        rtf = synth_s / dur if dur > 0 else 0
        logger.debug(
            f"[SmallestTTS] synth={synth_s:.3f}s  dur={dur:.2f}s  RTF={rtf:.2f}x"
        )

        self.play(audio)

    # ------------------------------------------------------------------
    # Full async pipeline: consume LLM sentence stream -> speak
    # ------------------------------------------------------------------

    async def speak_stream(
        self,
        sentence_stream: AsyncIterator[str],
    ) -> str:
        """
        Consume an async sentence generator and speak each sentence as it
        arrives.  Returns the full concatenated response text.
        """
        full_response = []
        first = True
        total_start = time.perf_counter()

        async for sentence in sentence_stream:
            sentence = sentence.strip()
            if not sentence:
                continue

            full_response.append(sentence)

            # Synthesize in thread pool so the event loop stays responsive
            t0 = time.perf_counter()
            audio = await asyncio.get_event_loop().run_in_executor(
                None, self.synthesize, sentence
            )
            synth_s = time.perf_counter() - t0

            if first:
                ttft = time.perf_counter() - total_start
                logger.info(f"[SmallestTTS] TTFT = {ttft:.3f}s")
                first = False

            dur = len(audio) / self.sample_rate if len(audio) > 0 else 0
            rtf = synth_s / dur if dur > 0 else 0
            logger.debug(
                f"[SmallestTTS] Sentence: '{sentence[:60]}...' | "
                f"synth={synth_s:.3f}s  dur={dur:.2f}s  RTF={rtf:.2f}x"
            )

            await asyncio.get_event_loop().run_in_executor(
                None, self.play, audio
            )

        logger.info(
            f"[SmallestTTS] Done. Total elapsed = "
            f"{time.perf_counter() - total_start:.2f}s"
        )
        return " ".join(full_response)

    # ------------------------------------------------------------------
    # Async streaming for browser WebSocket delivery
    # ------------------------------------------------------------------

    async def stream_to_client(
        self,
        sentence_stream: AsyncIterator[str],
    ) -> AsyncIterator[tuple[str, bytes]]:
        """
        Consume an async sentence generator and yield (sentence, wav_bytes)
        pairs for each sentence.

        The server sends these over WebSocket to the browser.
        """
        first = True
        total_start = time.perf_counter()

        async for sentence in sentence_stream:
            sentence = sentence.strip()
            if not sentence:
                continue

            t0 = time.perf_counter()
            wav_bytes = await asyncio.get_event_loop().run_in_executor(
                None, self.synthesize_wav_bytes, sentence
            )
            synth_s = time.perf_counter() - t0

            if first:
                ttft = time.perf_counter() - total_start
                logger.info(f"[SmallestTTS] TTFT (client stream) = {ttft:.3f}s")
                first = False

            logger.debug(
                f"[SmallestTTS] WAV chunk: '{sentence[:60]}...' | "
                f"synth={synth_s:.3f}s  size={len(wav_bytes)} bytes"
            )

            yield sentence, wav_bytes

        logger.info(
            f"[SmallestTTS] Client stream done. Total = "
            f"{time.perf_counter() - total_start:.2f}s"
        )
