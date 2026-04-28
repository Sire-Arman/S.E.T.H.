# -*- coding: utf-8 -*-
"""
services/tts/cartesia_tts.py
=============================
Cartesia TTS service for the Pipecat AI pipeline.

Provides CartesiaTTS, a drop-in alternative to KokoroTTS that uses the
Cartesia Sonic cloud API via WebSocket streaming.  The public interface
mirrors KokoroTTS so the rest of the pipeline (server.py, etc.) can
swap providers with a single config change.

Install:
    pip install 'cartesia[websockets]'

Environment:
    CARTESIA_API_KEY  – your Cartesia API key (set in .env)

Available voices (Cartesia voice IDs):
    See https://play.cartesia.ai for the full voice catalog.
    Default voice: "Barbershop Man" (f786b574-daa5-4673-aa0c-cbe3e8534c02)
"""

import io
import os
import time
import wave
import struct
import asyncio
from typing import AsyncIterator

import numpy as np
from loguru import logger
from cartesia import Cartesia, AsyncCartesia

# ---------------------------------------------------------------------------
# Constants  (match KokoroTTS so the rest of the pipeline is unchanged)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 24_000  # output sample rate in Hz (matches Kokoro for consistency)

# Curated subset of popular Cartesia voices
AVAILABLE_VOICES: dict[str, str] = {
    "barbershop_man":   "f786b574-daa5-4673-aa0c-cbe3e8534c02",
    "british_lady":     "79a125e8-cd45-4c13-8a67-188112f4dd22",
    "newsman":          "d46abd1d-2571-4413-b219-f84b5fb0d145",
    "reading_lady":     "15a9cd88-84b0-4a8b-95f2-5d583b54c72e",
    "maria":            "5345cf08-6f37-424d-a5d9-8ae1c1bae2a6",
    "midwestern_woman": "11af83e2-23eb-452f-956e-7fee218ccb5c",
    "sportsman":        "f9836c6e-a0bd-460e-9d3c-f7299fa60f94",
    "southern_man":     "a167e0f3-df7e-4d52-a9c3-f949145f52bd",
}

# Default voice and model
DEFAULT_VOICE_ID = "f786b574-daa5-4673-aa0c-cbe3e8534c02"  # Barbershop Man
DEFAULT_MODEL = "sonic-3"


# ---------------------------------------------------------------------------
# CartesiaTTS Service
# ---------------------------------------------------------------------------

class CartesiaTTS:
    """
    Cartesia cloud Text-to-Speech service.

    Usage
    -----
    Basic (blocking):
        tts = CartesiaTTS(api_key="sk_car_...")
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
    ) -> None:
        """
        Initialize CartesiaTTS.

        Parameters
        ----------
        api_key:    Cartesia API key.  Falls back to CARTESIA_API_KEY env var.
        voice_id:   Cartesia voice UUID (see AVAILABLE_VOICES or play.cartesia.ai).
        model_id:   Model to use (default: "sonic-3").
        speed:      Speech speed multiplier (0.5 = slow, 1.0 = normal, 1.5 = fast).
        """
        self._api_key = api_key or os.getenv("CARTESIA_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Cartesia API key is required. Set CARTESIA_API_KEY in .env "
                "or pass api_key= to CartesiaTTS()."
            )

        self.voice_id = voice_id
        self.model_id = model_id
        self.speed = speed

        # Sync client for blocking calls
        self._client = Cartesia(api_key=self._api_key)

        logger.info(
            f"[CartesiaTTS] Initialized  (model={model_id}, "
            f"voice={voice_id}, speed={speed})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _voice_spec(self) -> dict:
        """Return the voice specification dict for Cartesia API calls."""
        return {"mode": "id", "id": self.voice_id}

    def _output_format_raw(self) -> dict:
        """Raw PCM output format for WebSocket streaming."""
        return {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": SAMPLE_RATE,
        }

    def _output_format_wav(self) -> dict:
        """WAV container format for one-shot generation."""
        return {
            "container": "wav",
            "encoding": "pcm_s16le",
            "sample_rate": SAMPLE_RATE,
        }

    def _pcm_bytes_to_float32(self, pcm_bytes: bytes) -> np.ndarray:
        """Convert raw PCM s16le bytes to a float32 numpy array in [-1, 1]."""
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
        return pcm.astype(np.float32) / 32768.0

    def _pcm_bytes_to_wav(self, pcm_bytes: bytes) -> bytes:
        """Wrap raw PCM s16le bytes in a WAV container."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Core synthesis (sync, via WebSocket for streaming efficiency)
    # ------------------------------------------------------------------

    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize *text* and return a float32 numpy array at 24 kHz.

        Uses Cartesia's sync WebSocket API to stream chunks and
        concatenates them into a single array.
        """
        if not text or not text.strip():
            return np.array([], dtype=np.float32)

        chunks: list[bytes] = []

        with self._client.tts.websocket_connect() as connection:
            ctx = connection.context(
                model_id=self.model_id,
                voice=self._voice_spec(),
                output_format=self._output_format_raw(),
            )
            ctx.push(text.strip())
            ctx.no_more_inputs()

            for response in ctx.receive():
                if response.type == "chunk" and response.audio:
                    chunks.append(response.audio)
                elif response.type == "done":
                    break

        if not chunks:
            return np.array([], dtype=np.float32)

        pcm_bytes = b"".join(chunks)
        audio = self._pcm_bytes_to_float32(pcm_bytes)

        # Pad with 0.25s of silence (matches KokoroTTS behavior)
        silence = np.zeros(int(SAMPLE_RATE * 0.25), dtype=np.float32)
        return np.concatenate([audio, silence])

    # ------------------------------------------------------------------
    # Playback helpers
    # ------------------------------------------------------------------

    def play(self, audio: np.ndarray) -> None:
        """Play a float32 audio array (blocking until done)."""
        if len(audio) == 0:
            return
        import sounddevice as sd
        sd.play(audio, samplerate=SAMPLE_RATE)
        sd.wait()

    def speak(self, text: str) -> None:
        """
        Synthesize *text* and play it immediately (blocking).

        Use this when you have a complete sentence ready.
        """
        t0 = time.perf_counter()
        audio = self.synthesize(text)
        synth_s = time.perf_counter() - t0

        dur = len(audio) / SAMPLE_RATE if len(audio) > 0 else 0
        rtf = synth_s / dur if dur > 0 else 0
        logger.debug(
            f"[CartesiaTTS] synth={synth_s:.3f}s  dur={dur:.2f}s  RTF={rtf:.2f}x"
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
        Consume an async sentence generator (from LLMStore.invoke_stream) and
        speak each sentence as soon as it arrives.

        Returns the full concatenated response text.
        """
        full_response = []
        first = True
        total_start = time.perf_counter()

        async for sentence in sentence_stream:
            sentence = sentence.strip()
            if not sentence:
                continue

            full_response.append(sentence)

            # Synthesize in a thread so the event loop stays responsive
            t0 = time.perf_counter()
            audio = await asyncio.get_event_loop().run_in_executor(
                None, self.synthesize, sentence
            )
            synth_s = time.perf_counter() - t0

            if first:
                ttft = time.perf_counter() - total_start
                logger.info(f"[CartesiaTTS] TTFT = {ttft:.3f}s")
                first = False

            dur = len(audio) / SAMPLE_RATE if len(audio) > 0 else 0
            rtf = synth_s / dur if dur > 0 else 0
            logger.debug(
                f"[CartesiaTTS] Sentence: '{sentence[:60]}...' | "
                f"synth={synth_s:.3f}s  dur={dur:.2f}s  RTF={rtf:.2f}x"
            )

            # Play synchronously in executor
            await asyncio.get_event_loop().run_in_executor(
                None, self.play, audio
            )

        logger.info(
            f"[CartesiaTTS] Done. Total elapsed = "
            f"{time.perf_counter() - total_start:.2f}s"
        )
        return " ".join(full_response)

    # ------------------------------------------------------------------
    # WAV encoding for browser streaming
    # ------------------------------------------------------------------

    def synthesize_wav_bytes(self, text: str) -> bytes:
        """
        Synthesize *text* and return raw WAV bytes (24 kHz, 16-bit mono).

        Used to stream audio back to the browser over WebSocket.
        """
        if not text or not text.strip():
            return b""

        chunks: list[bytes] = []

        with self._client.tts.websocket_connect() as connection:
            ctx = connection.context(
                model_id=self.model_id,
                voice=self._voice_spec(),
                output_format=self._output_format_raw(),
            )
            ctx.push(text.strip())
            ctx.no_more_inputs()

            for response in ctx.receive():
                if response.type == "chunk" and response.audio:
                    chunks.append(response.audio)
                elif response.type == "done":
                    break

        if not chunks:
            return b""

        pcm_bytes = b"".join(chunks)
        return self._pcm_bytes_to_wav(pcm_bytes)

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

            # Synthesize in thread pool
            t0 = time.perf_counter()
            wav_bytes = await asyncio.get_event_loop().run_in_executor(
                None, self.synthesize_wav_bytes, sentence
            )
            synth_s = time.perf_counter() - t0

            if first:
                ttft = time.perf_counter() - total_start
                logger.info(f"[CartesiaTTS] TTFT (client stream) = {ttft:.3f}s")
                first = False

            logger.debug(
                f"[CartesiaTTS] WAV chunk: '{sentence[:60]}...' | "
                f"synth={synth_s:.3f}s  size={len(wav_bytes)} bytes"
            )

            yield sentence, wav_bytes

        logger.info(
            f"[CartesiaTTS] Client stream done. Total = "
            f"{time.perf_counter() - total_start:.2f}s"
        )
