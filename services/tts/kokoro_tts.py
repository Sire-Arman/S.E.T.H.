# -*- coding: utf-8 -*-
"""
services/tts/kokoro_tts.py
==========================
Kokoro-ONNX TTS service for the Pipecat AI pipeline.

Provides KokoroTTS, a lightweight wrapper around kokoro-onnx that:
  - Loads (and auto-downloads) the ONNX model once at instantiation
  - Exposes synthesize()        -> np.ndarray  (single text block)
  - Exposes speak()             -> plays audio to the default output device
  - Exposes speak_stream()      -> consumes an async sentence generator from
                                   LLMStore.invoke_stream() and plays each
                                   sentence as soon as it arrives (true pipeline)

Install:
    pip install kokoro-onnx sounddevice numpy

Model files (~350 MB) are downloaded from GitHub releases on first use and
cached in the same directory as this file.
"""

import os
import re
import sys
import time
import asyncio
import urllib.request
from typing import AsyncIterator

import numpy as np
import sounddevice as sd
from loguru import logger
from kokoro_onnx import Kokoro

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 24_000       # Kokoro always outputs 24 kHz mono audio

# Built-in voice ids shipped with kokoro-onnx
# prefix: a=American  b=British   f=female  m=male
AVAILABLE_VOICES: list[str] = [
    "af_heart",    # US female, warm & expressive  (recommended default)
    "af_bella",    # US female, clear & professional
    "af_nicole",   # US female, breathy & conversational
    "af_sarah",    # US female, bright & friendly
    "am_adam",     # US male,   deep & confident
    "am_michael",  # US male,   neutral, assistant-like
    "bf_emma",     # GB female, crisp
    "bm_george",   # GB male,   authoritative
    "bm_lewis",    # GB male,   casual
]

# ---------------------------------------------------------------------------
# Model file management
# ---------------------------------------------------------------------------
_SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_FILE  = os.path.join(_SERVICE_DIR, "kokoro-v1.0.onnx")
_VOICES_FILE = os.path.join(_SERVICE_DIR, "voices-v1.0.bin")
_GH_RELEASE  = (
    "https://github.com/thewh1teagle/kokoro-onnx/"
    "releases/download/model-files-v1.0/"
)


def _download_if_missing(local_path: str, filename: str) -> None:
    """Download a model file from GitHub releases if not already present."""
    if os.path.exists(local_path):
        return

    url = _GH_RELEASE + filename
    sizes = {"kokoro-v1.0.onnx": "~310 MB", "voices-v1.0.bin": "~40 MB"}
    logger.info(f"[KokoroTTS] Downloading {filename} {sizes.get(filename, '')} ...")
    logger.info(f"[KokoroTTS] URL: {url}")

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(block_num * block_size / total_size * 100, 100)
            print(f"\r  Downloading {filename}: {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, local_path, reporthook=_progress)
    print()  # newline after progress
    logger.info(f"[KokoroTTS] Saved -> {local_path}")


# ---------------------------------------------------------------------------
# KokoroTTS Service
# ---------------------------------------------------------------------------

class KokoroTTS:
    """
    Kokoro-ONNX Text-to-Speech service.

    Usage
    -----
    Basic (blocking):
        tts = KokoroTTS()
        tts.speak("Hello, how can I help you today?")

    Pipeline (non-blocking, streams from LLM):
        async for sentence in llm_store.invoke_stream(messages):
            tts.speak(sentence)          # synthesizes + plays per sentence

    Full async pipeline (recommended for production):
        await tts.speak_stream(llm_store.invoke_stream(messages))
    """

    def __init__(
        self,
        voice: str = "af_heart",
        speed: float = 1.0,
        model_file: str = _MODEL_FILE,
        voices_file: str = _VOICES_FILE,
    ) -> None:
        """
        Initialize KokoroTTS.

        Parameters
        ----------
        voice:        Kokoro voice id (see AVAILABLE_VOICES).
        speed:        Speech speed multiplier (0.5 = slow, 1.0 = normal, 1.5 = fast).
        model_file:   Path to kokoro-v1.0.onnx   (auto-downloaded if missing).
        voices_file:  Path to voices-v1.0.bin    (auto-downloaded if missing).
        """
        if voice not in AVAILABLE_VOICES:
            logger.warning(
                f"[KokoroTTS] Unknown voice '{voice}'. "
                f"Valid choices: {AVAILABLE_VOICES}. Falling back to 'af_heart'."
            )
            voice = "af_heart"

        self.voice = voice
        self.speed = speed

        logger.info("[KokoroTTS] Loading model ...")
        t0 = time.perf_counter()

        _download_if_missing(model_file,  "kokoro-v1.0.onnx")
        _download_if_missing(voices_file, "voices-v1.0.bin")

        self._kokoro = Kokoro(model_file, voices_file)

        elapsed = time.perf_counter() - t0
        logger.info(f"[KokoroTTS] Model ready in {elapsed:.2f}s  (voice={voice})")

    # ------------------------------------------------------------------
    # Core synthesis
    # ------------------------------------------------------------------

    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize *text* and return a float32 numpy array at 24 kHz.

        This is a synchronous, CPU-bound call. For a single sentence on a
        modern CPU it typically completes in 0.3 – 2 s depending on length.
        """
        if not text or not text.strip():
            return np.array([], dtype=np.float32)

        audio, _ = self._kokoro.create(
            text.strip(),
            voice=self.voice,
            speed=self.speed,
            lang="en-us",
        )
        return audio.astype(np.float32)

    # ------------------------------------------------------------------
    # Playback helpers
    # ------------------------------------------------------------------

    def play(self, audio: np.ndarray) -> None:
        """Play a float32 audio array (blocking until done)."""
        if len(audio) == 0:
            return
        sd.play(audio, samplerate=SAMPLE_RATE)
        sd.wait()

    def speak(self, text: str) -> None:
        """
        Synthesize *text* and play it immediately (blocking).

        Use this when you have a complete sentence ready, e.g. inside an
        `async for sentence in llm_store.invoke_stream(messages)` loop.
        """
        t0 = time.perf_counter()
        audio = self.synthesize(text)
        synth_s = time.perf_counter() - t0

        dur = len(audio) / SAMPLE_RATE if len(audio) > 0 else 0
        rtf = synth_s / dur if dur > 0 else 0
        logger.debug(
            f"[KokoroTTS] synth={synth_s:.3f}s  dur={dur:.2f}s  RTF={rtf:.2f}x"
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

        Synthesis runs in a thread pool so it does NOT block the event loop.
        Playback is sequential (each sentence plays fully before the next
        starts), which gives natural speech flow.

        Returns the full concatenated response text (for logging / history).

        Example
        -------
            full_text = await tts.speak_stream(
                llm_store.invoke_stream(messages)
            )
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
                logger.info(f"[KokoroTTS] TTFT = {ttft:.3f}s")
                first = False

            dur = len(audio) / SAMPLE_RATE if len(audio) > 0 else 0
            rtf = synth_s / dur if dur > 0 else 0
            logger.debug(
                f"[KokoroTTS] Sentence: '{sentence[:60]}...' | "
                f"synth={synth_s:.3f}s  dur={dur:.2f}s  RTF={rtf:.2f}x"
            )

            # Play synchronously in executor so we don't block the event loop
            await asyncio.get_event_loop().run_in_executor(
                None, self.play, audio
            )

        logger.info(
            f"[KokoroTTS] Done. Total elapsed = "
            f"{time.perf_counter() - total_start:.2f}s"
        )
        return " ".join(full_response)
