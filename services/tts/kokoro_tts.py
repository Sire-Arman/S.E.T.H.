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

Model files are downloaded from GitHub releases on first use and
cached in the same directory as this file.

Optimisation notes
------------------
* Default model is the **INT8** variant (~88 MB, ~3.5× smaller than FP32).
* ONNX Runtime thread counts are pinned to the physical core count at
  import time via OMP_NUM_THREADS / ORT_NUM_THREADS env vars.
* A silent warm-up synthesis runs during __init__ to pre-allocate
  ONNX internal memory pools, eliminating cold-start latency.
"""

import os
import re
import sys
import time
import asyncio
import urllib.request
from typing import AsyncIterator

# ---------------------------------------------------------------------------
# ONNX Runtime thread pinning  (MUST happen before kokoro_onnx import)
# ---------------------------------------------------------------------------
# Pin to physical core count to avoid hyperthreading overhead.
# os.cpu_count() returns logical processors; we detect physical cores
# via a simple heuristic or fall back to logical / 2.

def _detect_physical_cores() -> int:
    """Best-effort detection of physical (not logical) CPU core count."""
    try:
        # Windows: use wmi via subprocess (no extra deps)
        import subprocess
        out = subprocess.check_output(
            ["wmic", "cpu", "get", "NumberOfCores"],
            text=True, stderr=subprocess.DEVNULL,
        )
        for line in out.strip().splitlines():
            line = line.strip()
            if line.isdigit():
                return int(line)
    except Exception:
        pass
    # Fallback: assume ~75% of logical processors are physical
    logical = os.cpu_count() or 4
    return max(1, logical * 3 // 4)

_PHYSICAL_CORES = _detect_physical_cores()
os.environ.setdefault("OMP_NUM_THREADS", str(_PHYSICAL_CORES))
os.environ.setdefault("ORT_NUM_THREADS", str(_PHYSICAL_CORES))

import numpy as np
import sounddevice as sd
from loguru import logger
from kokoro_onnx import Kokoro

logger.info(
    f"[KokoroTTS] ONNX thread config: "
    f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}  "
    f"ORT_NUM_THREADS={os.environ.get('ORT_NUM_THREADS')}  "
    f"(physical cores={_PHYSICAL_CORES})"
)

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
_MODEL_FILE  = os.path.join(_SERVICE_DIR, "kokoro-v1.0.int8.onnx")   # INT8 (~88 MB)
_MODEL_FP32  = os.path.join(_SERVICE_DIR, "kokoro-v1.0.onnx")        # FP32 fallback (~310 MB)
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
    sizes = {
        "kokoro-v1.0.int8.onnx": "~88 MB",
        "kokoro-v1.0.onnx": "~310 MB",
        "voices-v1.0.bin": "~40 MB",
    }
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
        model_file:   Path to the ONNX model (default: INT8 variant, auto-downloaded).
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

        # Determine which model filename to download
        model_basename = os.path.basename(model_file)

        logger.info(f"[KokoroTTS] Loading model ({model_basename}) ...")
        t0 = time.perf_counter()

        _download_if_missing(model_file,  model_basename)
        _download_if_missing(voices_file, "voices-v1.0.bin")

        self._kokoro = Kokoro(model_file, voices_file)

        elapsed = time.perf_counter() - t0
        logger.info(f"[KokoroTTS] Model ready in {elapsed:.2f}s  (voice={voice}, model={model_basename})")

        # Warm-up: run a tiny synthesis to pre-allocate ONNX memory pools
        self._warmup()

    def _warmup(self) -> None:
        """Run a silent synthesis to eliminate cold-start latency."""
        t0 = time.perf_counter()
        _ = self._kokoro.create("Hello.", voice=self.voice, speed=self.speed, lang="en-us")
        logger.info(f"[KokoroTTS] Warm-up done in {time.perf_counter() - t0:.3f}s")

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
        
        audio = audio.astype(np.float32)
        # Pad with 0.25s of silence to prevent sounddevice from cutting off the end
        silence = np.zeros(int(SAMPLE_RATE * 0.25), dtype=np.float32)
        return np.concatenate([audio, silence])

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

    # ------------------------------------------------------------------
    # WAV encoding for browser streaming
    # ------------------------------------------------------------------

    def synthesize_wav_bytes(self, text: str) -> bytes:
        """
        Synthesize *text* and return raw WAV bytes (24 kHz, 16-bit mono).

        This is used to stream audio back to the browser over WebSocket
        instead of playing on the server's speakers.
        """
        import io
        import wave

        audio = self.synthesize(text)
        if len(audio) == 0:
            return b""

        # Convert float32 [-1, 1] → int16
        pcm = (audio * 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())

        return buf.getvalue()

    async def stream_to_client(
        self,
        sentence_stream: AsyncIterator[str],
    ) -> AsyncIterator[tuple[str, bytes]]:
        """
        Consume an async sentence generator and yield (sentence, wav_bytes)
        pairs for each sentence.

        The server can then send these over WebSocket to the browser.

        Example
        -------
            async for sentence, wav_bytes in tts.stream_to_client(
                llm_store.invoke_stream(messages)
            ):
                # send sentence text + base64(wav_bytes) to browser
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
                logger.info(f"[KokoroTTS] TTFT (client stream) = {ttft:.3f}s")
                first = False

            logger.debug(
                f"[KokoroTTS] WAV chunk: '{sentence[:60]}...' | "
                f"synth={synth_s:.3f}s  size={len(wav_bytes)} bytes"
            )

            yield sentence, wav_bytes

        logger.info(
            f"[KokoroTTS] Client stream done. Total = "
            f"{time.perf_counter() - total_start:.2f}s"
        )
