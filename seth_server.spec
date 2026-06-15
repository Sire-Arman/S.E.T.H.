# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for SETH Voice Assistant server.

Bundles server.py + all dependencies + TTS models + data files
into a portable directory that Tauri can spawn as a sidecar.

Build:
    .\.venv\Scripts\pyinstaller.exe seth_server.spec

Output:
    dist/seth-server/seth-server.exe
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(".")
SERVICES_DIR = os.path.join(PROJECT_ROOT, "services")
TTS_DIR = os.path.join(SERVICES_DIR, "tts")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

from PyInstaller.utils.hooks import collect_data_files

# ---------------------------------------------------------------------------
# Data files to bundle
# ---------------------------------------------------------------------------
datas = [
    # Config & prompts
    (".env", "."),
    ("system.prompt", "."),

    # TTS model files (Kokoro INT8 only — ~88 MB + ~27 MB voices)
    (os.path.join(TTS_DIR, "kokoro-v1.0.int8.onnx"), os.path.join("services", "tts")),
    (os.path.join(TTS_DIR, "voices-v1.0.bin"), os.path.join("services", "tts")),

    # Persistent data (memory DB, checkpoints)
    (DATA_DIR, "data"),
]

# Auto-collect library data files
datas += collect_data_files('kokoro_onnx')
datas += collect_data_files('language_tags')
datas += collect_data_files('phonemizer')

# Filter out missing files
datas = [(src, dst) for src, dst in datas if os.path.exists(src)]

# ---------------------------------------------------------------------------
# Hidden imports — modules that PyInstaller can't auto-detect
# ---------------------------------------------------------------------------
hiddenimports = [
    # Core
    "websockets",
    "websockets.legacy",
    "websockets.legacy.server",
    "dotenv",
    "loguru",
    "pydantic",

    # LangChain / LangGraph
    "langchain",
    "langchain.schema",
    "langchain_core",
    "langchain_core.messages",
    "langchain_core.tools",
    "langchain_core.language_models",
    "langchain_community",
    "langgraph",
    "langgraph.graph",
    "langgraph.graph.state",
    "langgraph.graph.message",
    "langgraph.prebuilt",

    # LLM providers
    "langchain_groq",
    "langchain_openai",
    "langchain_anthropic",
    "langchain_cohere",
    "langchain_ollama",
    "langchain_tavily",

    # Cloud SDKs
    "groq",
    "openai",
    "anthropic",
    "cohere",

    # STT
    "deepgram",

    # TTS
    "kokoro_onnx",
    "onnxruntime",
    "sounddevice",

    # Memory & search
    "lancedb",
    "sentence_transformers",
    "torch",
    "transformers",
    "pandas",
    "pyarrow",
    "lance",

    # Web tools
    "httpx",
    "trafilatura",

    # Langfuse
    "langfuse",
    "langfuse.langchain",

    # Standard lib that PyInstaller sometimes misses
    "zoneinfo",
    "sqlite3",
    "json",
    "asyncio",
    "typing_extensions",

    # Cartesia (imported but not used with kokoro, include anyway)
    "cartesia",
]

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(
    ["server.py"],
    pathex=[PROJECT_ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy unused packages to shrink bundle
        "tkinter",
        "matplotlib",
        "PIL",
        "cv2",
        "scipy.spatial.cKDTree",
        "IPython",
        "jupyter",
        "notebook",
    ],
    noarchive=False,
    optimize=0,
)

# ---------------------------------------------------------------------------
# PYZ (compressed Python modules archive)
# ---------------------------------------------------------------------------
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# ---------------------------------------------------------------------------
# EXE
# ---------------------------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="seth-server",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,       # UPX sometimes breaks ONNX Runtime DLLs
    console=True,    # Keep console for logging; Tauri hides it anyway
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# ---------------------------------------------------------------------------
# COLLECT (one-dir bundle)
# ---------------------------------------------------------------------------
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="seth-server",
)
