# Pipecat AI Voice Bot

A modular voice assistant with streaming LLM output, real-time Kokoro TTS, and speech-to-text via Deepgram — all over WebSocket.

---

## Project Structure

```
pipecat_ai/
├── config/
│   ├── __init__.py
│   └── settings.py            # Centralized env-based configuration
├── models/
│   ├── __init__.py
│   └── messages.py            # WebSocket message types (Pydantic)
├── services/
│   ├── __init__.py
│   ├── llm/
│   │   ├── __init__.py
│   │   └── store.py           # LLM provider (Cohere, OpenAI, Gemini, Anthropic)
│   ├── stt/
│   │   ├── __init__.py
│   │   └── deepgram_stt.py    # Deepgram speech-to-text
│   └── tts/
│       ├── __init__.py
│       └── kokoro_tts.py      # Kokoro-ONNX text-to-speech (local, CPU-friendly)
├── examples/
│   ├── bot.py                 # Pipecat SDK WebSocket client (reference)
│   └── daily_bot.py           # Pipecat SDK Daily.co WebRTC bot (reference)
├── logs/                      # Auto-created at runtime
├── server.py                  # Main WebSocket server
├── client.html                # Browser UI (text + voice chat)
├── tests.py                   # Integration tests
├── run_bot.ps1                # PowerShell helper to run with venv
├── requirements.txt
├── .env                       # Your API keys (not committed)
├── .env.example               # Template for .env
└── .gitignore
```

---

## Architecture

Every user turn follows this pipeline:

```
 User
  │ Voice / Text
  ▼
┌──────────────────────────────────────────────────────────┐
│  WebSocket Server  (server.py)                           │
│                                                          │
│  Audio ──► Deepgram STT ──► Transcript                  │
│  Text  ──────────────────► User message                 │
│                                                          │
│  Transcript / User message                               │
│        ──► LLMStore.invoke_stream()   ← streams 1 sentence at a time
│               │                                          │
│               │  sentence 1                              │
│               ▼                                          │
│          KokoroTTS.speak_stream()     ← synthesizes + plays immediately
│               │  sentence 2 (overlap)                    │
│               ▼                                          │
│            Speaker                                        │
│                                                          │
│  Full text response ──► WebSocket    ← client still gets text
└──────────────────────────────────────────────────────────┘
```

Key latency property: the user **hears sentence 1** while the LLM is still
generating sentence 2 onwards — eliminating the "wait for full response" delay.

The `examples/` folder contains reference implementations using the **Pipecat SDK framework** (a different architecture using pipelines + transport layers). These are not used by the main server but are useful for learning.


---

## Setup

### Prerequisites

- **Python 3.10+** (tested on 3.13 and 3.14)

Create the virtual environment (one-time):

```powershell
py -3.13 -m venv .venv   # or py -3.14, py -3.11, etc.
```

### 1. Configure API Keys

```powershell
Copy-Item .env.example .env
# Edit .env and add your actual API keys
```

**Required keys:**

| Key | Service | Purpose |
|-----|---------|---------|
| `DEEPGRAM_API_KEY` | [Deepgram](https://deepgram.com) | Speech-to-text (required) |

**Exactly one LLM key** — only the provider matching `DEFAULT_LLM` is loaded at startup:

| Key | Service | Set `DEFAULT_LLM` to |
|-----|---------|---------------------|
| `COHERE_API_KEY` | [Cohere](https://cohere.com) | `cohere` |
| `OPENAI_API_KEY` | [OpenAI](https://platform.openai.com) | `openai` |
| `GEMINI_API_KEY` | [Google AI](https://ai.google.dev) | `gemini` |
| `ANTHROPIC_API_KEY` | [Anthropic](https://anthropic.com) | `anthropic` |

### 2. Install Dependencies

```powershell
& ".venv\Scripts\pip.exe" install -r requirements.txt
```

---

## Running

### Start the Server

```powershell
# Option A: Direct
& ".venv\Scripts\python.exe" server.py

# Option B: Helper script
.\run_bot.ps1 server.py
```

**Expected output:**

```
[KokoroTTS] Loading model ...
[OK] Model ready in 1.0s
Pipecat Voice Bot Server initialized
Default LLM: cohere
LLM provider initialized: cohere
Server running on ws://0.0.0.0:8765
```

> **Note:** On the very first start, Kokoro will download ~350 MB of model files
> (`kokoro-v1.0.onnx` + `voices-v1.0.bin`) into `services/tts/`. Subsequent
> starts load from disk in ~1 second.

### Connect via Browser

Open `client.html` in your browser. It connects to `ws://127.0.0.1:8765` and supports:
- **Text chat**: Type a message and press Enter/Send
- **Voice chat**: Click record, speak, click stop — audio is transcribed and sent to the LLM

### Run Tests

```powershell
& ".venv\Scripts\python.exe" tests.py
```

---

## WebSocket API

### Sending Messages

**Text message:**
```json
{
  "type": "text",
  "data": "Hello, how are you?"
}
```

**Audio message (base64-encoded):**
```json
{
  "type": "audio",
  "data": "UklGRi4AAABXQVZFZm10IBAAAA..."
}
```

**Plain text** (non-JSON strings are also accepted as text input).

### Receiving Messages

**Bot response:**
```json
{
  "type": "response",
  "data": "I'm doing well! How can I help you?"
}
```

**Error:**
```json
{
  "type": "error",
  "data": "Error description"
}
```

---

## Configuration

All settings are in `.env` (see `.env.example` for the full list).

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_HOST` | `0.0.0.0` | Server bind address |
| `SERVER_PORT` | `8765` | Server port |
| `DEFAULT_LLM` | `cohere` | **Only this provider loads at startup.** Options: `cohere` `openai` `gemini` `anthropic` |
| `LLM_TEMPERATURE` | `0.7` | Response creativity (0.0–1.0) |
| `LLM_MAX_TOKENS` | `500` | Max response length |
| `COHERE_MODEL` | `command-a-03-2025` | Cohere model |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model |
| `ANTHROPIC_MODEL` | `claude-3-5-sonnet-20241022` | Anthropic model |
| `DEEPGRAM_MODEL` | `nova-2` | Deepgram STT model |
| `DEEPGRAM_LANGUAGE` | `en` | STT language |
| `SYSTEM_INSTRUCTION` | *(helpful assistant prompt)* | Bot system prompt |
| `LOG_LEVEL` | `INFO` | Logging level |

**TTS is configured in `server.py`** (no env var needed):
```python
tts_service = KokoroTTS(
    voice="af_heart",   # see AVAILABLE_VOICES in services/tts/kokoro_tts.py
    speed=1.0,
)
```

---

## Extending

### Adding a New LLM Provider

1. Add a provider class in `services/llm/store.py` inheriting from `LLMProvider`
2. Implement `invoke()`, `invoke_sync()`, and `invoke_stream()` methods
3. Register it in `LLMStore._PROVIDER_FACTORIES` and `_API_KEY_ATTRS`
4. Add the API key env var to `config/settings.py`
5. Set `DEFAULT_LLM=<your_provider>` in `.env`

```python
class MyProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, temperature: float):
        # Initialize your client
        pass

    async def invoke(self, messages: List[BaseMessage]) -> str:
        return await asyncio.to_thread(self.invoke_sync, messages)

    def invoke_sync(self, messages: List[BaseMessage]) -> str:
        response = self.client.invoke(messages)
        return response.content

    async def invoke_stream(self, messages: List[BaseMessage]) -> AsyncIterator[str]:
        # Use the shared _stream_sentences helper with your client's .stream()
        async def _tokens():
            for t in self.client.stream(messages):
                yield t
        async for sentence in self._stream_sentences(_tokens(), lambda t: t.content):
            yield sentence
```

### Changing the TTS Voice

Edit the `KokoroTTS` constructor in `server.py`:

```python
tts_service = KokoroTTS(voice="am_adam", speed=1.0)  # deep US male voice
```

Available voices (run `from services.tts import AVAILABLE_VOICES; print(AVAILABLE_VOICES)`):
- `af_heart` — US female, warm (default)
- `af_bella` — US female, professional
- `am_adam` — US male, deep
- `am_michael` — US male, neutral
- `bf_emma` — British female
- `bm_george` — British male

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `DEEPGRAM_API_KEY not set` | Add it to `.env` |
| `DEFAULT_LLM not available` | Ensure the matching API key is in `.env` |
| `Cannot import 'genai' from 'google'` | Run `pip install google-genai` |
| Connection refused on port 8765 | Start `server.py` first |
| Audio transcription fails | Check Deepgram API key; check `logs/pipecat.log` |
| `.venv not found` | Create it: `py -3.13 -m venv .venv` |
| Kokoro model files missing | They auto-download on first run (~350 MB); check your internet connection |
| TTS sounds choppy | Increase sentence chunk size by tuning the `_BOUNDARY` regex in `store.py` |
| High TTS latency (TTFT > 1s) | Short first sentences cause cold-start; try `speed=1.2` to reduce audio length |

---

## Resources

- [Pipecat Documentation](https://docs.pipecat.ai)
- [Deepgram Docs](https://developers.deepgram.com)
- [LangChain Docs](https://python.langchain.com)
