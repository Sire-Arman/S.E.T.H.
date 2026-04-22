# Pipecat AI Voice Bot

A modular voice assistant with multi-LLM support, speech-to-text via Deepgram, and a browser-based chat UI — all over WebSocket.

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
│   │   └── store.py           # Multi-LLM provider (Cohere, OpenAI, Gemini, Anthropic)
│   └── stt/
│       ├── __init__.py
│       └── deepgram_stt.py    # Deepgram speech-to-text
├── examples/
│   ├── bot.py                 # Pipecat SDK WebSocket client (reference)
│   └── daily_bot.py           # Pipecat SDK Daily.co WebRTC bot (reference)
├── logs/                      # Auto-created at runtime
├── server.py                  # Main WebSocket server
├── client.html                # Browser UI (text + voice chat)
├── tests.py                   # Integration tests
├── run_bot.ps1                # PowerShell helper to run with venv313
├── requirements.txt
├── .env                       # Your API keys (not committed)
├── .env.example               # Template for .env
└── .gitignore
```

---

## Architecture

```
┌──────────────────────────────────────────┐
│        WebSocket Server (server.py)      │
│                                          │
│   ┌──────────────────────────────────┐   │
│   │  Text Input ──► LLM ──► Response │   │
│   │                                  │   │
│   │  Audio Input ──► Deepgram STT    │   │
│   │       ──► LLM ──► Response       │   │
│   └──────────────────────────────────┘   │
│                                          │
│   Supported LLMs:                        │
│   • Cohere (default)                     │
│   • OpenAI                               │
│   • Google Gemini                        │
│   • Anthropic Claude                     │
│                                          │
│   Port: 8765 (configurable)              │
└──────────────────────────────────────────┘
              ⬆ WebSocket (JSON)
┌──────────────────────────────────────────┐
│   Clients                                │
│   • client.html (browser UI)             │
│   • Any WebSocket client                 │
│   • tests.py (automated tests)           │
└──────────────────────────────────────────┘
```

The `examples/` folder contains reference implementations using the **Pipecat SDK framework** (a different architecture using pipelines + transport layers). These are not used by the main server but are useful for learning.

---

## Setup

### Prerequisites

- **Python 3.10–3.13** (3.14 not yet supported due to numba dependency)

Create the virtual environment (one-time):

```powershell
py -3.13 -m venv .venv
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

**At least one LLM key:**

| Key | Service |
|-----|---------|
| `COHERE_API_KEY` | [Cohere](https://cohere.com) (default) |
| `OPENAI_API_KEY` | [OpenAI](https://platform.openai.com) |
| `GEMINI_API_KEY` | [Google AI](https://ai.google.dev) |
| `ANTHROPIC_API_KEY` | [Anthropic](https://anthropic.com) |

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
Pipecat Voice Bot Server initialized
Default LLM: cohere
Server running on ws://0.0.0.0:8765
```

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
| `DEFAULT_LLM` | `cohere` | LLM provider (`cohere`, `openai`, `gemini`, `anthropic`) |
| `LLM_TEMPERATURE` | `0.7` | Response creativity (0.0–1.0) |
| `LLM_MAX_TOKENS` | `500` | Max response length |
| `COHERE_MODEL` | `command-a-03-2025` | Cohere model |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model |
| `ANTHROPIC_MODEL` | `claude-3-5-sonnet-20241022` | Anthropic model |
| `DEEPGRAM_MODEL` | `nova-2` | Deepgram STT model |
| `DEEPGRAM_LANGUAGE` | `en` | STT language |
| `SYSTEM_INSTRUCTION` | *(helpful assistant prompt)* | Bot system prompt |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## Extending

### Adding a New LLM Provider

1. Add a provider class in `services/llm/store.py` inheriting from `LLMProvider`
2. Implement `invoke()` and `invoke_sync()` methods
3. Register it in `LLMStore._initialize_providers()`
4. Add the API key env var to `config/settings.py`

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
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `DEEPGRAM_API_KEY not set` | Add it to `.env` |
| `Default LLM not available` | Ensure the matching API key is in `.env` |
| `Cannot import 'genai' from 'google'` | Run `pip install google-genai` |
| Connection refused on port 8765 | Start `server.py` first |
| Audio transcription fails | Check Deepgram API key is valid; check `logs/pipecat.log` |
| `.venv not found` | Create it with `py -3.13 -m venv .venv` |

---

## Resources

- [Pipecat Documentation](https://docs.pipecat.ai)
- [Deepgram Docs](https://developers.deepgram.com)
- [LangChain Docs](https://python.langchain.com)
