# S.E.T.H. — AI Voice & Agent Assistant

A modular AI assistant featuring a real-time voice pipeline, a LangGraph ReAct agent with tool-calling, persistent semantic memory (LanceDB), conversation checkpoints (SQLite), and a Svelte 5 desktop UI built with Tauri.

Built across 7 development phases with a concrete system design — from a raw WebSocket voice bot to a fully optimized agentic assistant with long-term memory.

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 0** | ✅ Done | Core voice bot — WebSocket server, Deepgram STT, Kokoro/Cartesia TTS, multi-LLM streaming (Cohere, OpenAI, Gemini, Anthropic, Ollama, Groq) |
| **Phase 1** | ✅ Done | LangGraph ReAct agent — web search (Tavily), URL fetch, code generation tool, Groq/Cohere/Ollama providers, CLI REPL |
| **Phase 2** | ✅ Done | Persistent semantic memory (LanceDB + sentence-transformers) + conversation checkpoints (SQLite) with save/restore/fork |
| **Phase 3** | ✅ Done | Full integration — memory + checkpoints merged into the WebSocket voice server, replacing the simple LLM pipeline with the full agent graph |
| **Phase 4** | ✅ Done | Svelte 5 / Tauri 2 desktop frontend — ChatPanel, ArtifactPanel, AudioPlayer, AudioRecorder, WakeWordDetector, Markdown rendering (Shiki, KaTeX, Mermaid) |
| **Phase 5** | ✅ Done | Smart TTS routing (prose/code/table/list classification), `bot_start → bot_chunk → bot_end` streaming protocol, `<artifact>` code panel system, raw tool-tag sanitization |
| **Phase 6** | ✅ Done | Langfuse observability (full agent tracing), per-client session isolation, Groq `tool_use_failed` graceful fallback, auto-reconnect WebSocket |
| **Phase 7** | ✅ Done | Optimization & final touches — ONNX INT8 quantization + CPU thread pinning, TTS warm-up, AudioContext pre-unlock, queue-based playback with safety timeouts, connection state machine hardening |

---

## System Architecture

```
                          ┌──────────────────────────────────────────────────────┐
                          │              Svelte 5 / Tauri Frontend              │
                          │                                                      │
                          │  ChatPanel ── ChatInput ── StatusBar ── ArtifactPanel │
                          │  AudioPlayer (Web Audio API queue-based playback)    │
                          │  AudioRecorder (MediaRecorder → base64 WAV)          │
                          │  WakeWordDetector ("Hey" keyword activation)         │
                          └────────────────┬───────────────────────────────────────┘
                                           │ WebSocket (JSON messages)
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                           Python WebSocket Server (server.py)                       │
│                                                                                      │
│  ┌──────────┐    ┌──────────────────────────────────────────────────────────────┐    │
│  │ Deepgram │    │               LangGraph ReAct Agent (4-node graph)          │    │
│  │   STT    │    │                                                              │    │
│  │ (nova-2) │    │  memory_retrieve ──► agent ◄──► tools ──► post_process      │    │
│  └────┬─────┘    │       │                │           │            │            │    │
│       │          │  LanceDB search   Groq/Cohere   Tavily      Extract facts   │    │
│       │          │  top-k memories    LLM call    web_search   Save checkpoint  │    │
│       │          │                                fetch_url                     │    │
│       │          │                                generate_code                 │    │
│       │          └──────────────────────────────────────────────────────────────┘    │
│       │                         │                                                    │
│       │                         ▼ response text                                      │
│       │          ┌──────────────────────────────────┐                                │
│       │          │ Smart TTS Router                  │                                │
│       │          │  Prose → Cartesia/Kokoro TTS      │                                │
│       │          │  Code blocks → "Here's the code"  │                                │
│       │          │  Tables → "Here's a table"         │                                │
│       │          │  Lists → "Here are N items"        │                                │
│       │          └────────────────┬───────────────────┘                                │
│       │                          │ WAV audio (base64)                                  │
│       │                          ▼                                                     │
│  Streaming protocol: bot_start → bot_chunk (text) + audio_response (WAV) → bot_end   │
│                                                                                      │
│  Langfuse observability (optional) — traces every agent invocation                   │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
pipecat_ai/
├── config/
│   └── settings.py                 # All env-based config (LLM, TTS, STT, memory, checkpoints)
├── data/                           # Auto-created at runtime
│   ├── memory.db/                  # LanceDB — semantic memory vectors
│   └── checkpoints.sqlite          # SQLite — conversation snapshots
├── models/
│   └── messages.py                 # WebSocket message types (Pydantic)
├── services/
│   ├── agent/                      # LangGraph ReAct agent
│   │   ├── graph.py                # 4-node graph: memory_retrieve → agent ⇌ tools → post_process
│   │   ├── llm_factory.py          # Agent LLM factory (Cohere, Ollama, Groq)
│   │   └── tools.py                # web_search, fetch_url, generate_code, get_current_datetime
│   ├── memory/                     # Persistent semantic memory
│   │   ├── schema.py               # PyArrow schema for LanceDB memory table
│   │   ├── store.py                # MemoryStore — embed, search, add, clear (LanceDB)
│   │   └── extractor.py            # MemoryExtractor — LLM-powered fact extraction
│   ├── checkpoint/                 # Conversation snapshots
│   │   └── manager.py              # CheckpointManager — save, list, restore, fork (SQLite)
│   ├── llm/
│   │   └── store.py                # Voice bot LLM providers (Cohere, OpenAI, Gemini, Anthropic, Ollama, Groq)
│   ├── stt/
│   │   └── deepgram_stt.py         # Deepgram speech-to-text (nova-2)
│   └── tts/
│       ├── kokoro_tts.py           # Kokoro-ONNX TTS (local, CPU, ~88 MB INT8 model)
│       └── cartesia_tts.py         # Cartesia WebSocket TTS (cloud, Sonic-3)
├── frontend/                       # Svelte 5 + Tauri desktop app
│   ├── src/
│   │   ├── App.svelte              # Root layout — chat + artifact panel
│   │   ├── app.css                 # Global styles (dark theme, glassmorphism)
│   │   └── lib/
│   │       ├── components/
│   │       │   ├── ChatPanel.svelte      # Scrollable message list
│   │       │   ├── ChatMessage.svelte    # Markdown rendering (Shiki, KaTeX, Mermaid)
│   │       │   ├── ChatInput.svelte      # Text input + voice recording button
│   │       │   ├── StatusBar.svelte      # Connection state + wake word toggle
│   │       │   ├── ArtifactPanel.svelte  # Side panel for generated code artifacts
│   │       │   ├── TypingIndicator.svelte
│   │       │   └── Waveform.svelte       # Audio recording visualizer
│   │       ├── services/
│   │       │   ├── audio-player.ts       # Web Audio API queue-based WAV playback
│   │       │   ├── audio-recorder.ts     # MediaRecorder → base64 WAV
│   │       │   └── wake-word.ts          # "Hey" keyword detection (Web Speech API)
│   │       ├── stores/
│   │       │   └── chat.svelte.ts        # Reactive WebSocket state (Svelte 5 runes)
│   │       ├── utils/
│   │       │   ├── shiki.ts              # Syntax highlighting (Shiki)
│   │       │   └── purifier.ts           # DOMPurify + KaTeX math rendering
│   │       └── types.ts                  # Shared TypeScript types
│   ├── src-tauri/                        # Tauri 2 native shell
│   ├── index.html
│   ├── vite.config.ts
│   └── package.json
├── system.prompt                   # Agent personality & directives (loaded at startup)
├── server.py                       # Main WebSocket server (Phase 3 — full agent pipeline)
├── run_agent.py                    # CLI REPL — agent + memory + checkpoints
├── requirements.txt
├── .env                            # API keys (never committed)
└── .env.example                    # Template for .env
```

---

## Phase 0 — Core Voice Bot

Real-time voice pipeline over WebSocket with sentence-level streaming for minimal time-to-first-audio.

### Voice Pipeline

```
Browser                              Server
  │                                     │
  ├─ Record audio (MediaRecorder)       │
  ├─ base64 encode ──────────────────►  │
  │                                     ├─ Deepgram STT → transcript
  │                                     ├─ LangGraph agent → response text
  │                                     ├─ Split into sentences
  │                                     │
  │  ◄── bot_start ────────────────────┤
  │  ◄── bot_chunk (sentence 1 text) ──┤
  │  ◄── audio_response (WAV bytes) ───┤  ← TTS runs per sentence
  │  ◄── bot_chunk (sentence 2 text) ──┤
  │  ◄── audio_response (WAV bytes) ───┤
  │  ◄── bot_end ──────────────────────┤
  │                                     │
  └─ Queue-based playback (Web Audio)   │
```

The user **hears sentence 1** while the LLM is still generating sentence 2 — minimizing TTFT.

### Smart TTS Router

Not all content should be spoken aloud. The server classifies each response chunk:

| Content Type | TTS Action |
|-------------|------------|
| Prose (≤3 sentences) | Spoken in full via TTS |
| Fenced code blocks | `"Here's some Python code."` |
| `<artifact>` tags | `"I've generated the code for you."` |
| Markdown tables | `"Here's a table with the results."` |
| Long bulleted lists | `"Here's a list with N items."` |
| Math expressions | `"Here's a mathematical expression."` |
| Code-like lines (no fences) | Skipped entirely |

### TTS Providers

| Provider | Type | Latency | Size | Setup |
|----------|------|---------|------|-------|
| **Kokoro** (default) | Local ONNX (CPU) | ~0.3-2s/sentence | INT8: 88 MB | Auto-downloads on first run |
| **Cartesia** | Cloud (Sonic-3) | ~0.2-0.5s/sentence | — | Requires `CARTESIA_API_KEY` |

Both providers expose identical interfaces (`synthesize()`, `synthesize_wav_bytes()`, `speak_stream()`).

---

## Phase 1 — LangGraph ReAct Agent

### Agent Graph (4 Nodes)

```
User message
    │
    ▼
memory_retrieve_node
  ├─ Search LanceDB: top-k memories semantically similar to user message
  └─ Inject memory context into system prompt
    │
    ▼
agent_node  ◄──────────┐
  ├─ System prompt: base + memory context      │
  ├─ Invoke LLM with tools bound               │
  └─ Decides: tool call OR final answer        │
              │                tools_node ─────┘
              │                (web_search / fetch_url / generate_code)
              ▼
post_process_node
  ├─ Extract new durable facts (LLM-powered)
  ├─ Store facts → LanceDB vector embeddings
  └─ Save conversation snapshot → SQLite
    │
    ▼
   END
```

### Agent Tools

| Tool | Trigger | Description |
|------|---------|-------------|
| `get_current_datetime` | Time/date questions | Returns current date, time, and timezone (via `zoneinfo`) |
| `web_search` | Current events, scores, news | Tavily search with Google scrape fallback |
| `fetch_url` | User provides a specific URL | httpx + trafilatura text extraction (4K char limit) |
| `generate_code` | Code requests (>5 lines) | Wraps code in `<artifact>` tags for UI rendering |

### Supported LLM Providers (Agent)

| Provider | Model (default) | Tool Calling | Notes |
|----------|----------------|--------------|-------|
| **Groq** | `llama-3.3-70b-versatile` | ✅ | Fast inference, free tier, recommended default |
| **Cohere** | `command-a-03-2025` | ✅ | Also used for memory fact extraction |
| **Ollama** | `qwen3:8b` | ✅ | Local, no API key needed |

### Supported LLM Providers (Voice Bot)

Groq, Cohere, OpenAI (`gpt-4o-mini`), Gemini, Anthropic (`claude-3-5-sonnet`), Ollama.

---

## Phase 2 — Persistent Memory & Checkpoints

### Semantic Memory (LanceDB)

- **Storage**: `data/memory.db/` — LanceDB vector database (zero-server, file-based)
- **Embedding model**: `all-MiniLM-L6-v2` (80 MB, sentence-transformers, downloaded once from HuggingFace)
- **Retrieval**: Top-k semantic search per turn — most relevant facts injected into system prompt
- **Extraction**: After every response, an LLM call extracts new durable facts (name, preferences, project details) and stores each as a separate vector-embedded memory record
- **Persistence**: Cross-session — memories survive restarts and are scoped to `user_id`

### Conversation Checkpoints (SQLite)

- **Storage**: `data/checkpoints.sqlite` (single file, zero external deps)
- **Trigger**: Saved automatically after every agent response
- **Schema**: `id, user_id, session_id, thread_id, label, messages_json, created_at`
- **Operations**: save, list, restore, fork into new session branch

### ID Strategy

| ID | Source | Scope |
|----|--------|-------|
| `user_id` | `--user` CLI flag or `DEFAULT_USER_ID` env var | Cross-session persistent identity |
| `session_id` | Auto-generated UUID at startup (or `--session` flag) | Groups one conversation run |
| `checkpoint_id` | Auto-generated UUID at save time | Single snapshot within a session |

---

## Phase 3 — Server Integration

Merged the Phase 2 agent (memory + checkpoints + tools) into the WebSocket voice server, replacing the simple `LLMStore.invoke_stream()` pipeline with the full LangGraph `agent_graph.ainvoke()` call.

### What Changed from Phase 0

| Aspect | Phase 0 | Phase 3 |
|--------|---------|---------|
| LLM pipeline | Direct `LLMStore.invoke_stream()` | Full LangGraph `agent_graph.ainvoke()` |
| Memory | None | LanceDB semantic retrieval per turn |
| Checkpoints | None | SQLite auto-save after every response |
| Tools | None | web_search, fetch_url, generate_code, datetime |
| Response flow | Stream sentences directly | Agent invoke → split → stream |

---

## Phase 4 — Svelte 5 / Tauri Desktop Frontend

Built a production desktop application using Svelte 5 (runes) + Tauri 2, replacing the original plain HTML widget.

### Frontend Features

- **Rich Markdown rendering** — Shiki syntax highlighting, KaTeX math, Mermaid diagrams, DOMPurify sanitization
- **Code artifact panel** — Side panel for generated code with language detection
- **Voice input** — MediaRecorder → base64 WAV → Deepgram STT
- **Wake word detection** — `"Hey"` keyword via Web Speech API for hands-free activation
- **Reactive state management** — Svelte 5 runes (`$state`, `$derived`) for fine-grained reactivity
- **Tauri 2 native shell** — Desktop app with native window controls

### Component Architecture

| Component | Purpose |
|-----------|---------|
| `ChatPanel` | Scrollable message list with auto-scroll |
| `ChatMessage` | Per-message Markdown rendering (Shiki + KaTeX + Mermaid) |
| `ChatInput` | Text input + voice recording toggle |
| `StatusBar` | Connection state indicator + wake word toggle |
| `ArtifactPanel` | Side panel for `<artifact>` code blocks |
| `TypingIndicator` | Animated dots during agent processing |
| `Waveform` | Audio recording visualizer |

---

## Phase 5 — Smart TTS Routing & Streaming Protocol

Introduced content-aware TTS routing so only prose is spoken aloud, and a chunked streaming protocol for single-bubble response rendering.

### Streaming Protocol

```
Server → Client message flow per response:

  bot_start           → Create empty bot bubble
  bot_chunk (text)    → Append sentence to bubble
  audio_response (WAV)→ Enqueue audio for playback
  bot_chunk (text)    → Append next sentence
  audio_response (WAV)→ Enqueue next audio
  ...
  bot_end             → Finalize bubble
  response (full text)→ Completion signal
```

### TTS Content Classification

| Content Type | TTS Action |
|-------------|------------|
| Prose (≤3 sentences) | Spoken in full via TTS |
| Fenced code blocks | `"Here's some Python code."` |
| `<artifact>` tags | `"I've generated the code for you."` |
| Markdown tables | `"Here's a table with the results."` |
| Long bulleted lists | `"Here's a list with N items."` |
| Math expressions | `"Here's a mathematical expression."` |
| Code-like lines (no fences) | Skipped entirely |

### Artifact System

- `generate_code` tool outputs `<artifact language="..." title="...">` tags
- Server detects raw `<function=generate_code>` leakage from Groq/Llama and converts to proper `<artifact>` tags
- Strips `<|python_tag|>`, `<|eot_id|>`, and other LLM internal tokens
- Frontend parses artifacts into the side panel with syntax highlighting

---

## Phase 6 — Observability & Production Hardening

### Langfuse Tracing

When `LANGFUSE_ENABLED=true`, every agent invocation is traced through Langfuse with:
- User ID and session ID metadata
- Full LLM call traces (input/output tokens, latency)
- Tool call execution details

### Per-Client Session Isolation

Each WebSocket connection gets a unique `ClientSession` with its own `session_id`, message history, and retrieved memories. The shared agent graph is stateless — all session state lives in `ClientSession`.

### Error Handling

- **Groq `tool_use_failed`** — Automatically retries without tools bound so the agent can still answer
- **TTS timeout** — 30s safety net per sentence; skips to next chunk on timeout
- **Client disconnect** — Graceful mid-stream detection; stops processing immediately
- **Empty responses** — Fallback messages when the model outputs only tool calls with no text

---

## Phase 7 — Optimization & Final Touches

### Backend Optimizations

- **ONNX INT8 quantization** — Kokoro model reduced from ~310 MB (FP32) to ~88 MB (INT8), ~3.5× smaller with negligible quality loss
- **CPU thread pinning** — `OMP_NUM_THREADS` and `ORT_NUM_THREADS` auto-set to physical core count (not logical/hyperthreaded), eliminating contention
- **TTS warm-up** — Silent synthesis at startup pre-allocates ONNX memory pools, eliminating cold-start latency on first request
- **Memory embedding pre-warm** — `sentence-transformers` model loaded at startup, not on first query
- **HuggingFace warning suppression** — Silences noisy symlink/auth warnings without hiding real errors

### Frontend Optimizations

- **AudioContext pre-unlock** — Warm-up on first user gesture (click/keypress) to bypass browser autoplay policy before audio arrives
- **Queue-based playback** — Sequential WAV decoding via Web Audio API with `onended` chaining
- **Safety timeouts** — If `onended` doesn't fire within `buffer.duration + 5s`, force-advances to next chunk (prevents permanent stuck state)
- **Force-stop** — Hard deadline (15s) on `speaking` state after final `response` message; clears queue and resets
- **Auto-reconnect** — WebSocket reconnects after 3s on disconnect with state cleanup
- **Connection state machine** — `disconnected → connecting → idle → listening → processing → speaking` with guarded transitions


---

## Running

### Agent CLI (REPL)

```powershell
.venv\Scripts\activate
python run_agent.py

# With custom user / restore a session
python run_agent.py --user my_uuid --session abc123
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `<message>` | Chat with the agent |
| `/memory` | Show memory chunks retrieved for the last turn |
| `/checkpoints` | List checkpoints in the current session |
| `/checkpoints all` | List checkpoints across all sessions |
| `/restore <id>` | Restore a past checkpoint (8-char prefix works) |
| `/fork <id>` | Branch a checkpoint into a new session |
| `/clear-memory` | Delete all stored memories for current user |
| `/whoami` | Show current user ID and session ID |
| `/help` | Show help |
| `clear` | Reset conversation (memories are preserved) |
| `quit` / `exit` | Exit |

### Voice Server + Frontend

```powershell
# Terminal 1: Start Python WebSocket server
.venv\Scripts\activate
python server.py

# Terminal 2: Start Svelte frontend
cd frontend
npm install
npm run dev

# Or run as Tauri desktop app
npm run tauri:dev
```

The frontend connects to `ws://127.0.0.1:8765` in dev mode (proxied through Vite) or reads `VITE_WS_URL` in production.

---

## Setup

### Prerequisites

- Python 3.10+ (tested on 3.13)
- Node.js 18+ (for frontend)
- Rust toolchain (only for Tauri desktop builds)

```powershell
# Backend
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Frontend
cd frontend
npm install
```

### Configure API Keys

```powershell
Copy-Item .env.example .env
# Edit .env and fill in your keys
```

**Required:**

| Variable | Service | Purpose |
|----------|---------|------------|
| `DEEPGRAM_API_KEY` | [Deepgram](https://deepgram.com) | Voice → text (STT) |
| `GROQ_API_KEY` | [Groq](https://console.groq.com) | Agent LLM (fast, free tier) |
| `COHERE_API_KEY` | [Cohere](https://cohere.com) | Memory fact extraction LLM |
| `TAVILY_API_KEY` | [Tavily](https://tavily.com) | Web search tool |

**Optional:**

| Variable | Purpose |
|----------|---------|
| `CARTESIA_API_KEY` | Cloud TTS (set `DEFAULT_TTS=cartesia`) |
| `OPENAI_API_KEY` | OpenAI LLM provider |
| `GEMINI_API_KEY` | Google Gemini LLM provider |
| `ANTHROPIC_API_KEY` | Anthropic LLM provider |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Langfuse observability |
| `HF_TOKEN` | HuggingFace (for private models, usually not needed) |

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_HOST` | `0.0.0.0` | WebSocket server bind address |
| `SERVER_PORT` | `8765` | WebSocket server port |
| `DEFAULT_LLM` | `groq` | Voice bot LLM provider |
| `AGENT_LLM` | `groq` | Agent graph LLM provider |
| `MEMORY_LLM` | `cohere` | Fact extraction LLM |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model |
| `COHERE_MODEL` | `command-a-03-2025` | Cohere model |
| `MEMORY_ENABLED` | `true` | Enable/disable semantic memory |
| `MEMORY_DB_PATH` | `./data/memory.db` | LanceDB directory |
| `MEMORY_TOP_K` | `5` | Memories retrieved per turn |
| `MEMORY_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `CHECKPOINT_ENABLED` | `true` | Enable/disable checkpoints |
| `CHECKPOINT_DB_PATH` | `./data/checkpoints.sqlite` | SQLite file path |
| `DEFAULT_USER_ID` | `user_arman_admin` | Default user (override with `--user`) |
| `DEFAULT_TTS` | `kokoro` | TTS provider (`kokoro` or `cartesia`) |
| `LLM_TEMPERATURE` | `0.7` | LLM response creativity |
| `LLM_MAX_TOKENS` | `500` | Max tokens per LLM response |
| `LANGFUSE_ENABLED` | `false` | Enable Langfuse tracing |
| `LOG_LEVEL` | `INFO` | Logging level |

### System Prompt

Edit `system.prompt` in the project root — loaded automatically at startup. Supports full Markdown. Falls back to `SYSTEM_INSTRUCTION` in `.env` if the file is missing.

---

## Extending

### Adding a New LLM Provider (Voice Bot)

1. Add a class inheriting `LLMProvider` in `services/llm/store.py`
2. Implement `invoke()`, `invoke_sync()`, `invoke_stream()`
3. Register in `LLMStore._PROVIDER_FACTORIES` and `_API_KEY_ATTRS`
4. Add the key to `config/settings.py`

### Adding a New LLM Provider (Agent)

1. Add a `_create_<provider>` function in `services/agent/llm_factory.py`
2. Register in the `create_llm` dispatch table
3. Add the key to `config/settings.py`

### Adding Agent Tools

Add a `@tool`-decorated function to `services/agent/tools.py` and include it in `get_tools()`.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `DEEPGRAM_API_KEY not set` | Add it to `.env` |
| `GROQ_API_KEY not configured` | Add it to `.env` and set `AGENT_LLM=groq` |
| `no checkpoints found` | Ensure `data/` directory exists; check logs |
| Memory shows 0 on startup | `data/memory.db/` may not exist yet — send a message first |
| HuggingFace download on first run | `all-MiniLM-L6-v2` (~80 MB) downloads once then caches |
| Groq `tool_use_failed` error | Rephrase — Groq's tool-calling retries without tools automatically |
| Audio transcription fails | Check Deepgram key; see `logs/pipecat.log` |
| Kokoro model missing | Auto-downloads INT8 (~88 MB) + voices (~40 MB) on first run |
| Frontend won't connect | Ensure Python server is running on port 8765 |
| Audio doesn't play in browser | Click/type first to unlock AudioContext (browser autoplay policy) |
| Tauri build fails | Ensure Rust toolchain is installed (`rustup`) |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Agent Framework** | LangGraph, LangChain |
| **LLM Providers** | Groq, Cohere, OpenAI, Gemini, Anthropic, Ollama |
| **Speech-to-Text** | Deepgram (nova-2) |
| **Text-to-Speech** | Kokoro-ONNX (local) / Cartesia Sonic-3 (cloud) |
| **Semantic Memory** | LanceDB + sentence-transformers (all-MiniLM-L6-v2) |
| **Checkpoints** | SQLite |
| **Web Search** | Tavily (with Google scrape fallback) |
| **URL Extraction** | httpx + trafilatura |
| **Frontend** | Svelte 5 (runes), TypeScript, Vite |
| **Desktop Shell** | Tauri 2 (Rust) |
| **Markdown** | Shiki (syntax), KaTeX (math), Mermaid (diagrams), DOMPurify |
| **Audio** | Web Audio API, MediaRecorder, Web Speech API |
| **Observability** | Langfuse |
| **Server** | Python asyncio + websockets |

---

## Resources

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LanceDB Docs](https://lancedb.github.io/lancedb/)
- [Groq Console](https://console.groq.com)
- [Deepgram Docs](https://developers.deepgram.com)
- [Tavily API](https://docs.tavily.com)
- [Svelte 5 Docs](https://svelte.dev/docs)
- [Tauri 2 Docs](https://tauri.app)
- [Langfuse Docs](https://langfuse.com/docs)
