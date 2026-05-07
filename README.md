# Pipecat AI — SETH Voice & Agent Assistant

A modular AI assistant with a real-time voice pipeline and a persistent LangGraph agent with semantic memory and conversation checkpoints.

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 0** | ✅ Done | Core voice bot — WebSocket server, Deepgram STT, Kokoro/Cartesia TTS, multi-LLM streaming |
| **Phase 1** | ✅ Done | LangGraph ReAct agent — web search (Tavily), URL fetch, Groq/Cohere/Ollama providers, CLI REPL |
| **Phase 2** | ✅ Done | Persistent memory (LanceDB) + conversation checkpoints (SQLite) in the agent CLI |
| **Phase 3** | 🔲 Planned | Merge Phase 2 memory + checkpoints into WebSocket voice server |

---

## Project Structure

```
pipecat_ai/
├── config/
│   └── settings.py                 # All env-based config (LLM, TTS, memory, checkpoints)
├── data/                           # Auto-created at runtime
│   ├── memory.db/                  # LanceDB — semantic memory vectors (user_arman_admin)
│   └── checkpoints.sqlite          # SQLite — conversation snapshots
├── models/
│   └── messages.py                 # WebSocket message types (Pydantic)
├── services/
│   ├── agent/                      # LangGraph ReAct agent
│   │   ├── graph.py                # 4-node graph: memory_retrieve → agent ⇌ tools → post_process
│   │   ├── llm_factory.py          # Agent LLM factory (Cohere, Ollama, Groq)
│   │   └── tools.py                # web_search (Tavily) + fetch_url (httpx + trafilatura)
│   ├── memory/                     # Phase 2 — Semantic memory
│   │   ├── schema.py               # PyArrow schema for LanceDB memory table
│   │   ├── store.py                # MemoryStore — embed, search, add, clear (LanceDB)
│   │   └── extractor.py            # MemoryExtractor — Cohere LLM fact extraction
│   ├── checkpoint/                 # Phase 2 — Conversation snapshots
│   │   └── manager.py              # CheckpointManager — save, list, restore, fork (SQLite)
│   ├── llm/
│   │   └── store.py                # Voice bot LLM providers (Cohere, OpenAI, Gemini, Anthropic, Ollama, Groq)
│   ├── stt/
│   │   └── deepgram_stt.py         # Deepgram speech-to-text
│   └── tts/
│       ├── kokoro_tts.py           # Kokoro-ONNX TTS (local, CPU)
│       └── cartesia_tts.py         # Cartesia WebSocket TTS (cloud)
├── system.prompt                   # Agent system prompt (edit freely, loaded at startup)
├── server.py                       # Main WebSocket voice server
├── run_agent.py                    # Phase 2 CLI REPL — agent + memory + checkpoints
├── examples/                       # Pipecat SDK reference bots (not used by main server)
├── frontend/                       # Browser widget (widget.html, widget.css, widget.js)
├── logs/                           # Auto-created at runtime
├── requirements.txt
├── .env                            # Your API keys (never committed)
└── .env.example                    # Template for .env
```

---

## Phase 1 — LangGraph ReAct Agent

### Agent Architecture

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
  ├─ Invoke Groq LLM with tools bound          │
  └─ Decides: tool call OR final answer        │
              │                tools_node ─────┘
              │                (web_search / fetch_url)
              ▼
post_process_node
  ├─ Extract new durable facts (Cohere LLM)
  ├─ Store facts → LanceDB
  └─ Save conversation snapshot → SQLite
    │
    ▼
   END
```

### Running the Agent CLI

```powershell
# Activate venv first
.venv\Scripts\activate

# Run with defaults (user: user_arman_admin, new session UUID)
python run_agent.py

# Run with custom user / restore a specific session
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

---

## Phase 2 — Persistent Memory & Checkpoints

### Memory (LanceDB)

- **Storage**: `data/memory.db/` (LanceDB vector database)
- **Embedding model**: `all-MiniLM-L6-v2` (80 MB, CPU-only, downloaded once from HuggingFace)
- **Retrieval**: Top-k semantic search per turn — most relevant facts injected into system prompt
- **Extraction**: After every response, a **Cohere LLM call** extracts new durable facts (name, preferences, etc.) and stores them as separate memory records
- **Persistence**: Cross-session — memories survive restarts and are tied to `user_id`

### Checkpoints (SQLite)

- **Storage**: `data/checkpoints.sqlite` (single file, zero external deps)
- **Trigger**: Saved automatically after every agent response
- **Schema**: `id, user_id, session_id, thread_id, label, messages_json, created_at`
- **Operations**: save, list, restore, fork into new session branch

### ID Strategy

| ID | Source | Scope |
|----|--------|-------|
| `user_id` | `--user` CLI flag or `DEFAULT_USER_ID` in settings | Cross-session persistent identity |
| `session_id` | Auto-generated UUID at startup or `--session` flag | Groups one conversation run |
| `checkpoint_id` | Auto-generated UUID at save time | Single snapshot within a session |

> **Future login support**: `user_id` defaults to `user_arman_admin` but the system is designed to accept any UUID — just pass `--user <uuid>` when a login system is in place.

---

## Phase 0 — Voice Bot (WebSocket Server)

### Voice Pipeline

```
User
  │ Voice / Text (via WebSocket)
  ▼
WebSocket Server (server.py)
  │
  ├─ Audio → Deepgram STT → Transcript
  ├─ Text  ──────────────► User message
  │
  └─ LLMStore.invoke_stream()   ← streams one sentence at a time
         │
         ▼ sentence 1
     TTS.stream_to_client()     ← synthesizes + sends audio immediately
         │
         ▼ sentence 2 (overlap)
     Browser audio player
```

The user **hears sentence 1** while the LLM is still generating sentence 2 — minimizing TTFT.

### Running the Voice Server

```powershell
.venv\Scripts\activate
python server.py
```

**Expected startup output:**
```
Pipecat Voice Bot Server initialized
Default LLM: groq
LLM provider initialized: groq
Server running on ws://0.0.0.0:8765
```

Then open `frontend/widget.html` in your browser.

---

## Setup

### Prerequisites

- Python 3.10+ (tested on 3.13)

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Configure API Keys

```powershell
Copy-Item .env.example .env
# Edit .env and fill in your keys
```

**Required:**

| Variable | Service | Purpose |
|----------|---------|---------|
| `DEEPGRAM_API_KEY` | [Deepgram](https://deepgram.com) | Voice → text |
| `GROQ_API_KEY` | [Groq](https://console.groq.com) | Agent LLM (fast, free tier) |
| `COHERE_API_KEY` | [Cohere](https://cohere.com) | Memory extraction LLM |
| `TAVILY_API_KEY` | [Tavily](https://tavily.com) | Web search tool |

**Optional (pick one for DEFAULT_LLM):**

| Variable | Set `DEFAULT_LLM` to |
|----------|---------------------|
| `GROQ_API_KEY` | `groq` |
| `COHERE_API_KEY` | `cohere` |
| `OPENAI_API_KEY` | `openai` |
| `GEMINI_API_KEY` | `gemini` |
| `ANTHROPIC_API_KEY` | `anthropic` |
| *(none)* | `ollama` (local) |

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_LLM` | `ollama` | Voice bot LLM provider |
| `AGENT_LLM` | `groq` | Agent REPL LLM provider |
| `MEMORY_LLM` | `cohere` | Fact extraction LLM |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model |
| `GROQ_API_KEY` | — | Groq API key |
| `MEMORY_ENABLED` | `true` | Enable/disable semantic memory |
| `MEMORY_DB_PATH` | `./data/memory.db` | LanceDB directory |
| `MEMORY_TOP_K` | `5` | Memories retrieved per turn |
| `CHECKPOINT_ENABLED` | `true` | Enable/disable checkpoints |
| `CHECKPOINT_DB_PATH` | `./data/checkpoints.sqlite` | SQLite file path |
| `DEFAULT_USER_ID` | `user_arman_admin` | Default user (override with `--user`) |
| `LLM_TEMPERATURE` | `0.7` | LLM response creativity |
| `DEFAULT_TTS` | `cartesia` | TTS provider (`kokoro` or `cartesia`) |
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
| `no checkpoints found` | Ensure `data/` directory exists; check `logs/agent.log` |
| Memory shows 0 on startup | `data/memory.db/` may not exist yet — send a message first |
| HuggingFace download on first run | `all-MiniLM-L6-v2` (~80 MB) downloads once then caches |
| Groq `tool_use_failed` error | Rephrase — Groq's tool-calling is sensitive to ambiguous queries |
| Audio transcription fails | Check Deepgram key; see `logs/pipecat.log` |
| Kokoro model missing | Auto-downloads ~350 MB on first run |

---

## Resources

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LanceDB Docs](https://lancedb.github.io/lancedb/)
- [Groq Console](https://console.groq.com)
- [Deepgram Docs](https://developers.deepgram.com)
- [Tavily API](https://docs.tavily.com)
