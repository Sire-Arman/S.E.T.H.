"""Pipecat WebSocket server for voice bot — Phase 3.

Uses the full LangGraph ReAct agent (memory + tools + checkpoints)
instead of the simple LLM stream pipeline.  The agent runs ainvoke()
per turn, then the response text is split into sentences and streamed
through TTS → WebSocket exactly as before.
"""
import asyncio
import json
import re
import os
import base64
import uuid
import warnings
from dataclasses import dataclass, field

from loguru import logger
import websockets

from config import Settings
from services.agent import create_llm, get_tools, build_agent_graph
from services.stt import DeepgramSTT
from services.tts import KokoroTTS, CartesiaTTS
from models import MessageType
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langfuse.langchain import CallbackHandler

# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────

settings = Settings()
settings.validate()

# ── STT ───────────────────────────────────────────────────────────

stt_service = DeepgramSTT(
    api_key=settings.DEEPGRAM_API_KEY,
    model=settings.DEEPGRAM_MODEL,
    language=settings.DEEPGRAM_LANGUAGE,
)

# ── TTS ───────────────────────────────────────────────────────────

if settings.DEFAULT_TTS == "cartesia":
    tts_service = CartesiaTTS(
        api_key=settings.CARTESIA_API_KEY,
        voice_id=settings.CARTESIA_VOICE_ID,
        model_id=settings.CARTESIA_MODEL,
    )
else:
    tts_service = KokoroTTS(
        voice="af_heart",
        speed=1.0,
    )

# ── LangGraph Agent (shared, compiled once) ───────────────────────

def _build_agent():
    """Build the shared LangGraph agent at startup."""
    provider = settings.AGENT_LLM
    llm = create_llm(provider, settings)
    tools = get_tools()

    # Memory services (lazy-loaded, shared across sessions)
    memory_store = None
    memory_extractor = None

    if settings.MEMORY_ENABLED:
        # Suppress noisy HuggingFace warnings
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        warnings.filterwarnings("ignore", message=".*unauthenticated.*")
        warnings.filterwarnings("ignore", message=".*symlinks.*")

        import transformers as _tf
        _tf.logging.set_verbosity_error()

        from services.memory import MemoryStore, MemoryExtractor

        memory_store = MemoryStore(
            user_id=settings.DEFAULT_USER_ID,
            db_path=settings.MEMORY_DB_PATH,
        )
        extraction_llm = create_llm(settings.MEMORY_LLM, settings)
        memory_extractor = MemoryExtractor(llm=extraction_llm)

        # Pre-warm the embedding model so first request isn't slow
        logger.info("Pre-warming memory embedding model...")
        memory_store._get_encoder()
        logger.info(f"Memory ready — {memory_store.count()} existing memories for '{settings.DEFAULT_USER_ID}'")

    # Checkpoint manager (shared instance, session_id updated per client)
    checkpoint_manager = None
    if settings.CHECKPOINT_ENABLED:
        from services.checkpoint import CheckpointManager
        checkpoint_manager = CheckpointManager(
            user_id=settings.DEFAULT_USER_ID,
            session_id="server-default",  # overridden per ClientSession
            db_path=settings.CHECKPOINT_DB_PATH,
        )
        logger.info("Checkpoints enabled (server mode)")

    agent = build_agent_graph(
        llm=llm,
        tools=tools,
        system_prompt=settings.get_system_instruction(),
        memory_store=memory_store,
        memory_extractor=memory_extractor,
        checkpoint_manager=checkpoint_manager,
        memory_top_k=settings.MEMORY_TOP_K,
    )

    tool_names = ", ".join(t.name for t in tools)
    logger.info(f"Agent graph compiled — provider={provider}, tools=[{tool_names}]")

    return agent, memory_store, memory_extractor, checkpoint_manager, tools


agent_graph, _memory_store, _memory_extractor, _checkpoint_manager, _tools = _build_agent()

# ── Langfuse Setup ────────────────────────────────────────────────
langfuse_handler = None
if settings.LANGFUSE_ENABLED:
    # Picks up LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST from environment
    langfuse_handler = CallbackHandler()
    logger.info(f"Langfuse observability enabled")


# ──────────────────────────────────────────────────────────────────
# Per-client session state
# ──────────────────────────────────────────────────────────────────

@dataclass
class ClientSession:
    """Per-connection conversation state."""
    user_id: str
    session_id: str
    messages: list = field(default_factory=list)
    last_retrieved_memories: list = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────
# Sentence splitter — robust block-aware chunking
# ──────────────────────────────────────────────────────────────────

# Sentence-ending punctuation followed by whitespace or end-of-string
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_MIN_WORDS = 5

# Regex to detect fenced code blocks (closed AND unclosed) + artifact tags
# Greedy: matches ```lang\n...``` OR ``` through end-of-string (unclosed)
_BLOCK_RE = re.compile(
    r"("
    r"```[^\n]*\n[\s\S]*?(?:```|\Z)"
    r"|"
    r"<artifact[\s\S]*?(?:</artifact>|\Z)"
    r")",
)


def split_into_sentences(text: str) -> list[str]:
    """Split response text into chunks for TTS + display.

    Rules:
      • Fenced code blocks (```...```) → kept as single atomic chunks
      • <artifact> blocks → kept as single atomic chunks
      • Unclosed fences (LLM truncation) → captured through end of text
      • Prose → split on sentence boundaries (.!?) with minimum word count
    """
    segments = _BLOCK_RE.split(text)
    sentences: list[str] = []

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # Atomic block — keep as-is (code or artifact)
        if segment.startswith("```") or segment.startswith("<artifact"):
            sentences.append(segment)
            continue

        # Prose — split on sentence boundaries
        parts = _SENTENCE_BOUNDARY.split(segment)
        buffer = ""
        for part in parts:
            candidate = (buffer + " " + part).strip() if buffer else part.strip()
            if len(candidate.split()) >= _MIN_WORDS:
                sentences.append(candidate)
                buffer = ""
            else:
                buffer = candidate

        if buffer.strip():
            # Merge tiny remainder into last sentence if possible
            if sentences and len(buffer.split()) < 3:
                sentences[-1] += " " + buffer.strip()
            else:
                sentences.append(buffer.strip())

    return sentences


# ──────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────

logger.add(
    "logs/pipecat.log",
    level=settings.LOG_LEVEL,
    rotation="500 MB",
    retention="7 days",
)
logger.info("Pipecat Voice Bot Server initialized (Phase 3 — LangGraph agent)")
logger.info(f"Agent LLM: {settings.AGENT_LLM}")


# ------------------------------------------------------------------
# Smart TTS router — decides what to speak vs. what to only display
# ------------------------------------------------------------------

# Patterns that indicate a line is "code-like" (even without fences)
_CODE_INDICATORS = re.compile(
    r"(?:"
    r"^\s*(?:def |class |import |from |if |for |while |return |raise |try:|except |async |await )"
    r"|^\s*(?:const |let |var |function |=>|export |interface |type )"
    r"|[{}();]\s*$"
    r"|^\s*(?:#include|#define|#pragma)"
    r"|=\s*\[.*\]\s*$"
    r"|=\s*\{.*\}\s*$"
    r"|^\s*@\w+"
    r")",
    re.MULTILINE,
)

# Max number of prose sentences to speak aloud per response
_MAX_SPOKEN_SENTENCES = 3


def _get_tts_text(chunk: str) -> str | None:
    """Return the text to send to TTS for a given response chunk.

    Returns None for chunks that should be display-only (code, tables, etc.).
    For prose, strips markdown formatting for cleaner TTS output.
    """
    stripped = chunk.strip()
    if not stripped:
        return None

    # ── Raw function tags (LLM leakage) → brief spoken summary ───
    if stripped.startswith("<function=") or stripped.startswith("<function "):
        if "generate_code" in stripped:
            return "I've generated the code for you."
        return None  # Unknown function tag — skip TTS

    # ── Fenced code blocks / Artifacts → brief spoken summary ────
    if stripped.startswith("```"):
        first_line = stripped.split("\n", 1)[0]
        lang = first_line.replace("```", "").strip()
        return f"Here's some {lang} code." if lang else "Here's the code."

    if stripped.startswith("<artifact"):
        lang_m = re.search(r'language="([^"]+)"', stripped)
        title_m = re.search(r'title="([^"]+)"', stripped)
        lang = lang_m.group(1) if lang_m else "code"
        title = title_m.group(1) if title_m else ""
        if title:
            return f"I've generated the {lang} code for {title}."
        return f"I've generated the {lang} code for you."

    # ── Code-like content without fences → skip TTS entirely ─────
    lines = stripped.split("\n")
    code_lines = sum(1 for l in lines if _CODE_INDICATORS.search(l))
    if code_lines >= 2 or (len(lines) <= 3 and code_lines >= 1):
        return None  # Looks like code — don't speak

    # ── Markdown tables → summarize ──────────────────────────────
    table_lines = sum(1 for l in lines if l.strip().startswith("|") and "|" in l.strip()[1:])
    if table_lines >= 3:
        return "Here's a table with the results."

    # ── Heavy markdown lists → summarize ─────────────────────────
    list_lines = sum(1 for l in lines if l.strip().startswith(("- ", "* ", "• ")))
    if list_lines >= 4:
        return f"Here's a list with {list_lines} items."

    numbered = sum(1 for l in lines if re.match(r"^\s*\d+[.)]", l.strip()))
    if numbered >= 4:
        return f"Here are {numbered} points."

    # ── Math blocks → skip TTS ───────────────────────────────────
    if stripped.startswith("$$") or stripped.endswith("$$"):
        return "Here's a mathematical expression."

    # ── Regular prose → clean markdown formatting for TTS ────────
    tts = stripped
    tts = re.sub(r"`([^`]+)`", r"\1", tts)            # inline code
    tts = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", tts) # bold/italic
    tts = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", tts)  # links
    tts = re.sub(r"^#{1,6}\s+", "", tts, flags=re.MULTILINE)  # headings
    tts = re.sub(r"!\[.*?\]\(.*?\)", "", tts)            # images
    tts = re.sub(r"\n+", " ", tts)                       # collapse newlines
    tts = re.sub(r"\s{2,}", " ", tts)                    # collapse spaces

    return tts.strip() or None


# ------------------------------------------------------------------
# Helper: invoke agent, split response, stream TTS → WebSocket
# ------------------------------------------------------------------

async def _ws_send(websocket, msg_type: str, data: str) -> bool:
    """Send a JSON message, returning False if the client disconnected."""
    try:
        await websocket.send(json.dumps({"type": msg_type, "data": data}))
        return True
    except websockets.exceptions.ConnectionClosed:
        return False


async def process_and_stream(user_text: str, session: ClientSession, websocket) -> str:
    """
    Invoke the LangGraph agent, then stream the response into a single
    chat bubble via bot_start → bot_chunk → bot_end protocol.

    TTS is limited to the first few prose sentences to keep audio concise
    while the full response is always displayed.

    Returns the full response text.
    """
    logger.info(f"User text: {user_text}")

    try:
        # Update checkpoint manager session for this client
        if _checkpoint_manager is not None:
            _checkpoint_manager.session_id = session.session_id

        # Send a "processing" status so the client knows we're alive
        if not await _ws_send(websocket, "status", "processing"):
            logger.warning("Client disconnected before agent invocation")
            return ""

        # Invoke the full agent graph
        import time as _time
        _t0 = _time.perf_counter()
        logger.debug(f"Invoking agent graph (provider={settings.AGENT_LLM})...")

        # Configuration for Langchain (checkpoints + observability)
        config = {
            "configurable": {
                "user_id": session.user_id,
                "session_id": session.session_id,
            },
            "callbacks": [langfuse_handler] if langfuse_handler else [],
            "metadata": {
                "user_id": session.user_id,
                "session_id": session.session_id,
                "agent": settings.AGENT_LLM
            }
        }

        result = await agent_graph.ainvoke({
            "messages": session.messages + [HumanMessage(content=user_text)],
            "memory_context": None,
            "user_id": session.user_id,
            "session_id": session.session_id,
            "last_retrieved_memories": [],
        }, config=config)
        _elapsed = _time.perf_counter() - _t0
        logger.info(f"Agent ainvoke completed in {_elapsed:.2f}s")

        # Sync session state from graph result
        session.messages = result["messages"]
        session.last_retrieved_memories = result.get("last_retrieved_memories", [])

        # Extract the final AI response text
        ai_message = session.messages[-1]
        raw_text = (
            ai_message.content
            if isinstance(ai_message.content, str)
            else str(ai_message.content)
        )

        # ── Collect generate_code tool results from this turn ──────────
        # When the LLM uses generate_code, the code is in ToolMessage.content
        # (an <artifact> tag), but the final AIMessage only has the prose summary.
        # We need to prepend the artifact blocks so the frontend renders them.
        code_artifacts = []
        for msg in reversed(session.messages):
            # Stop scanning when we reach the user's HumanMessage (start of turn)
            if hasattr(msg, "type") and msg.type == "human":
                break
            if hasattr(msg, "type") and msg.type == "tool" and getattr(msg, "name", "") == "generate_code":
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                if content.strip():
                    code_artifacts.append(content.strip())

        # Prepend code artifacts (in order) before the prose summary
        if code_artifacts:
            code_artifacts.reverse()  # put them back in original order
            raw_text = "\n\n".join(code_artifacts) + "\n\n" + raw_text
            logger.debug(f"Prepended {len(code_artifacts)} code artifact(s) to response")

        # ── Sanity Filter: Strip / convert raw tool tags (Groq/Llama leakage)
        clean_text = raw_text

        # Convert <function=generate_code>{JSON}</function> → proper <artifact> tags
        def _convert_generate_code(m):
            try:
                import json as _json
                payload = _json.loads(m.group(1))
                lang = payload.get("language", "text")
                code = payload.get("code", "")
                title = payload.get("title", "")
                return f'<artifact language="{lang}" title="{title}">\n{code}\n</artifact>'
            except Exception:
                return ""  # Strip if we can't parse

        clean_text = re.sub(
            r"<function=generate_code>\s*(\{[\s\S]*?\})\s*</function>",
            _convert_generate_code,
            clean_text,
        )

        # Remove any remaining <function...>...</function> style tags
        clean_text = re.sub(r"<function[\s=].*?</function>", "", clean_text, flags=re.DOTALL)
        # Remove <|python_tag|>, <|eot_id|>, etc. (Llama internal tokens)
        clean_text = re.sub(r"<\|[a-z_]+\|>", "", clean_text)
        response_text = clean_text.strip()

        # Fallback if the model only outputted tool calls but no response text
        if not response_text:
            if any(hasattr(m, "tool_calls") and m.tool_calls for m in session.messages[-2:]):
                 response_text = "I couldn't find anything useful on that. Try being more specific."
            else:
                 response_text = "What's up? I'm ready when you are."

        logger.info(f"Bot response ({len(response_text)} chars): {response_text[:120]}...")

        # Log tool usage during this turn
        for msg in session.messages:
            if hasattr(msg, "type") and msg.type == "tool":
                logger.debug(f"Tool [{msg.name}] → {len(msg.content)} chars")

        # ── Stream response using bot_start / bot_chunk / bot_end ──

        # 1. Signal the frontend to create an empty bot bubble
        if not await _ws_send(websocket, MessageType.BOT_START, ""):
            return response_text

        # 2. Split into display chunks and stream them
        sentences = split_into_sentences(response_text)
        logger.debug(f"Split response into {len(sentences)} chunk(s)")

        spoken_count = 0  # Track how many prose sentences we've spoken

        for i, sentence in enumerate(sentences):
            # Send the chunk text for display (appends to the current bubble)
            if not await _ws_send(websocket, MessageType.BOT_CHUNK, sentence):
                logger.warning(f"Client disconnected mid-stream (chunk {i}/{len(sentences)})")
                return response_text

            # Determine what to speak via TTS (capped at _MAX_SPOKEN_SENTENCES)
            if spoken_count < _MAX_SPOKEN_SENTENCES:
                tts_text = _get_tts_text(sentence)
                if tts_text:
                    spoken_count += 1
                    logger.info(f"TTS [{spoken_count}/{_MAX_SPOKEN_SENTENCES}]: {tts_text[:60]}...")
                    try:
                        wav_bytes = await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(
                                None, tts_service.synthesize_wav_bytes, tts_text
                            ),
                            timeout=30.0,  # 30s safety net per sentence
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"TTS timed out for: {tts_text[:60]}...")
                        wav_bytes = None
                    except Exception as tts_err:
                        logger.error(f"TTS error: {tts_err}", exc_info=True)
                        wav_bytes = None

                    if wav_bytes:
                        logger.info(f"TTS audio produced: {len(wav_bytes)} bytes, sending to client...")
                        if not await _ws_send(
                            websocket,
                            MessageType.AUDIO_RESPONSE,
                            base64.b64encode(wav_bytes).decode("ascii"),
                        ):
                            return response_text
                        logger.info(f"TTS audio_response sent successfully")
                    else:
                        logger.warning(f"TTS produced empty/null wav_bytes for: {tts_text[:60]}...")

        # 3. Finalize the bot bubble
        if not await _ws_send(websocket, MessageType.BOT_END, ""):
            logger.warning("Client disconnected before bot_end")

        # 4. Send full response for backward compat / completion signal
        await _ws_send(websocket, MessageType.RESPONSE, response_text)

        return response_text

    except Exception as api_error:
        logger.error(f"Agent pipeline error: {api_error}")
        try:
            await websocket.send(json.dumps({
                "type": MessageType.ERROR,
                "data": f"Agent error: {str(api_error)}",
            }))
        except websockets.exceptions.ConnectionClosed:
            pass
        return ""


async def handle_client(websocket):
    """Handle incoming WebSocket client connection."""
    client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
    logger.info(f"Client connected: {client_id}")

    # Create per-client session with unique session_id
    session = ClientSession(
        user_id=settings.DEFAULT_USER_ID,
        session_id=str(uuid.uuid4()),
    )
    logger.info(f"Session created: {session.session_id[:8]}… (user={session.user_id})")

    try:
        async for message in websocket:
            try:
                # Try to parse as JSON
                data = json.loads(message)
                logger.debug(f"Received from {client_id}: {data.get('type')}")

                if data.get("type") == MessageType.TEXT or data.get("type") == 'message':
                    user_message = data.get("data", "").strip()
                    if not user_message:
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": MessageType.ERROR,
                                    "data": "Empty message",
                                }
                            )
                        )
                        continue

                    await process_and_stream(user_message, session, websocket)

                elif data.get("type") == MessageType.AUDIO or data.get("type") == "audio":
                    logger.info("Audio message received")

                    audio_data = data.get("data", "")
                    if not audio_data:
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": MessageType.ERROR,
                                    "data": "Empty audio data",
                                }
                            )
                        )
                        continue

                    try:
                        # Decode base64 audio
                        logger.debug("Decoding base64 audio...")
                        audio_bytes = base64.b64decode(audio_data)
                        logger.info(f"Audio received: {len(audio_bytes)} bytes")

                        # Transcribe using Deepgram
                        logger.debug("Transcribing audio with Deepgram...")
                        transcript = await stt_service.transcribe(audio_bytes)

                        if not transcript:
                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": MessageType.ERROR,
                                        "data": "Failed to transcribe audio",
                                    }
                                )
                            )
                            continue

                        # Send transcript back to client for display
                        await websocket.send(json.dumps({
                            "type": "transcript",
                            "data": transcript,
                        }))

                        await process_and_stream(transcript, session, websocket)

                    except Exception as audio_error:
                        logger.error(f"Audio processing error: {audio_error}")
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": MessageType.ERROR,
                                    "data": f"Audio processing error: {str(audio_error)}",
                                }
                            )
                        )

                else:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": MessageType.ERROR,
                                "data": f"Unknown message type: {data.get('type')}",
                            }
                        )
                    )

            except json.JSONDecodeError:
                # Not JSON, treat as plain text message
                user_message = message.strip()
                logger.info(f"Text message from {client_id}: {user_message}")
                await process_and_stream(user_message, session, websocket)

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                try:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": MessageType.ERROR,
                                "data": f"Server error: {str(e)}",
                            }
                        )
                    )
                except:
                    pass

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"Client error: {e}")


async def main():
    """Run the WebSocket server."""
    logger.info(f"Starting server on ws://{settings.SERVER_HOST}:{settings.SERVER_PORT}")

    try:
        async with websockets.serve(
            handle_client, settings.SERVER_HOST, settings.SERVER_PORT,
            ping_interval=30,
            ping_timeout=30,
            close_timeout=10,
        ):
            logger.info("WebSocket server ready - waiting for connections")
            await asyncio.Future()  # run forever
    except Exception as e:
        logger.error(f"Server error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
