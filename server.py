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
# Sentence splitter (matches LLMProvider._stream_sentences logic)
# ──────────────────────────────────────────────────────────────────

# Split after . ! ? ; : \n when followed by whitespace or end-of-string
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;:\n])(?=\s|$)")
_MIN_WORDS = 5


def split_into_sentences(text: str) -> list[str]:
    """Split response text into sentence-level chunks for TTS.

    Keeps fenced code blocks (```...```) and <artifact> blocks as single chunks.
    Prose is split on sentence boundaries for natural TTS intonation.
    """
    # 1. Separate code blocks and artifacts from prose
    # Pattern matches ```...``` OR <artifact ...>...</artifact>
    block_re = re.compile(r"(```[\s\S]*?```|<artifact[\s\S]*?</artifact>)", re.MULTILINE)
    segments = block_re.split(text)

    sentences = []
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # If it's a code block or artifact, keep it as one chunk
        if (segment.startswith("```") and segment.endswith("```")) or \
           (segment.startswith("<artifact") and segment.endswith("</artifact>")):
            sentences.append(segment)
            continue

        # Otherwise, split prose on sentence boundaries
        parts = _SENTENCE_BOUNDARY.split(segment)
        buffer = ""
        for part in parts:
            buffer += part
            if len(buffer.split()) >= _MIN_WORDS:
                clean = buffer.strip()
                if clean:
                    sentences.append(clean)
                buffer = ""

        remainder = buffer.strip()
        if remainder:
            sentences.append(remainder)

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

def _get_tts_text(chunk: str) -> str | None:
    """Return the text to send to TTS for a given response chunk.

    - Code blocks (```...```)     → brief spoken summary, not the code itself
    - Markdown lists / tables     → brief spoken summary
    - Regular prose               → pass through unchanged
    - Empty / whitespace          → skip (return None)
    """
    stripped = chunk.strip()
    if not stripped:
        return None

    # ── Code blocks / Artifacts: speak a brief summary ──────────────
    if stripped.startswith("```") or stripped.startswith("<artifact"):
        # Handle <artifact> tags
        if stripped.startswith("<artifact"):
            import re as _re
            lang_match = _re.search(r'language="([^"]+)"', stripped)
            title_match = _re.search(r'title="([^"]+)"', stripped)
            lang = lang_match.group(1) if lang_match else "code"
            title = title_match.group(1) if title_match else ""
            
            if title:
                return f"I've generated the {lang} code for {title}."
            return f"I've generated the {lang} code for you."

        # Handle standard ``` blocks
        first_line = stripped.split("\n", 1)[0]
        lang = first_line.replace("```", "").strip()
        if lang:
            return f"Here's some {lang} code."
        return "Here's the code."

    # ── Heavy markdown (many list items): summarize ───────────────
    lines = stripped.split("\n")
    list_lines = sum(1 for l in lines if l.strip().startswith(("- ", "* ", "• ")))
    if list_lines >= 4:
        return f"Here's a list with {list_lines} items."

    numbered_lines = sum(1 for l in lines if l.strip()[:2].rstrip(".").isdigit())
    if numbered_lines >= 4:
        return f"Here are {numbered_lines} points."

    # ── Regular prose: strip markdown formatting for cleaner TTS ──
    import re as _re
    tts = stripped
    # Remove inline code backticks
    tts = _re.sub(r"`([^`]+)`", r"\1", tts)
    # Remove bold/italic markers
    tts = _re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", tts)
    # Remove links [text](url) → just text
    tts = _re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", tts)
    # Remove heading markers
    tts = _re.sub(r"^#{1,6}\s+", "", tts)

    return tts.strip() or None


# ------------------------------------------------------------------
# Helper: invoke agent, split response, stream TTS → WebSocket
# ------------------------------------------------------------------

async def process_and_stream(user_text: str, session: ClientSession, websocket) -> str:
    """
    Invoke the LangGraph agent, then stream the response sentence-by-sentence
    through TTS to the browser.

    Flow:
      1. agent.ainvoke() — runs memory_retrieve → agent → tools → post_process
      2. Extract final AI text from result messages
      3. Split into sentences
      4. For each sentence: TTS synthesize → send 'sentence' + 'audio_response'
      5. Send final 'response' message (signals completion)

    Returns the full response text.
    """
    logger.info(f"User text: {user_text}")

    try:
        # Update checkpoint manager session for this client
        if _checkpoint_manager is not None:
            _checkpoint_manager.session_id = session.session_id

        # Send a "processing" status so the client knows we're alive
        try:
            await websocket.send(json.dumps({
                "type": "status",
                "data": "processing",
            }))
        except websockets.exceptions.ConnectionClosed:
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

        # ── Sanity Filter: Strip raw tool tags (Groq/Llama leakage) ────
        import re as _re
        # Remove <function.web_search(...)>...</function> style tags
        clean_text = _re.sub(r"<function\b.*?</function>", "", raw_text, flags=_re.DOTALL)
        # Remove <|python_tag|>, <|eot_id|>, etc. (Llama internal tokens)
        clean_text = _re.sub(r"<\|[a-z_]+\|>", "", clean_text)
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

        # Split into sentences and stream through TTS
        sentences = split_into_sentences(response_text)
        logger.debug(f"Split response into {len(sentences)} sentence(s)")

        for sentence in sentences:
            try:
                # Send full text to frontend for rendering (always)
                await websocket.send(json.dumps({
                    "type": MessageType.SENTENCE,
                    "data": sentence,
                }))

                # Determine what to speak via TTS
                tts_text = _get_tts_text(sentence)

                if tts_text:
                    wav_bytes = await asyncio.get_event_loop().run_in_executor(
                        None, tts_service.synthesize_wav_bytes, tts_text
                    )
                    if wav_bytes:
                        await websocket.send(json.dumps({
                            "type": MessageType.AUDIO_RESPONSE,
                            "data": base64.b64encode(wav_bytes).decode("ascii"),
                        }))
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"Client disconnected mid-stream (sent {sentences.index(sentence)}/{len(sentences)} sentences)")
                return response_text

        # Send full text response (signals completion)
        try:
            await websocket.send(json.dumps({
                "type": MessageType.RESPONSE,
                "data": response_text,
            }))
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Client disconnected before completion signal")

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
