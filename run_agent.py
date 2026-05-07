"""Interactive CLI REPL for the LangGraph agent — Phase 2.

Usage:
    python run_agent.py [--user USER_ID] [--session SESSION_ID]

Phase 2 additions on top of Phase 0:
  - Persistent semantic memory via LanceDB (top-k retrieval + fact extraction)
  - Conversation checkpoints: save, list, restore, fork

CLI Commands (prefix with /):
    /memory              Show last retrieved memory chunks
    /checkpoints         List checkpoints for current session
    /checkpoints all     List all checkpoints across all sessions
    /restore <id>        Restore a past checkpoint (rewinds message history)
    /fork <id>           Fork a checkpoint into a new session branch
    /clear-memory        Delete all stored memories for current user
    /whoami              Show current user_id and session_id
    /help                Show this help
    clear                Reset current conversation (keeps memories)
    quit / exit          Exit
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import uuid

from loguru import logger

from config import Settings
from services.agent import create_llm, get_tools, build_agent_graph


# ── Banner & formatting helpers ────────────────────────────────────

_DIVIDER = "═" * 62

def _banner(provider: str, user_id: str, session_id: str, tools: list) -> None:
    tool_names = ", ".join(t.name for t in tools)
    print(f"\n{_DIVIDER}")
    print(f"  SETH Agent  —  Phase 2  (memory + checkpoints)")
    print(f"  Provider : {provider}")
    print(f"  User     : {user_id}")
    print(f"  Session  : {session_id[:8]}…")
    print(f"  Tools    : {tool_names}")
    print(f"{_DIVIDER}")
    print("  Type a message to chat.  Use /help for commands.\n")


def _fmt_checkpoints(checkpoints: list[dict], current_session: str) -> None:
    if not checkpoints:
        print("  (no checkpoints found)\n")
        return
    print(f"\n  {'ID (short)':<10}  {'Session':<10}  {'Label':<30}  {'Msgs':>4}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*30}  {'-'*4}")
    for cp in checkpoints:
        short_id  = cp["id"][:8]
        short_ses = cp["session_id"][:8]
        is_cur    = " ◀" if cp["session_id"] == current_session else ""
        print(f"  {short_id}  {short_ses}  {cp['label']:<30}  {cp['message_count']:>4}{is_cur}")
    print()


# ── Main REPL ──────────────────────────────────────────────────────

async def run_repl(user_id: str, session_id: str) -> None:
    """Run the interactive agent REPL with memory and checkpoint support."""
    settings = Settings()

    # ── Build Agent LLM ────────────────────────────────────────────
    provider = settings.AGENT_LLM
    llm = create_llm(provider, settings)

    # ── Build Memory Extractor (Cohere) ────────────────────────────
    memory_store     = None
    memory_extractor = None

    if settings.MEMORY_ENABLED:
        from services.memory import MemoryStore, MemoryExtractor
        # Suppress noisy HuggingFace download warnings
        import os as _os
        import warnings
        _os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        _os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
        _os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        warnings.filterwarnings("ignore", message=".*unauthenticated.*")
        warnings.filterwarnings("ignore", message=".*symlinks.*")

        import transformers as _tf
        _tf.logging.set_verbosity_error()

        extraction_llm = create_llm(settings.MEMORY_LLM, settings)
        memory_store     = MemoryStore(user_id=user_id, db_path=settings.MEMORY_DB_PATH)
        memory_extractor = MemoryExtractor(llm=extraction_llm)

        # Pre-warm the encoder so downloads happen before the first prompt
        existing = memory_store.count()
        print(f"  Loading embedding model… ", end="", flush=True)
        memory_store._get_encoder()
        print(f"done  ({existing} existing memories for '{user_id}')")


    # ── Build Checkpoint Manager ───────────────────────────────────
    checkpoint_manager = None
    if settings.CHECKPOINT_ENABLED:
        from services.checkpoint import CheckpointManager
        checkpoint_manager = CheckpointManager(
            user_id=user_id,
            session_id=session_id,
            db_path=settings.CHECKPOINT_DB_PATH,
        )
        logger.info("Checkpoints enabled")

    # ── Build LangGraph ────────────────────────────────────────────
    tools = get_tools()
    agent = build_agent_graph(
        llm=llm,
        tools=tools,
        system_prompt=settings.get_system_instruction(),
        memory_store=memory_store,
        memory_extractor=memory_extractor,
        checkpoint_manager=checkpoint_manager,
        memory_top_k=settings.MEMORY_TOP_K,
    )

    _banner(provider, user_id, session_id, tools)

    messages: list = []
    last_memories: list[str] = []

    # ── REPL loop ──────────────────────────────────────────────────
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # ── Built-in commands ──────────────────────────────────────

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            messages.clear()
            print("  [conversation cleared — memories are preserved]\n")
            continue

        if user_input.lower() == "/help":
            print(__doc__)
            continue

        if user_input.lower() == "/whoami":
            print(f"\n  User    : {user_id}")
            print(f"  Session : {session_id}")
            if checkpoint_manager:
                print(f"  Active session in manager: {checkpoint_manager.session_id[:8]}…")
            print()
            continue

        if user_input.lower() == "/memory":
            if not memory_store:
                print("  [memory is disabled]\n")
            elif not last_memories:
                print("  [no memories retrieved yet — ask me something first]\n")
            else:
                print(f"\n  Last retrieved {len(last_memories)} memory chunk(s):")
                for i, m in enumerate(last_memories, 1):
                    print(f"  {i}. {m}")
                print()
            continue

        if user_input.lower() == "/clear-memory":
            if memory_store:
                count = memory_store.clear()
                print(f"  [cleared {count} memories for '{user_id}']\n")
            else:
                print("  [memory is disabled]\n")
            continue

        if user_input.lower().startswith("/checkpoints"):
            if not checkpoint_manager:
                print("  [checkpoints are disabled]\n")
                continue
            parts = user_input.split()
            scope = "__all__" if len(parts) > 1 and parts[1].lower() == "all" else None
            cps = checkpoint_manager.list_checkpoints(session_id=scope)
            _fmt_checkpoints(cps, checkpoint_manager.session_id)
            continue

        if user_input.lower().startswith("/restore "):
            if not checkpoint_manager:
                print("  [checkpoints are disabled]\n")
                continue
            short_id = user_input.split(None, 1)[1].strip()
            # Support 8-char prefix matching
            try:
                cps = checkpoint_manager.list_checkpoints(session_id="__all__")
                matches = [c for c in cps if c["id"].startswith(short_id)]
                if not matches:
                    print(f"  [no checkpoint matching '{short_id}']\n")
                    continue
                full_id = matches[0]["id"]
                messages = checkpoint_manager.restore(full_id)
                print(f"  [restored checkpoint {full_id[:8]}… — {len(messages)} messages]\n")
            except ValueError as e:
                print(f"  [error: {e}]\n")
            continue

        if user_input.lower().startswith("/fork "):
            if not checkpoint_manager:
                print("  [checkpoints are disabled]\n")
                continue
            short_id = user_input.split(None, 1)[1].strip()
            try:
                cps = checkpoint_manager.list_checkpoints(session_id="__all__")
                matches = [c for c in cps if c["id"].startswith(short_id)]
                if not matches:
                    print(f"  [no checkpoint matching '{short_id}']\n")
                    continue
                full_id = matches[0]["id"]
                new_cp_id, new_session = checkpoint_manager.fork(full_id)
                session_id = new_session  # update local variable
                messages = checkpoint_manager.restore(new_cp_id)
                print(f"  [forked → new session {new_session[:8]}… | checkpoint {new_cp_id[:8]}…]\n")
            except ValueError as e:
                print(f"  [error: {e}]\n")
            continue

        # ── Agent invocation ───────────────────────────────────────

        from langchain_core.messages import HumanMessage

        try:
            result = await agent.ainvoke({
                "messages": messages + [HumanMessage(content=user_input)],
                "memory_context": None,
                "user_id": user_id,
                "session_id": session_id,
                "last_retrieved_memories": [],
            })

            # Sync state
            messages = result["messages"]
            last_memories = result.get("last_retrieved_memories", [])

            # Extract final AI response
            ai_message = messages[-1]
            response_text = ai_message.content if isinstance(ai_message.content, str) else str(ai_message.content)

            # Log tool usage
            for msg in messages:
                if hasattr(msg, "type") and msg.type == "tool":
                    logger.debug(f"Tool [{msg.name}] → {len(msg.content)} chars")

            print(f"\nSETH: {response_text}\n")

        except Exception as e:
            logger.opt(exception=True).error("Agent error: " + str(e).replace("{", "{{").replace("}", "}}"))
            print(f"\n  [Error: {e}]\n")


# ── Entry point ────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SETH Agent CLI — Phase 2")
    parser.add_argument(
        "--user",
        default=None,
        help="User ID (default: value of DEFAULT_USER_ID in settings, or 'user_arman_admin')",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Session ID (default: auto-generated UUID per run)",
    )
    args = parser.parse_args()

    # Resolve user_id: CLI flag > settings > hardcoded default
    settings = Settings()
    user_id   = args.user    or settings.DEFAULT_USER_ID
    session_id = args.session or str(uuid.uuid4())

    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="WARNING",
        format="<dim>{time:HH:mm:ss}</dim> | <level>{level:<7}</level> | {message}",
    )
    logger.add(
        "logs/agent.log",
        level="DEBUG",
        rotation="50 MB",
        retention="7 days",
    )

    asyncio.run(run_repl(user_id=user_id, session_id=session_id))


if __name__ == "__main__":
    main()
