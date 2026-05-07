"""LangGraph ReAct agent graph — Phase 2.

Turn flow:
    memory_retrieve_node → agent_node ⇌ tools_node → post_process_node → END

- memory_retrieve_node  : fetch top-k memories, inject into system prompt
- agent_node            : standard ReAct LLM + tool-calling loop
- post_process_node     : extract new facts → LanceDB, save checkpoint
"""
from __future__ import annotations

from typing import Annotated, TYPE_CHECKING

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from loguru import logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph
    from services.memory.store import MemoryStore
    from services.memory.extractor import MemoryExtractor
    from services.checkpoint.manager import CheckpointManager


# ── State Schema ───────────────────────────────────────────────────


class AgentState(TypedDict):
    """Conversation state tracked across the graph."""
    messages: Annotated[list, add_messages]
    memory_context: str | None   # injected before agent_node
    user_id: str
    session_id: str
    # Internal: stores last retrieved memory chunks for display in CLI
    last_retrieved_memories: list[str]


# ── Graph Builder ──────────────────────────────────────────────────


def build_agent_graph(
    llm: "BaseChatModel",
    tools: list,
    system_prompt: str | None = None,
    memory_store: "MemoryStore | None" = None,
    memory_extractor: "MemoryExtractor | None" = None,
    checkpoint_manager: "CheckpointManager | None" = None,
    memory_top_k: int = 5,
) -> "CompiledStateGraph":
    """Build and compile the Phase 2 ReAct agent graph.

    Args:
        llm:                LangChain chat model with ``bind_tools`` support.
        tools:              LangChain tool objects.
        system_prompt:      Base system instruction for the agent.
        memory_store:       If provided, enables top-k memory retrieval.
        memory_extractor:   If provided, enables post-turn fact extraction.
        checkpoint_manager: If provided, saves a checkpoint after every turn.
        memory_top_k:       Number of memory chunks to retrieve per turn.

    Returns:
        Compiled LangGraph graph.
    """
    llm_with_tools = llm.bind_tools(tools)

    # ── Node 1: Memory Retrieve ────────────────────────────────────

    async def memory_retrieve_node(state: AgentState) -> dict:
        """Fetch top-k relevant memories and inject into state."""
        if memory_store is None:
            return {"memory_context": None, "last_retrieved_memories": []}

        # Get the latest user message for the query
        user_msg = ""
        for msg in reversed(state["messages"]):
            if hasattr(msg, "type") and msg.type == "human":
                user_msg = msg.content
                break

        if not user_msg:
            return {"memory_context": None, "last_retrieved_memories": []}

        retrieved = memory_store.search(user_msg, top_k=memory_top_k)
        if not retrieved:
            logger.debug("Memory: no relevant memories found")
            return {"memory_context": None, "last_retrieved_memories": []}

        context = "## What I know about you\n" + "\n".join(f"- {m}" for m in retrieved)
        logger.debug(f"Memory: injected {len(retrieved)} chunks")
        return {"memory_context": context, "last_retrieved_memories": retrieved}

    # ── Node 2: Agent ──────────────────────────────────────────────

    def agent_node(state: AgentState) -> dict:
        """Invoke the LLM with memory-augmented system prompt."""
        messages = list(state["messages"])

        # Build system prompt: base + memory context
        full_system = system_prompt or ""
        if state.get("memory_context"):
            full_system = (full_system + "\n\n" + state["memory_context"]).strip()

        if full_system and (not messages or messages[0].type != "system"):
            from langchain_core.messages import SystemMessage
            messages = [SystemMessage(content=full_system)] + messages

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # ── Node 3: Post-process (extract facts + checkpoint) ──────────

    async def post_process_node(state: AgentState) -> dict:
        """After final agent response: extract new facts and save checkpoint."""
        messages = state["messages"]

        # --- Memory extraction ---
        if memory_store is not None and memory_extractor is not None:
            # Find last human and AI messages
            user_msg, ai_msg = "", ""
            for msg in reversed(messages):
                if not ai_msg and hasattr(msg, "type") and msg.type == "ai":
                    ai_msg = msg.content if isinstance(msg.content, str) else ""
                if not user_msg and hasattr(msg, "type") and msg.type == "human":
                    user_msg = msg.content if isinstance(msg.content, str) else ""
                if user_msg and ai_msg:
                    break

            retrieved = state.get("last_retrieved_memories", [])
            new_facts = await memory_extractor.extract(retrieved, user_msg, ai_msg)

            if new_facts:
                # Each bullet point becomes its own memory record
                for line in new_facts.splitlines():
                    line = line.lstrip("•- ").strip()
                    if line:
                        memory_store.add(line, session_id=state.get("session_id", ""))
                logger.info(f"Memory: stored {new_facts.count(chr(10)) + 1} new fact(s)")

        # --- Checkpoint ---
        if checkpoint_manager is not None:
            try:
                cp_id = checkpoint_manager.save(messages)
                logger.debug(f"Checkpoint saved: {cp_id[:8]}…")
            except Exception as e:
                logger.error(f"Checkpoint save failed: {e}")
                print(f"\n  [⚠ Checkpoint save failed: {e}]\n")

        return {}

    # ── Routing: tools_condition replacement ───────────────────────

    def route_after_agent(state: AgentState) -> str:
        """Route to tools if tool calls pending, else to post_process."""
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "post_process"

    # ── Assemble Graph ─────────────────────────────────────────────

    graph = StateGraph(AgentState)

    graph.add_node("memory_retrieve", memory_retrieve_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("post_process", post_process_node)

    graph.set_entry_point("memory_retrieve")
    graph.add_edge("memory_retrieve", "agent")
    graph.add_conditional_edges("agent", route_after_agent, {
        "tools": "tools",
        "post_process": "post_process",
    })
    graph.add_edge("tools", "agent")
    graph.add_edge("post_process", END)

    compiled = graph.compile()
    logger.info("Agent graph compiled (Phase 2: memory + checkpoints)")
    return compiled
