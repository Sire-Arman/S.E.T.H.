"""LangGraph ReAct agent graph.

Builds a simple tool-calling loop:
    agent (LLM) ──▶ tools ──▶ agent ──▶ ... ──▶ END

The agent decides whether to call a tool or respond directly.
"""
from __future__ import annotations

from typing import Annotated, TYPE_CHECKING

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from loguru import logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph


# ── State Schema ───────────────────────────────────────────────────


class AgentState(TypedDict):
    """Conversation state tracked across the graph."""

    messages: Annotated[list, add_messages]


# ── Graph Builder ──────────────────────────────────────────────────


def build_agent_graph(
    llm: "BaseChatModel",
    tools: list,
    system_prompt: str | None = None,
) -> "CompiledStateGraph":
    """Build and compile the ReAct agent graph.

    Args:
        llm: A LangChain chat model (must support ``bind_tools``).
        tools: List of LangChain tool objects.
        system_prompt: Optional system instruction prepended to every invocation.

    Returns:
        A compiled LangGraph that can be invoked with
        ``{"messages": [HumanMessage("...")]}``
    """
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        """Invoke the LLM with the current message history."""
        messages = state["messages"]

        # Prepend system prompt if provided and not already present
        if system_prompt and (
            not messages or messages[0].type != "system"
        ):
            from langchain_core.messages import SystemMessage

            messages = [SystemMessage(content=system_prompt)] + list(messages)

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Build the graph
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    compiled = graph.compile()
    logger.info("Agent graph compiled successfully")
    return compiled
