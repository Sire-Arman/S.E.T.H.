"""Interactive CLI for the LangGraph agent.

Usage:
    python run_agent.py

Loads settings from .env, creates the configured LLM, and starts
a REPL loop where you can chat with the agent.

Phase 0 — simple CLI entry point.
TODO: integrate with WebSocket server for browser-based agent queries.
"""
import asyncio
import sys

from loguru import logger

from config import Settings
from services.agent import create_llm, get_tools, build_agent_graph


SYSTEM_PROMPT = (
    "You are a helpful AI assistant called SETH. "
    "You have access to tools for searching the web and fetching URL content. "
    "Use them when you need current information or when the user asks about a URL. "
    "For general knowledge questions, answer directly without using tools. "
    "Keep your responses clear and concise."
)


async def run_repl():
    """Run the interactive agent REPL."""
    settings = Settings()

    # Create LLM via factory
    provider = settings.AGENT_LLM
    llm = create_llm(provider, settings)

    # Build tools and graph
    tools = get_tools()
    agent = build_agent_graph(llm, tools, system_prompt=SYSTEM_PROMPT)

    tool_names = [t.name for t in tools]
    print(f"\n{'═' * 60}")
    print(f"  SETH Agent  —  Phase 0 CLI")
    print(f"  Provider: {provider}")
    print(f"  Tools: {', '.join(tool_names)}")
    print(f"{'═' * 60}")
    print("  Type your message and press Enter.")
    print("  Commands: 'quit' or 'exit' to stop, 'clear' to reset.\n")

    from langchain_core.messages import HumanMessage

    messages = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            messages.clear()
            print("  [conversation cleared]\n")
            continue

        # Invoke the agent
        try:
            result = await agent.ainvoke(
                {"messages": messages + [HumanMessage(content=user_input)]}
            )

            # Update message history from agent result
            messages = result["messages"]

            # Extract the final AI response
            ai_message = messages[-1]
            response_text = ai_message.content

            # Show tool calls if any happened
            for msg in result["messages"]:
                if hasattr(msg, "type") and msg.type == "tool":
                    logger.debug(f"Tool [{msg.name}] returned {len(msg.content)} chars")

            print(f"\nSETH: {response_text}\n")

        except Exception as e:
            logger.error(f"Agent error: {e}")
            print(f"\n  [Error: {e}]\n")


def main():
    """Entry point."""
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

    asyncio.run(run_repl())


if __name__ == "__main__":
    main()
