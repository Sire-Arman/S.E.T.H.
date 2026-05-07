"""Lightweight Cohere-based memory fact extractor.

Uses a focused LLM prompt to compare existing stored memories against the
latest exchange and returns only genuinely NEW durable facts, or None.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

_EXTRACTION_PROMPT = """\
You are a memory extraction assistant. Extract DURABLE facts about the user from a conversation.

## Existing Known Facts
{existing_facts}

## New Exchange
User: {user_message}
Assistant: {assistant_message}

## Instructions
List any NEW durable facts (name, job, location, skills, preferences, goals, habits, relationships) \
that are NOT already captured above.
- One concise bullet point per fact.
- If there are NO new facts worth remembering, reply with exactly: None
- Do NOT repeat or rephrase existing facts.
- Do NOT include temporary or conversational details."""


class MemoryExtractor:
    """Uses a Cohere LLM to extract durable facts from a conversation turn."""

    def __init__(self, llm: "BaseChatModel"):
        self.llm = llm

    async def extract(
        self,
        existing_memories: list[str],
        user_message: str,
        assistant_message: str,
    ) -> str | None:
        """Extract new facts from the latest exchange.

        Returns a bullet-point string of new facts, or None if nothing new.
        """
        if not user_message.strip() or not assistant_message.strip():
            return None

        existing_facts = (
            "\n".join(f"- {m}" for m in existing_memories)
            if existing_memories
            else "(none yet)"
        )

        prompt = _EXTRACTION_PROMPT.format(
            existing_facts=existing_facts,
            user_message=user_message[:600],
            assistant_message=assistant_message[:600],
        )

        try:
            from langchain_core.messages import HumanMessage
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = response.content.strip()

            if not result or result.lower().rstrip(".") == "none":
                return None

            logger.debug(f"Memory extracted: {result[:120]}")
            return result

        except Exception as e:
            logger.error(f"Memory extraction failed (non-fatal): {e}")
            return None
