"""Agent tools for the LangGraph workflow.

Tools:
    - web_search  — real-time web search via Tavily
    - fetch_url   — fetch & extract text from a URL (httpx + trafilatura)
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from langchain_core.tools import tool
from loguru import logger

if TYPE_CHECKING:
    from config import Settings


# ── Web Search ──────────────────────────────────────────────────────


@tool
async def web_search(query: str) -> str:
    """Search the web for current information about a topic.

    Use this when you need up-to-date information, recent events,
    or facts you are not confident about.
    """
    from langchain_tavily import TavilySearch

    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return "Error: TAVILY_API_KEY is not configured."

    try:
        searcher = TavilySearch(
            max_results=5,
            topic="general",
            tavily_api_key=api_key,
        )
        results = await searcher.ainvoke({"query": query})

        # TavilySearch returns a list of dicts or a string depending on version
        if isinstance(results, str):
            return results

        # Format results into readable text
        if isinstance(results, list):
            formatted = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                url = r.get("url", "")
                content = r.get("content", "")
                formatted.append(f"[{i}] {title}\n    {url}\n    {content}")
            return "\n\n".join(formatted) if formatted else "No results found."

        return str(results)
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return f"Search failed: {str(e)}"


# ── URL Fetcher ─────────────────────────────────────────────────────


@tool
async def fetch_url(url: str) -> str:
    """Fetch and extract the main text content from a URL.

    Use this when the user provides a specific URL and wants you to
    read, summarize, or answer questions about its content.
    """
    import httpx
    import trafilatura

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/133.0.0.0 Safari/537.36"
        )
    }

    try:
        async with httpx.AsyncClient(
            headers=headers,
            follow_redirects=True,
            timeout=30.0,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            html = response.text

        # Extract main content using trafilatura
        extracted = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            output_format="txt",
        )

        if not extracted:
            return f"Could not extract meaningful text content from {url}."

        # Truncate to avoid overwhelming the LLM context
        max_chars = 8000
        if len(extracted) > max_chars:
            extracted = extracted[:max_chars] + "\n\n... [content truncated]"

        return extracted

    except httpx.HTTPStatusError as e:
        return f"HTTP error {e.response.status_code} fetching {url}"
    except httpx.RequestError as e:
        return f"Request failed for {url}: {str(e)}"
    except Exception as e:
        logger.error(f"fetch_url error: {e}")
        return f"Failed to fetch URL: {str(e)}"


# ── Tool Registry ───────────────────────────────────────────────────


def get_tools() -> list:
    """Return the list of tools available to the agent."""
    return [web_search, fetch_url]
