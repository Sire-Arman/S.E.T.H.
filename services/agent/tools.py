"""Agent tools for the LangGraph workflow.

Tools:
    - get_current_datetime — returns the current date, time, and timezone
    - web_search           — real-time web search via Tavily
    - fetch_url            — fetch & extract text from a URL (httpx + trafilatura)
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from langchain_core.tools import tool
from loguru import logger

if TYPE_CHECKING:
    from config import Settings


# ── Date / Time ────────────────────────────────────────────────────


@tool
def get_current_datetime(timezone_name: str = "Asia/Karachi") -> str:
    """Return the current date and time.

    Use this whenever the user asks what time it is, what today's date is,
    what day of the week it is, or any other current date/time question.
    Do NOT use web_search for date or time queries.

    Args:
        timezone_name: IANA timezone string, e.g. 'Asia/Karachi', 'UTC', 'America/New_York'.
                       Defaults to 'Asia/Karachi'.
    """
    try:
        from datetime import datetime
        import zoneinfo
        tz = zoneinfo.ZoneInfo(timezone_name)
        now = datetime.now(tz)
        return now.strftime(f"%A, %B %-d %Y  %I:%M %p  ({timezone_name})")
    except Exception:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        return now.strftime("%A, %B %d %Y  %H:%M UTC")


# ── Web Search ──────────────────────────────────────────────────────


@tool
async def web_search(query: str) -> str:
    """Search the web for real-time, current information.

    ALWAYS use this tool when the user asks about:
    - Sports scores, match results, standings
    - News, current events, recent happenings
    - Weather, stock prices, crypto prices
    - Any factual question you're not 100% certain about
    - "Who won", "what's the score", "what happened", "latest"

    Do NOT refuse to search. Do NOT explain that you can't search.
    Just call this tool with a clear, specific query.

    Args:
        query: A clear, specific search query. Be precise.
              Good: "DC vs KKR IPL 2026 live score today"
              Bad:  "cricket score"
    """
    # ── Try Tavily first ──────────────────────────────────────────
    tavily_result = await _search_tavily(query)
    if tavily_result:
        return tavily_result

    # ── Fallback: Google via fetch_url ─────────────────────────────
    logger.warning("Tavily search failed, falling back to Google scrape")
    return await _search_google_fallback(query)


async def _search_tavily(query: str) -> str | None:
    """Try Tavily search. Returns formatted results or None on failure."""
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        logger.warning("TAVILY_API_KEY not configured")
        return None

    try:
        from langchain_tavily import TavilySearch

        searcher = TavilySearch(
            max_results=3,
            topic="general",
            tavily_api_key=api_key,
        )
        raw = await searcher.ainvoke({"query": query})

        # TavilySearch returns different shapes depending on version:
        #   - dict with "results" key (langchain_tavily >= 1.x)
        #   - list of dicts (older versions)
        #   - str (some error cases)
        if isinstance(raw, str):
            return raw.strip() or None

        # Extract the results list from dict or use directly if already a list
        if isinstance(raw, dict):
            results = raw.get("results", [])
        elif isinstance(raw, list):
            results = raw
        else:
            logger.warning(f"Unexpected Tavily response type: {type(raw)}")
            return None

        if not results:
            return None

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            content = r.get("content", "")
            if content:
                formatted.append(f"[{i}] {title}: {content}")
        return "\n".join(formatted) if formatted else None

    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return None


async def _search_google_fallback(query: str) -> str:
    """Scrape Google search results as a fallback."""
    import httpx
    import trafilatura

    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&hl=en"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/133.0.0.0 Safari/537.36"
        )
    }

    try:
        async with httpx.AsyncClient(
            headers=headers, follow_redirects=True, timeout=15.0
        ) as client:
            resp = await client.get(search_url)
            resp.raise_for_status()

        extracted = trafilatura.extract(
            resp.text, include_comments=False, include_tables=True, output_format="txt"
        )
        if extracted:
            # Trim to keep context small
            return extracted[:3000]
        return "Search returned no usable results. Try a more specific query."
    except Exception as e:
        logger.error(f"Google fallback error: {e}")
        return f"Search unavailable: {str(e)}"


# ── URL Fetcher ─────────────────────────────────────────────────────


@tool
async def fetch_url(url: str) -> str:
    """Fetch and extract text from a specific URL.

    ONLY use this when the user gives you a specific URL to read.
    Do NOT use this for searching — use web_search instead.
    Do NOT invent URLs. Only fetch URLs the user provides.

    Args:
        url: The full URL to fetch (must start with http:// or https://).
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
        max_chars = 4000
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
    return [get_current_datetime, web_search, fetch_url]
