"""
server_api.py
=============
Lightweight aiohttp HTTP REST API server for the SETH Control Panel.
Runs on a separate port (default 8766) alongside the main WebSocket server.

Endpoints:
  GET  /api/stats     → live token/service/uptime snapshot
  GET  /api/settings  → current readable settings
  PATCH /api/settings → hot-reload or flag restart-required settings
  GET  /api/env       → masked .env variables
  PATCH /api/env      → write env var(s) to .env and hot-reload where possible
  GET  /api/prompt    → system.prompt contents
  PUT  /api/prompt    → overwrite system.prompt file
  GET  /api/limits    → rate-limit gauges
  GET  /api/logs      → recent log buffer (backend + merged)
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Any

from aiohttp import web
from aiohttp.web_middlewares import middleware
from loguru import logger

from services.stats import stats, FREE_LIMITS
from config import Settings

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
ENV_PATH = ROOT / ".env"
PROMPT_PATH = ROOT / "system.prompt"

# ── Settings that can be hot-reloaded (no server restart needed) ─────────────
HOT_RELOAD_KEYS = {
    "DEFAULT_TTS", "SMALLEST_VOICE_ID", "SMALLEST_MODEL",
    "CARTESIA_VOICE_ID", "CARTESIA_MODEL",
    "LLM_TEMPERATURE", "LLM_MAX_TOKENS",
    "LOG_LEVEL", "MEMORY_TOP_K",
}

# Settings that require a full server restart ────────────────────────────────
RESTART_KEYS = {
    "AGENT_LLM", "DEFAULT_LLM",
    "GROQ_MODEL", "OPENAI_MODEL", "COHERE_MODEL", "ANTHROPIC_MODEL",
    "GROQ_API_KEY", "OPENAI_API_KEY", "COHERE_API_KEY",
    "ANTHROPIC_API_KEY", "DEEPGRAM_API_KEY", "SMALLEST_API_KEY",
    "CARTESIA_API_KEY", "SERVER_PORT", "SERVER_HOST",
    "MEMORY_ENABLED", "CHECKPOINT_ENABLED",
}

# Settings that need agent graph rebuild (soft restart) ──────────────────────
SOFT_RESTART_KEYS = {"AGENT_LLM", "DEFAULT_LLM"}


# ── CORS middleware ───────────────────────────────────────────────────────────

@middleware
async def cors_middleware(request: web.Request, handler):
    response = await handler(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, PATCH, PUT, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


async def options_handler(request: web.Request) -> web.Response:
    return web.Response(headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, PATCH, PUT, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    })


# ── Helpers ───────────────────────────────────────────────────────────────────

def _json(data: Any, status: int = 200) -> web.Response:
    return web.Response(
        text=json.dumps(data, default=str),
        content_type="application/json",
        status=status,
    )


def _read_env_file() -> dict[str, str]:
    """Parse .env into an ordered dict preserving comments."""
    result = {}
    if not ENV_PATH.exists():
        return result
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, _, val = stripped.partition("=")
            result[key.strip()] = val.strip().strip('"').strip("'")
    return result


def _write_env_file(updates: dict[str, str]) -> None:
    """Merge updates into .env file, preserving all other lines."""
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    updated_keys = set()

    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.partition("=")[0].strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                updated_keys.add(key)
                continue
        new_lines.append(line)

    # Append any new keys that weren't in the file
    for key, val in updates.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={val}")

    ENV_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _mask_value(key: str, value: str) -> str:
    """Mask API key values for safe display."""
    is_secret = any(x in key.upper() for x in ("KEY", "SECRET", "TOKEN", "PASSWORD"))
    if not is_secret or len(value) < 8:
        return value
    return value[:6] + "****" + value[-4:]


def _is_secret(key: str) -> bool:
    return any(x in key.upper() for x in ("KEY", "SECRET", "TOKEN", "PASSWORD"))


# ── Route handlers ────────────────────────────────────────────────────────────

async def handle_stats(request: web.Request) -> web.Response:
    """GET /api/stats — live runtime snapshot."""
    return _json(stats.snapshot())


async def handle_settings_get(request: web.Request) -> web.Response:
    """GET /api/settings — readable subset of current settings."""
    s = Settings
    return _json({
        "agent": {
            "llm": s.AGENT_LLM,
            "temperature": s.LLM_TEMPERATURE,
            "max_tokens": s.LLM_MAX_TOKENS,
            "groq_model": s.GROQ_MODEL,
            "openai_model": s.OPENAI_MODEL,
            "memory_enabled": s.MEMORY_ENABLED,
            "memory_top_k": s.MEMORY_TOP_K,
            "checkpoint_enabled": s.CHECKPOINT_ENABLED,
        },
        "tts": {
            "provider": s.DEFAULT_TTS,
            "smallest_voice": s.SMALLEST_VOICE_ID,
            "smallest_model": s.SMALLEST_MODEL,
            "cartesia_voice": s.CARTESIA_VOICE_ID,
            "cartesia_model": s.CARTESIA_MODEL,
        },
        "server": {
            "host": s.SERVER_HOST,
            "port": s.SERVER_PORT,
            "log_level": s.LOG_LEVEL,
        },
        "system_prompt_preview": (
            Settings.get_system_instruction()[:200] + "..."
            if len(Settings.get_system_instruction()) > 200
            else Settings.get_system_instruction()
        ),
    })


async def handle_settings_patch(request: web.Request) -> web.Response:
    """PATCH /api/settings — apply setting changes, categorize restart needs."""
    try:
        body: dict = await request.json()
    except Exception:
        return _json({"error": "Invalid JSON"}, 400)

    # Map friendly keys back to env var names
    env_map = {
        "temperature": "LLM_TEMPERATURE",
        "max_tokens": "LLM_MAX_TOKENS",
        "llm": "AGENT_LLM",
        "groq_model": "GROQ_MODEL",
        "tts_provider": "DEFAULT_TTS",
        "smallest_voice": "SMALLEST_VOICE_ID",
        "smallest_model": "SMALLEST_MODEL",
        "cartesia_voice": "CARTESIA_VOICE_ID",
        "cartesia_model": "CARTESIA_MODEL",
        "memory_top_k": "MEMORY_TOP_K",
        "log_level": "LOG_LEVEL",
    }

    restart_required = []
    hot_applied = []
    env_updates: dict[str, str] = {}

    for friendly_key, value in body.items():
        env_key = env_map.get(friendly_key, friendly_key.upper())
        env_updates[env_key] = str(value)

        if env_key in RESTART_KEYS:
            restart_required.append(env_key)
        else:
            hot_applied.append(env_key)

    # Write to .env regardless (so restart picks it up)
    _write_env_file(env_updates)

    # Apply hot-reloadable changes to live Settings class
    for env_key, value in env_updates.items():
        if env_key in HOT_RELOAD_KEYS:
            try:
                os.environ[env_key] = value
                # Update Settings class attributes directly
                if env_key == "LLM_TEMPERATURE":
                    Settings.LLM_TEMPERATURE = float(value)
                elif env_key == "LLM_MAX_TOKENS":
                    Settings.LLM_MAX_TOKENS = int(value)
                elif env_key == "LOG_LEVEL":
                    Settings.LOG_LEVEL = value
                elif env_key == "MEMORY_TOP_K":
                    Settings.MEMORY_TOP_K = int(value)
                elif env_key == "DEFAULT_TTS":
                    Settings.DEFAULT_TTS = value
                elif env_key == "SMALLEST_VOICE_ID":
                    Settings.SMALLEST_VOICE_ID = value
                    # Hot-reload TTS service
                    if stats.tts_service_ref and hasattr(stats.tts_service_ref, "voice_id"):
                        stats.tts_service_ref.voice_id = value
                        logger.info(f"[ControlPanel] TTS voice hot-reloaded → {value}")
                elif env_key == "SMALLEST_MODEL":
                    Settings.SMALLEST_MODEL = value
                    if stats.tts_service_ref and hasattr(stats.tts_service_ref, "model_id"):
                        stats.tts_service_ref.model_id = value
                        logger.info(f"[ControlPanel] TTS model hot-reloaded → {value}")
                elif env_key == "CARTESIA_VOICE_ID":
                    Settings.CARTESIA_VOICE_ID = value
                    if stats.tts_service_ref and hasattr(stats.tts_service_ref, "voice_id"):
                        stats.tts_service_ref.voice_id = value
                elif env_key == "CARTESIA_MODEL":
                    Settings.CARTESIA_MODEL = value
                    if stats.tts_service_ref and hasattr(stats.tts_service_ref, "model_id"):
                        stats.tts_service_ref.model_id = value
            except Exception as e:
                logger.warning(f"[ControlPanel] Failed to hot-reload {env_key}: {e}")

    stats.add_log("INFO", f"Settings updated: {list(env_updates.keys())}", "control_panel")

    return _json({
        "applied": hot_applied,
        "restart_required": restart_required,
        "message": (
            "Some changes require a server restart." if restart_required
            else "All changes applied successfully."
        ),
    })


async def handle_env_get(request: web.Request) -> web.Response:
    """GET /api/env — masked .env key-value pairs."""
    env_vars = _read_env_file()
    result = []
    for key, value in env_vars.items():
        result.append({
            "key": key,
            "value": _mask_value(key, value),
            "is_secret": _is_secret(key),
            "restart_required": key in RESTART_KEYS,
            "hot_reload": key in HOT_RELOAD_KEYS,
        })
    return _json(result)


async def handle_env_patch(request: web.Request) -> web.Response:
    """PATCH /api/env — update one or more env vars."""
    try:
        updates: dict = await request.json()
    except Exception:
        return _json({"error": "Invalid JSON"}, 400)

    if not updates:
        return _json({"error": "Empty payload"}, 400)

    _write_env_file(updates)

    restart_required = [k for k in updates if k in RESTART_KEYS]
    hot_applied = [k for k in updates if k in HOT_RELOAD_KEYS]

    # Hot-apply safe env changes
    for k, v in updates.items():
        if k in HOT_RELOAD_KEYS:
            os.environ[k] = str(v)

    stats.add_log("INFO", f"Env vars updated: {list(updates.keys())}", "control_panel")

    return _json({
        "applied": list(updates.keys()),
        "restart_required": restart_required,
        "hot_applied": hot_applied,
        "message": (
            "Saved. Some changes need a server restart to take effect."
            if restart_required else "All changes saved and applied."
        ),
    })


async def handle_prompt_get(request: web.Request) -> web.Response:
    """GET /api/prompt — system.prompt file contents."""
    content = ""
    if PROMPT_PATH.exists():
        content = PROMPT_PATH.read_text(encoding="utf-8")
    return _json({"content": content, "path": str(PROMPT_PATH)})


async def handle_prompt_put(request: web.Request) -> web.Response:
    """PUT /api/prompt — overwrite system.prompt and soft-rebuild agent."""
    try:
        body: dict = await request.json()
        content: str = body.get("content", "")
    except Exception:
        return _json({"error": "Invalid JSON"}, 400)

    if not content.strip():
        return _json({"error": "Prompt cannot be empty"}, 400)

    PROMPT_PATH.write_text(content, encoding="utf-8")
    logger.info("[ControlPanel] system.prompt updated")
    stats.add_log("INFO", "System prompt updated via control panel", "control_panel")

    # Trigger async agent rebuild if callback is registered
    rebuild_required = False
    if stats.rebuild_agent_cb:
        try:
            asyncio.create_task(stats.rebuild_agent_cb())
            rebuild_required = True
            logger.info("[ControlPanel] Agent graph rebuild scheduled")
        except Exception as e:
            logger.warning(f"[ControlPanel] Rebuild failed: {e}")

    return _json({
        "saved": True,
        "rebuild_scheduled": rebuild_required,
        "message": (
            "Prompt saved. Agent is being rebuilt (~10s)."
            if rebuild_required else "Prompt saved. Restart server to apply."
        ),
    })


async def handle_limits(request: web.Request) -> web.Response:
    """GET /api/limits — rate limit gauges for all tracked services."""
    snap = stats.snapshot()
    limits = []
    for name, svc_data in snap["services"].items():
        info = FREE_LIMITS.get(name, {})
        gauges = []

        # RPM gauge (all services)
        rpm = svc_data["rpm"]
        rpm_limit = info.get("rpm", 60)
        gauges.append({
            "label": "Req/min",
            "used": rpm,
            "limit": rpm_limit,
            "pct": round(rpm / rpm_limit * 100, 1) if rpm_limit else 0,
        })

        # Token gauge (LLM services)
        if info.get("tpd"):
            gauges.append({
                "label": "Tokens today",
                "used": svc_data["tokens_today"],
                "limit": info["tpd"],
                "pct": svc_data["tokens_pct"],
            })

        # Char gauge (TTS services)
        if info.get("chars_credit"):
            gauges.append({
                "label": "TTS chars (credit)",
                "used": svc_data["chars_total"],
                "limit": info["chars_credit"],
                "pct": svc_data["chars_pct"],
            })

        # Audio gauge (STT services)
        if info.get("minutes_yr"):
            gauges.append({
                "label": "Audio minutes (yr)",
                "used": round(svc_data["audio_seconds"] / 60, 1),
                "limit": info["minutes_yr"],
                "pct": svc_data["audio_pct"],
            })

        # Observation gauge (Langfuse)
        if info.get("obs_month"):
            gauges.append({
                "label": "Observations (mo)",
                "used": svc_data["observations_month"],
                "limit": info["obs_month"],
                "pct": svc_data["obs_pct"],
            })

        max_pct = max((g["pct"] for g in gauges), default=0)
        limits.append({
            "service": name,
            "label": svc_data["label"],
            "gauges": gauges,
            "status": (
                "critical" if max_pct >= 85 else
                "warning"  if max_pct >= 60 else
                "ok"
            ),
        })

    return _json(limits)


async def handle_logs(request: web.Request) -> web.Response:
    """GET /api/logs?n=100&level=INFO — recent log entries."""
    n = int(request.rel_url.query.get("n", "100"))
    level = request.rel_url.query.get("level", "DEBUG")
    return _json(stats.recent_logs(n=n, min_level=level))


# ── App factory ───────────────────────────────────────────────────────────────

def build_app() -> web.Application:
    app = web.Application(middlewares=[cors_middleware])

    app.router.add_route("OPTIONS", "/{path_info:.*}", options_handler)
    app.router.add_get("/api/stats",    handle_stats)
    app.router.add_get("/api/settings", handle_settings_get)
    app.router.add_route("PATCH", "/api/settings", handle_settings_patch)
    app.router.add_get("/api/env",      handle_env_get)
    app.router.add_route("PATCH", "/api/env", handle_env_patch)
    app.router.add_get("/api/prompt",   handle_prompt_get)
    app.router.add_route("PUT", "/api/prompt", handle_prompt_put)
    app.router.add_get("/api/limits",   handle_limits)
    app.router.add_get("/api/logs",     handle_logs)

    return app


async def start_api_server(host: str, port: int) -> None:
    """Start the aiohttp API server inside the running asyncio event loop."""
    app = build_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info(f"[ControlPanel API] HTTP server ready → http://{host}:{port}/api/")
