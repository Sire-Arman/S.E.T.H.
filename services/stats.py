"""
services/stats.py
=================
Global shared stats singleton used by both server.py and server_api.py.
Tracks token usage, per-service request rates, session counts,
and maintains an in-memory log buffer for the control panel.
"""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Callable, Any

# ---------------------------------------------------------------------------
# Rate limit definitions (free tier defaults)
# ---------------------------------------------------------------------------
FREE_LIMITS: dict[str, dict] = {
    "groq": {
        "rpm": 30,          # requests per minute
        "rpd": 14_400,      # requests per day
        "tpd": 500_000,     # tokens per day
        "label": "Groq",
    },
    "deepgram": {
        "rpm": 100,         # generous, main limit is monthly minutes
        "minutes_yr": 12_000,  # free tier minutes/year
        "label": "Deepgram",
    },
    "smallest": {
        "rpm": 100,
        "chars_credit": 200_000,  # ~$1 credit @ $0.175/10k chars ≈ 57k chars; being generous
        "label": "Smallest.ai",
    },
    "langfuse": {
        "rpm": 60,
        "obs_month": 50_000,   # free Hobby tier observations/month
        "label": "Langfuse",
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TokenTotals:
    input_tokens: int = 0
    output_tokens: int = 0
    total_requests: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ServiceBucket:
    name: str
    # Sliding-window RPM tracking
    _timestamps: list = field(default_factory=list)
    requests_total: int = 0
    # TTS-specific
    chars_total: int = 0
    # STT-specific
    audio_seconds: float = 0.0
    # LLM-specific
    tokens_today: int = 0
    requests_today: int = 0
    # Observation count (Langfuse)
    observations_month: int = 0

    def record(self, chars: int = 0, audio_sec: float = 0.0, tokens: int = 0):
        now = time.time()
        self._timestamps = [t for t in self._timestamps if now - t < 60]
        self._timestamps.append(now)
        self.requests_total += 1
        self.chars_total += chars
        self.audio_seconds += audio_sec
        self.tokens_today += tokens
        self.requests_today += 1

    @property
    def rpm(self) -> int:
        now = time.time()
        self._timestamps = [t for t in self._timestamps if now - t < 60]
        return len(self._timestamps)

    def rpm_pct(self) -> float:
        lim = FREE_LIMITS.get(self.name, {}).get("rpm", 60)
        return min(self.rpm / lim * 100, 100)

    def tokens_pct(self) -> float:
        lim = FREE_LIMITS.get(self.name, {}).get("tpd", 1)
        if lim == 1:
            return 0.0
        return min(self.tokens_today / lim * 100, 100)

    def chars_pct(self) -> float:
        lim = FREE_LIMITS.get(self.name, {}).get("chars_credit", 1)
        if lim == 1:
            return 0.0
        return min(self.chars_total / lim * 100, 100)

    def audio_pct(self) -> float:
        lim = FREE_LIMITS.get(self.name, {}).get("minutes_yr", 1)
        if lim == 1:
            return 0.0
        minutes = self.audio_seconds / 60
        return min(minutes / lim * 100, 100)

    def obs_pct(self) -> float:
        lim = FREE_LIMITS.get(self.name, {}).get("obs_month", 1)
        if lim == 1:
            return 0.0
        return min(self.observations_month / lim * 100, 100)


@dataclass
class LogEntry:
    timestamp: float
    level: str      # DEBUG INFO WARNING ERROR
    message: str
    source: str     # module or "server" / "api"


# ---------------------------------------------------------------------------
# Global Stats Store
# ---------------------------------------------------------------------------

class StatsStore:
    """
    Central telemetry hub.  Holds all runtime stats and mutable references
    to live server objects so the HTTP API can hot-reload settings.
    """

    def __init__(self):
        self.start_time = time.time()
        self.session_count: int = 0
        self.token_totals = TokenTotals()

        self.services: dict[str, ServiceBucket] = {
            name: ServiceBucket(name=name)
            for name in ("groq", "deepgram", "smallest", "cartesia", "langfuse")
        }

        self.log_buffer: Deque[LogEntry] = deque(maxlen=300)

        # ── Mutable references set by server.py at startup ──────────────
        # These allow server_api.py to reach into the live server.
        self.tts_service_ref: Any = None          # live TTS service object
        self.rebuild_agent_cb: Callable | None = None  # async fn to rebuild agent
        self.get_settings_cb: Callable | None = None   # fn() → dict of current settings
        self.update_tts_cb: Callable | None = None     # fn(voice, model) → hot-reload TTS

    # ── Token tracking ───────────────────────────────────────────────────

    def record_llm(self, input_tokens: int, output_tokens: int, provider: str = "groq"):
        self.token_totals.input_tokens += input_tokens
        self.token_totals.output_tokens += output_tokens
        self.token_totals.total_requests += 1
        svc = self.services.get(provider)
        if svc:
            svc.record(tokens=input_tokens + output_tokens)

    def record_tts(self, chars: int, provider: str = "smallest"):
        svc = self.services.get(provider)
        if svc:
            svc.record(chars=chars)

    def record_stt(self, audio_seconds: float, provider: str = "deepgram"):
        svc = self.services.get(provider)
        if svc:
            svc.record(audio_sec=audio_seconds)

    def record_observation(self, provider: str = "langfuse"):
        svc = self.services.get(provider)
        if svc:
            svc.record()
            svc.observations_month += 1

    # ── Log buffer ───────────────────────────────────────────────────────

    def add_log(self, level: str, message: str, source: str = "server"):
        self.log_buffer.append(LogEntry(
            timestamp=time.time(),
            level=level,
            message=message,
            source=source,
        ))

    def recent_logs(self, n: int = 100, min_level: str = "DEBUG") -> list[dict]:
        order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        min_idx = order.index(min_level) if min_level in order else 0
        entries = [
            {
                "timestamp": e.timestamp,
                "level": e.level,
                "message": e.message,
                "source": e.source,
            }
            for e in self.log_buffer
            if order.index(e.level) >= min_idx if e.level in order else True
        ]
        return entries[-n:]

    # ── Convenience ─────────────────────────────────────────────────────

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    def snapshot(self) -> dict:
        """Full stats snapshot for /api/stats."""
        return {
            "uptime_seconds": self.uptime_seconds,
            "session_count": self.session_count,
            "tokens": {
                "input": self.token_totals.input_tokens,
                "output": self.token_totals.output_tokens,
                "total": self.token_totals.total_tokens,
                "requests": self.token_totals.total_requests,
            },
            "services": {
                name: {
                    "rpm": svc.rpm,
                    "rpm_pct": round(svc.rpm_pct(), 1),
                    "rpm_limit": FREE_LIMITS.get(name, {}).get("rpm", 60),
                    "requests_total": svc.requests_total,
                    "chars_total": svc.chars_total,
                    "chars_pct": round(svc.chars_pct(), 1),
                    "audio_seconds": round(svc.audio_seconds, 1),
                    "audio_pct": round(svc.audio_pct(), 1),
                    "tokens_today": svc.tokens_today,
                    "tokens_pct": round(svc.tokens_pct(), 1),
                    "observations_month": svc.observations_month,
                    "obs_pct": round(svc.obs_pct(), 1),
                    "label": FREE_LIMITS.get(name, {}).get("label", name),
                }
                for name, svc in self.services.items()
            },
        }


# ── Module-level singleton ───────────────────────────────────────────────────
stats = StatsStore()
