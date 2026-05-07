"""Semantic memory store backed by LanceDB with sentence-transformer embeddings."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from loguru import logger

MEMORY_TABLE = "memories"


class MemoryStore:
    """Vector-backed persistent memory store.

    Stores durable facts about a user as semantic embeddings in LanceDB.
    Supports fuzzy top-k retrieval so the most relevant facts are surfaced
    per turn regardless of exact wording.
    """

    def __init__(self, user_id: str, db_path: str = "./data/memory.db"):
        import lancedb
        self.user_id = user_id
        self.db_path = db_path
        self._db = lancedb.connect(db_path)
        self._table = self._get_or_create_table()
        self._encoder = None  # lazy-loaded on first use

    # ── Internal helpers ───────────────────────────────────────────

    def _get_encoder(self):
        """Lazy-load the sentence-transformer model."""
        if self._encoder is None:
            import logging
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._encoder

    def _embed(self, text: str) -> list[float]:
        return self._get_encoder().encode(text, show_progress_bar=False).tolist()

    def _get_or_create_table(self):
        from .schema import MEMORY_SCHEMA
        if MEMORY_TABLE in self._db.table_names():
            return self._db.open_table(MEMORY_TABLE)
        logger.info(f"Creating LanceDB memory table at '{self.db_path}'")
        return self._db.create_table(MEMORY_TABLE, schema=MEMORY_SCHEMA)

    # ── Public API ─────────────────────────────────────────────────

    def add(self, content: str, session_id: str = "") -> None:
        """Embed and persist a new memory fact."""
        record = {
            "id": str(uuid.uuid4()),
            "user_id": self.user_id,
            "content": content,
            "vector": self._embed(content),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_session_id": session_id,
        }
        self._table.add([record])
        logger.debug(f"Memory stored [{self.user_id}]: {content[:80]}")

    def search(self, query: str, top_k: int = 5) -> list[str]:
        """Return top-k most relevant memory facts for the query string."""
        try:
            total = self._table.count_rows()
        except Exception:
            return []

        if total == 0:
            return []

        query_vec = self._embed(query)
        try:
            results = (
                self._table.search(query_vec)
                .where(f"user_id = '{self.user_id}'", prefilter=True)
                .limit(top_k)
                .to_list()
            )
            return [r["content"] for r in results]
        except Exception as e:
            logger.warning(f"Memory search failed (non-fatal): {e}")
            return []

    def clear(self) -> int:
        """Delete all memories for the current user. Returns count deleted."""
        try:
            before = self._table.count_rows()
            self._table.delete(f"user_id = '{self.user_id}'")
            logger.info(f"Cleared memories for '{self.user_id}'")
            return before
        except Exception as e:
            logger.error(f"Memory clear failed: {e}")
            return 0

    def count(self) -> int:
        """Total stored memories for this user."""
        try:
            df = self._table.to_pandas()
            return int((df["user_id"] == self.user_id).sum())
        except Exception:
            return 0
