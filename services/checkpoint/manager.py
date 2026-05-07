"""Conversation checkpoint manager backed by SQLite.

Stores full message-history snapshots in a single `checkpoints.db` file.
Supports list, restore, and fork operations so a user can travel back
to any previous state of the conversation.

Zero external dependencies — uses Python's built-in sqlite3 module.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager
from loguru import logger


_DDL = """
CREATE TABLE IF NOT EXISTS checkpoints (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    session_id   TEXT NOT NULL,
    thread_id    TEXT NOT NULL,
    label        TEXT NOT NULL,
    messages_json TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_user_session ON checkpoints (user_id, session_id);
"""


class CheckpointManager:
    """Save, list, restore, and fork conversation checkpoints via SQLite.

    Args:
        user_id:    Persistent user identity (e.g. ``"user_arman_admin"`` or a UUID).
        session_id: Current conversation session UUID.
        db_path:    Path to the SQLite database file (single file, no directory).
    """

    def __init__(self, user_id: str, session_id: str, db_path: str = "./data/checkpoints.db"):
        import os
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.user_id = user_id
        self.session_id = session_id
        self.db_path = db_path
        self._init_db()

    # ── Internal helpers ───────────────────────────────────────────

    @contextmanager
    def _conn(self):
        """Yield a thread-safe SQLite connection."""
        con = sqlite3.connect(self.db_path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _init_db(self) -> None:
        with self._conn() as con:
            con.executescript(_DDL)
        logger.debug(f"Checkpoint DB ready: {self.db_path}")

    @staticmethod
    def _serialize(messages: list) -> str:
        """Serialize LangChain messages to a stable JSON string."""
        out = []
        for msg in messages:
            item: dict = {
                "type": msg.type,
                "content": msg.content if isinstance(msg.content, str) else str(msg.content),
            }
            if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                item["tool_call_id"] = msg.tool_call_id
            if hasattr(msg, "name") and msg.name:
                item["name"] = msg.name
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                item["tool_calls"] = [
                    {"id": tc.get("id", ""), "name": tc.get("name", ""), "args": tc.get("args", {})}
                    for tc in msg.tool_calls
                ]
            out.append(item)
        return json.dumps(out, ensure_ascii=False)

    @staticmethod
    def _deserialize(messages_json: str) -> list:
        """Deserialize JSON back to LangChain message objects."""
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
        _MAP = {
            "human": HumanMessage,
            "ai": AIMessage,
            "system": SystemMessage,
            "tool": ToolMessage,
        }
        result = []
        for item in json.loads(messages_json):
            cls = _MAP.get(item["type"], HumanMessage)
            kwargs: dict = {"content": item["content"]}
            if item.get("tool_call_id"):
                kwargs["tool_call_id"] = item["tool_call_id"]
            if item.get("name"):
                kwargs["name"] = item["name"]
            result.append(cls(**kwargs))
        return result

    def _session_count(self) -> int:
        with self._conn() as con:
            row = con.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE user_id=? AND session_id=?",
                (self.user_id, self.session_id),
            ).fetchone()
            return row[0] if row else 0

    # ── Public API ─────────────────────────────────────────────────

    def save(self, messages: list, label: str = "", thread_id: str = "") -> str:
        """Snapshot the current messages. Returns the new checkpoint_id."""
        checkpoint_id = str(uuid.uuid4())
        auto_label = label or f"Turn {self._session_count() + 1} — {datetime.now().strftime('%H:%M:%S')}"
        with self._conn() as con:
            con.execute(
                """INSERT INTO checkpoints
                   (id, user_id, session_id, thread_id, label, messages_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    checkpoint_id,
                    self.user_id,
                    self.session_id,
                    thread_id or self.session_id,
                    auto_label,
                    self._serialize(messages),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        logger.debug(f"Checkpoint saved: {checkpoint_id[:8]}… ({len(messages)} msgs)")
        return checkpoint_id

    def list_checkpoints(self, session_id: str | None = None) -> list[dict]:
        """List checkpoints for this user.

        Args:
            session_id: Filter to a specific session. Pass ``"__all__"`` to
                        list every session for the user. Defaults to the
                        current session.
        """
        target_session = session_id if session_id is not None else self.session_id

        if target_session == "__all__":
            sql = "SELECT * FROM checkpoints WHERE user_id=? ORDER BY created_at ASC"
            params = (self.user_id,)
        else:
            sql = "SELECT * FROM checkpoints WHERE user_id=? AND session_id=? ORDER BY created_at ASC"
            params = (self.user_id, target_session)

        with self._conn() as con:
            rows = con.execute(sql, params).fetchall()

        results = []
        for row in rows:
            try:
                msg_count = len(self._deserialize(row["messages_json"]))
            except Exception:
                msg_count = 0
            results.append({
                "id": row["id"],
                "session_id": row["session_id"],
                "label": row["label"],
                "created_at": row["created_at"],
                "message_count": msg_count,
            })
        return results

    def restore(self, checkpoint_id: str) -> list:
        """Load and return the messages from a checkpoint.

        Raises:
            ValueError: If the checkpoint_id is not found.
        """
        with self._conn() as con:
            row = con.execute(
                "SELECT messages_json FROM checkpoints WHERE id=?",
                (checkpoint_id,),
            ).fetchone()

        if row is None:
            raise ValueError(f"Checkpoint '{checkpoint_id}' not found.")

        messages = self._deserialize(row["messages_json"])
        logger.info(f"Restored checkpoint {checkpoint_id[:8]}… ({len(messages)} msgs)")
        return messages

    def fork(self, checkpoint_id: str, new_session_id: str | None = None) -> tuple[str, str]:
        """Fork a checkpoint into a new session branch.

        Returns:
            Tuple of (new_checkpoint_id, new_session_id).
        """
        with self._conn() as con:
            row = con.execute(
                "SELECT messages_json FROM checkpoints WHERE id=?",
                (checkpoint_id,),
            ).fetchone()

        if row is None:
            raise ValueError(f"Checkpoint '{checkpoint_id}' not found.")

        new_session = new_session_id or str(uuid.uuid4())
        new_checkpoint_id = str(uuid.uuid4())

        with self._conn() as con:
            con.execute(
                """INSERT INTO checkpoints
                   (id, user_id, session_id, thread_id, label, messages_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    new_checkpoint_id,
                    self.user_id,
                    new_session,
                    new_session,
                    f"Fork of {checkpoint_id[:8]}… @ {datetime.now().strftime('%H:%M:%S')}",
                    row["messages_json"],
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

        self.session_id = new_session
        logger.info(f"Forked {checkpoint_id[:8]}… → session {new_session[:8]}…")
        return new_checkpoint_id, new_session

    def delete_session(self, session_id: str | None = None) -> int:
        """Delete all checkpoints for a session. Returns count deleted."""
        sid = session_id or self.session_id
        with self._conn() as con:
            cur = con.execute(
                "DELETE FROM checkpoints WHERE user_id=? AND session_id=?",
                (self.user_id, sid),
            )
        logger.info(f"Deleted {cur.rowcount} checkpoints for session {sid[:8]}…")
        return cur.rowcount
