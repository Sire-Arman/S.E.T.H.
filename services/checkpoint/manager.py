"""Conversation checkpoint manager backed by LanceDB.

Stores full message-history snapshots indexed by user_id / session_id /
checkpoint_id.  Supports list, restore, and fork operations so a user can
travel back to any previous state of the conversation.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from loguru import logger

CHECKPOINT_TABLE = "checkpoints"


class CheckpointManager:
    """Save, list, restore, and fork conversation checkpoints.

    Args:
        user_id:    Persistent user identity (e.g. "user_arman_admin" or a UUID).
        session_id: Current conversation session UUID.
        db_path:    Path to the LanceDB checkpoint database file.
    """

    def __init__(self, user_id: str, session_id: str, db_path: str = "./data/checkpoints.db"):
        import lancedb
        self.user_id = user_id
        self.session_id = session_id
        self.db_path = db_path
        self._db = lancedb.connect(db_path)
        self._table = self._get_or_create_table()

    # ── Internal helpers ───────────────────────────────────────────

    def _get_or_create_table(self):
        from services.memory.schema import CHECKPOINT_SCHEMA
        if CHECKPOINT_TABLE in self._db.table_names():
            return self._db.open_table(CHECKPOINT_TABLE)
        logger.info(f"Creating LanceDB checkpoint table at '{self.db_path}'")
        return self._db.create_table(CHECKPOINT_TABLE, schema=CHECKPOINT_SCHEMA)

    @staticmethod
    def _serialize(messages: list) -> str:
        """Serialize LangChain messages to a stable JSON format."""
        serialized = []
        for msg in messages:
            item = {
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
            serialized.append(item)
        return json.dumps(serialized, ensure_ascii=False)

    @staticmethod
    def _deserialize(messages_json: str) -> list:
        """Deserialize JSON back to LangChain message objects."""
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
        _MAP = {"human": HumanMessage, "ai": AIMessage, "system": SystemMessage, "tool": ToolMessage}
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
        """Count checkpoints in the current session (used for auto-labelling)."""
        try:
            df = self._table.to_pandas()
            return int(
                ((df["user_id"] == self.user_id) & (df["session_id"] == self.session_id)).sum()
            )
        except Exception:
            return 0

    # ── Public API ─────────────────────────────────────────────────

    def save(self, messages: list, label: str = "", thread_id: str = "") -> str:
        """Snapshot the current messages. Returns the new checkpoint_id."""
        checkpoint_id = str(uuid.uuid4())
        auto_label = label or f"Turn {self._session_count() + 1} — {datetime.now().strftime('%H:%M:%S')}"
        record = {
            "id": checkpoint_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "thread_id": thread_id or self.session_id,
            "label": auto_label,
            "messages_json": self._serialize(messages),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._table.add([record])
        logger.debug(f"Checkpoint saved: {checkpoint_id[:8]}… ({len(messages)} msgs)")
        return checkpoint_id

    def list_checkpoints(self, session_id: str | None = None) -> list[dict]:
        """List checkpoints for this user.

        Args:
            session_id: Filter to a specific session. Pass ``"__all__"`` to
                        list every session for the user.  Defaults to the
                        current session.
        """
        try:
            df = self._table.to_pandas()
        except Exception:
            return []

        df = df[df["user_id"] == self.user_id]
        target_session = session_id if session_id is not None else self.session_id
        if target_session != "__all__":
            df = df[df["session_id"] == target_session]

        results = []
        for _, row in df.iterrows():
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

        results.sort(key=lambda x: x["created_at"])
        return results

    def restore(self, checkpoint_id: str) -> list:
        """Load and return the messages from a checkpoint.

        Raises:
            ValueError: If the checkpoint_id is not found.
        """
        try:
            df = self._table.to_pandas()
        except Exception as e:
            raise ValueError(f"Could not read checkpoint table: {e}") from e

        row = df[df["id"] == checkpoint_id]
        if row.empty:
            raise ValueError(f"Checkpoint '{checkpoint_id}' not found.")

        messages = self._deserialize(row.iloc[0]["messages_json"])
        logger.info(f"Restored checkpoint {checkpoint_id[:8]}… ({len(messages)} msgs)")
        return messages

    def fork(self, checkpoint_id: str, new_session_id: str | None = None) -> tuple[str, str]:
        """Fork a checkpoint into a new session branch.

        Returns:
            Tuple of (new_checkpoint_id, new_session_id).
        """
        try:
            df = self._table.to_pandas()
        except Exception as e:
            raise ValueError(f"Could not read checkpoint table: {e}") from e

        row = df[df["id"] == checkpoint_id]
        if row.empty:
            raise ValueError(f"Checkpoint '{checkpoint_id}' not found.")

        new_session = new_session_id or str(uuid.uuid4())
        new_checkpoint_id = str(uuid.uuid4())
        original = row.iloc[0]

        record = {
            "id": new_checkpoint_id,
            "user_id": self.user_id,
            "session_id": new_session,
            "thread_id": new_session,
            "label": f"Fork of {checkpoint_id[:8]}… @ {datetime.now().strftime('%H:%M:%S')}",
            "messages_json": original["messages_json"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._table.add([record])
        self.session_id = new_session  # manager switches to new branch
        logger.info(f"Forked {checkpoint_id[:8]}… → session {new_session[:8]}…")
        return new_checkpoint_id, new_session
