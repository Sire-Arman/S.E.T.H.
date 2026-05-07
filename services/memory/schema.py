"""LanceDB table schemas for memory and checkpoint storage."""
import pyarrow as pa

# Dimension of all-MiniLM-L6-v2 embeddings
EMBEDDING_DIM = 384

# Memory table: stores durable user facts with semantic vectors
MEMORY_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("user_id", pa.string()),
    pa.field("content", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
    pa.field("created_at", pa.string()),
    pa.field("source_session_id", pa.string()),
])

# Checkpoint table: pure key-value snapshot of message history (no vectors)
CHECKPOINT_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("user_id", pa.string()),
    pa.field("session_id", pa.string()),
    pa.field("thread_id", pa.string()),
    pa.field("label", pa.string()),
    pa.field("messages_json", pa.string()),
    pa.field("created_at", pa.string()),
])
