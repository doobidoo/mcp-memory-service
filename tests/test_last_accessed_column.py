"""RED test for issue #1239: retrieve()/access persist must populate the
`last_accessed` COLUMN, not only metadata['last_accessed_at'].

Without the fix, the column stays NULL and staleness/decay measure age.
"""
import pytest
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest.mark.asyncio
async def test_persist_access_populates_last_accessed_column(tmp_path):
    """After persisting access metadata, the last_accessed COLUMN must be set."""
    db = str(tmp_path / "t.db")
    storage = SqliteVecMemoryStorage(db)
    await storage.initialize()

    content = "SQLite-Vec WAL mode enables concurrent access."
    mem = Memory(content=content, content_hash=generate_content_hash(content),
                 tags=["test"], memory_type="decision")
    await storage.store(mem)

    # Simulate an access: record + persist (the retrieve write-back path)
    mem.record_access("how to handle concurrency")
    await storage._persist_access_metadata_batch([mem])

    # The COLUMN must now reflect the access (not NULL).
    def _col():
        cur = storage.conn.execute(
            "SELECT last_accessed FROM memories WHERE content_hash = ?",
            (mem.content_hash,),
        )
        return cur.fetchone()[0]
    col_value = _col()
    assert col_value is not None, "issue #1239: last_accessed column still NULL after access"
    # And it should match the metadata timestamp (same source of truth).
    assert abs(float(col_value) - float(mem.metadata["last_accessed_at"])) < 2, \
        "column and metadata.last_accessed_at diverge"
