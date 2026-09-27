"""Nearest-neighbor limits apply to eligible memories, not filtered-out rows."""

import json
import sqlite3

import pytest
import sqlite_vec

from mcp_memory_service.storage.mixins.retrieve import RetrieveMixin


class QueryStorage(RetrieveMixin):
    """Exercise production retrieval SQL against real sqlite-vec without a model."""

    def __init__(self, conn):
        self.conn = conn
        self.embedding_model = object()

    def _generate_embedding(self, query):
        return [0.0, 0.0]

    async def _execute_with_retry(self, operation):
        return operation()

    def _safe_json_loads(self, value, context):
        return json.loads(value)

    def _effective_confidence(self, confidence, last_accessed, created_at):
        return 1.0

    async def _persist_access_metadata_batch(self, memories):
        pass


@pytest.fixture
def storage():
    """Create an isolated two-dimensional vector table and matching memory rows."""
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("""
        CREATE TABLE memories (
            id INTEGER PRIMARY KEY, content_hash TEXT, content TEXT, tags TEXT,
            memory_type TEXT, metadata TEXT, created_at REAL, updated_at REAL,
            created_at_iso TEXT, updated_at_iso TEXT, deleted_at REAL, superseded_by TEXT
        )
    """)
    conn.execute("CREATE VIRTUAL TABLE memory_embeddings USING vec0(content_embedding float[2], store text)")
    yield QueryStorage(conn)
    conn.close()


def add_memory(storage, row_id, distance, *, created_at=1500, deleted_at=None, tags="keep", superseded_by=None):
    """Seed known distances so expected ranking does not depend on model output."""
    storage.conn.execute(
        "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (row_id, str(row_id), f"Memory {row_id}", tags, "note", "{}", created_at, created_at,
         None, None, deleted_at, superseded_by),
    )
    storage.conn.execute(
        "INSERT INTO memory_embeddings(rowid, content_embedding, store) VALUES (?, ?, ?)",
        (row_id, sqlite_vec.serialize_float32([distance, 0.0]), "default"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["recall", "retrieve"])
@pytest.mark.parametrize("excluded", ["deleted", "before", "after", "outside"])
async def test_time_and_deleted_filters_precede_knn_limit(storage, method, excluded):
    """Even more than 4096 excluded neighbors cannot crowd out valid matches."""
    for row_id in range(1, 4101):
        timestamp = 500 if excluded in ("before", "outside") else 2500 if excluded == "after" else 1500
        add_memory(storage, row_id, 0.1, created_at=timestamp, deleted_at=1600 if excluded == "deleted" else None)
    for row_id in range(4101, 4106):
        add_memory(storage, row_id, float(row_id - 4100))

    bounds = {}
    if excluded in ("before", "outside"):
        bounds["start_timestamp" if method == "recall" else "start_time"] = 1000
    if excluded in ("after", "outside"):
        bounds["end_timestamp" if method == "recall" else "end_time"] = 2000
    results = await getattr(storage, method)("query", n_results=3, **bounds)

    assert [r.memory.content_hash for r in results] == ["4101", "4102", "4103"]
    assert [r.debug_info["distance"] for r in results] == [1.0, 2.0, 3.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("excluded", ["tag", "superseded"])
async def test_retrieve_metadata_filters_precede_knn_limit(storage, excluded):
    """Tag and supersession predicates use the same eligible-row boundary."""
    for row_id in range(1, 4101):
        add_memory(storage, row_id, 0.1, tags="other" if excluded == "tag" else "keep",
                   superseded_by="newer" if excluded == "superseded" else None)
    add_memory(storage, 4101, 1.0)
    results = await storage.retrieve("query", n_results=1, tags=["keep"] if excluded == "tag" else None)
    assert [r.memory.content_hash for r in results] == ["4101"]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["recall", "retrieve"])
async def test_empty_and_smaller_eligible_sets(storage, method):
    """Return all available eligible rows without inventing matches to fill k."""
    add_memory(storage, 1, 0.1, deleted_at=1600)
    assert await getattr(storage, method)("query", n_results=3) == []
    add_memory(storage, 2, 1.0)
    results = await getattr(storage, method)("query", n_results=3)
    assert [r.memory.content_hash for r in results] == ["2"]
