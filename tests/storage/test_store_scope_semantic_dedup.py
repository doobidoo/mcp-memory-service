"""Store-scoped semantic and exact duplicate checks (#1517)."""

import json
import sqlite3
from collections.abc import Callable
from typing import Any

import pytest
import sqlite_vec

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest.fixture
async def storage(temp_db_path, monkeypatch):
    storage = SqliteVecMemoryStorage(f"{temp_db_path}/store_scope.db")
    await storage.initialize()
    storage.semantic_dedup_enabled = True

    # Keep the test deterministic while exercising the real sqlite-vec query.
    embedding = [1.0] + [0.0] * (storage.embedding_dimension - 1)
    monkeypatch.setattr(storage, "_generate_embedding", lambda _content: embedding)

    yield storage
    await storage.close()


def _memory(content: str) -> Memory:
    return Memory(content=content, content_hash=generate_content_hash(content))


class _MissedLookupCursor:
    def fetchone(self) -> None:
        return None


class _RaceInjectingConnection:
    """Make one exact-hash lookup miss, then let SQLite raise the real conflict."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        on_lookup_miss: Callable[[], None],
    ) -> None:
        self.connection = connection
        self.on_lookup_miss = on_lookup_miss
        self.injected = False
        self.integrity_error_seen = False

    def execute(
        self, sql: str, parameters: tuple[Any, ...] = ()
    ) -> sqlite3.Cursor | _MissedLookupCursor:
        normalized_sql = " ".join(sql.split()).upper()
        if not self.injected and normalized_sql.startswith(
            "SELECT STORE FROM MEMORIES WHERE CONTENT_HASH = ?"
        ):
            self.injected = True
            self.on_lookup_miss()
            return _MissedLookupCursor()

        try:
            return self.connection.execute(sql, parameters)
        except sqlite3.IntegrityError:
            if normalized_sql.startswith("INSERT INTO MEMORIES"):
                self.integrity_error_seen = True
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self.connection, name)


def _insert_competing_exact_memory(
    storage: SqliteVecMemoryStorage, memory: Memory, store: str
) -> None:
    connection = sqlite3.connect(storage.db_path, timeout=5, check_same_thread=False)
    try:
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
        cursor = connection.execute(
            """
            INSERT INTO memories (
                content_hash, content, tags, memory_type, metadata,
                created_at, updated_at, created_at_iso, updated_at_iso, store
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.content_hash,
                memory.content,
                ",".join(memory.tags or []),
                memory.memory_type,
                json.dumps(memory.metadata or {}),
                memory.created_at,
                memory.updated_at,
                memory.created_at_iso,
                memory.updated_at_iso,
                store,
            ),
        )
        connection.execute(
            """
            INSERT INTO memory_embeddings (rowid, content_embedding, store)
            VALUES (?, ?, ?)
            """,
            (
                cursor.lastrowid,
                sqlite_vec.serialize_float32(
                    [1.0] + [0.0] * (storage.embedding_dimension - 1)
                ),
                store,
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _race_connection(
    storage: SqliteVecMemoryStorage, memory: Memory, winning_store: str
) -> _RaceInjectingConnection:
    return _RaceInjectingConnection(
        storage.conn,
        lambda: _insert_competing_exact_memory(storage, memory, winning_store),
    )


def _assert_only_winning_record(
    connection: sqlite3.Connection | _RaceInjectingConnection,
    memory: Memory,
    winning_store: str,
) -> None:
    rows = connection.execute(
        "SELECT id, store FROM memories WHERE content_hash = ?",
        (memory.content_hash,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][1] == winning_store

    embeddings = connection.execute(
        """
        SELECT COUNT(*) FROM memory_embeddings
        WHERE rowid = ?
        """,
        (rows[0][0],),
    ).fetchone()[0]
    assert embeddings == 1


@pytest.mark.asyncio
async def test_semantic_dedup_only_rejects_matches_in_target_store(storage):
    default_memory = _memory("same vector in default")
    docs_memory = _memory("same vector in docs")
    duplicate_docs_memory = _memory("another same-vector doc")

    ok, message = await storage.store(default_memory, store="default")
    assert ok, message

    ok, message = await storage.store(docs_memory, store="docs")
    assert ok, message

    ok, message = await storage.store(duplicate_docs_memory, store="docs")
    assert not ok
    assert "semantically similar" in message
    assert docs_memory.content_hash in message


@pytest.mark.asyncio
async def test_exact_duplicate_names_the_store_that_already_has_the_hash(storage):
    content = "identical content must remain globally unique"
    memory = _memory(content)

    ok, message = await storage.store(memory, store="knowledge")
    assert ok, message

    ok, message = await storage.store(_memory(content), store="default")
    assert not ok
    assert "exact match" in message
    assert "knowledge" in message

    batch_results = await storage.store_batch([_memory(content)], store="default")
    assert not batch_results[0][0]
    assert "knowledge" in batch_results[0][1]


@pytest.mark.asyncio
async def test_store_race_names_winning_store_without_partial_writes(storage):
    memory = _memory("content inserted by a competing writer")
    racing_connection = _race_connection(storage, memory, "winner")
    storage.conn = racing_connection

    ok, message = await storage.store(
        memory, skip_semantic_dedup=True, store="requested"
    )

    assert racing_connection.injected
    assert racing_connection.integrity_error_seen
    assert not ok
    assert "winner" in message
    _assert_only_winning_record(storage.conn, memory, "winner")


@pytest.mark.asyncio
async def test_store_batch_race_names_winning_store_without_partial_writes(storage):
    memory = _memory("batch content inserted by a competing writer")
    racing_connection = _race_connection(storage, memory, "winner")
    storage.conn = racing_connection

    results = await storage.store_batch([memory], store="requested")

    assert racing_connection.injected
    assert racing_connection.integrity_error_seen
    assert not results[0][0]
    assert "winner" in results[0][1]
    _assert_only_winning_record(storage.conn, memory, "winner")
