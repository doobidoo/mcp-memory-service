"""Regression test for #1622.

A sqlite-vec database written before v8.7.0 has no `metadata` table. The schema
runner keeps `schema_version` there, so on such a database the first pending
migration failed, the runner stopped, and every later migration was skipped.
`retrieve()` and `search_by_tag()` then queried `embedding_pending`, a column
migration 017 never got to add, and returned nothing while the data was intact.
"""

import sqlite3
from pathlib import Path

import pytest
import sqlite_vec
from sqlite_vec import serialize_float32

from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

DIM = 384
MIGRATIONS_DIR = (
    Path(__file__).parents[2] / "src" / "mcp_memory_service" / "storage" / "migrations"
)
AUTH_CONTENT = "Authentication uses short-lived JWT access tokens"
DB_CONTENT = "Schema migrations run at service startup"


def _embedding(seed: int) -> bytes:
    """Deterministic DIM-length float32 vector in sqlite-vec's serialized form."""
    return serialize_float32([((seed * 31 + i) % 97) / 97.0 for i in range(DIM)])


def _build_pre_v8_7_database(db_path: str) -> None:
    """Create the v7 shape: `memories` + `memory_embeddings`, and no `metadata`."""
    conn = sqlite3.connect(db_path)
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute(
            """
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL,
                tags TEXT,
                memory_type TEXT,
                metadata TEXT,
                created_at REAL,
                updated_at REAL,
                created_at_iso TEXT,
                updated_at_iso TEXT
            )
            """
        )
        conn.execute(
            f"CREATE VIRTUAL TABLE memory_embeddings USING vec0("
            f"content_embedding FLOAT[{DIM}])"
        )
        # Tags are stored comma-separated, as MemoryStore writes them.
        conn.executemany(
            "INSERT INTO memories (content_hash, content, tags, memory_type, "
            "created_at, updated_at, created_at_iso, updated_at_iso) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "hash_auth",
                    AUTH_CONTENT,
                    "auth,security",
                    "note",
                    1700000000.0,
                    1700000000.0,
                    "2023-11-14T22:13:20+00:00",
                    "2023-11-14T22:13:20+00:00",
                ),
                (
                    "hash_db",
                    DB_CONTENT,
                    "database",
                    "note",
                    1700000001.0,
                    1700000001.0,
                    "2023-11-14T22:13:21+00:00",
                    "2023-11-14T22:13:21+00:00",
                ),
            ],
        )
        conn.executemany(
            "INSERT INTO memory_embeddings (rowid, content_embedding) VALUES (?, ?)",
            [(1, _embedding(0)), (2, _embedding(1))],
        )
        conn.commit()
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_pre_v8_7_database_finishes_its_upgrade(temp_db_path, monkeypatch):
    # This test is about the migrations, not embeddings: pin the deterministic
    # hash model so it never downloads one when sentence-transformers happens to
    # be installed (conftest only disables the ONNX download).
    monkeypatch.setattr(
        SqliteVecMemoryStorage,
        "_initialize_embedding_model",
        SqliteVecMemoryStorage._initialize_hash_embedding_fallback,
    )
    db_path = str(Path(temp_db_path) / "legacy.db")
    _build_pre_v8_7_database(db_path)

    # Precondition: the fixture really is the shape the bug needs.
    conn = sqlite3.connect(db_path)
    try:
        has_metadata = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'"
        ).fetchone()
    finally:
        conn.close()
    assert has_metadata is None, "fixture must not create the metadata table"

    storage = SqliteVecMemoryStorage(db_path)
    await storage.initialize()
    try:
        # Every migration applied, not just the ones before 008.
        latest_version = max(
            int(path.name[:3]) for path in MIGRATIONS_DIR.glob("[0-9][0-9][0-9]_*.sql")
        )
        version = storage.conn.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()
        assert version is not None, "schema_version was never written"
        assert int(version[0]) == latest_version

        # 017 is the migration whose columns the search paths query.
        columns = {row[1] for row in storage.conn.execute("PRAGMA table_info(memories)")}
        assert "embedding_pending" in columns, "migration 017 was skipped"

        results = await storage.retrieve("authentication tokens", n_results=5)
        assert {result.memory.content for result in results} == {AUTH_CONTENT, DB_CONTENT}

        tagged = await storage.search_by_tag(["auth"])
        assert [memory.content for memory in tagged] == [AUTH_CONTENT]
        assert tagged[0].tags == ["auth", "security"]
    finally:
        await storage.close()
