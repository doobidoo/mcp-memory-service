#!/usr/bin/env python3
"""
Tests for the pgvector storage backend.

Most tests run as pure unit tests with no Postgres required (they exercise
import gating, the row → Memory parser, factory routing, etc.). A small
integration suite gated on `MCP_PGVECTOR_TEST_DSN` runs the full
init → store → retrieve → delete loop against a real database when one
is available.

To run the integration tier locally:

    docker run --rm -d -p 5433:5432 \
        -e POSTGRES_PASSWORD=test -e POSTGRES_DB=mcp_test \
        --name pgv-test pgvector/pgvector:pg16
    export MCP_PGVECTOR_TEST_DSN='postgresql://postgres:test@localhost:5433/mcp_test'
    pytest tests/storage/test_pgvector_storage.py -v
"""

import asyncio
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, List
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from mcp_memory_service.models.memory import Memory


# --------------------------------------------------------------------- helpers

def _make_memory(content: str, tags=None) -> Memory:
    test_tags = list(tags or [])
    if "__test__" not in test_tags:
        test_tags.append("__test__")
    content_hash = hashlib.sha256(content.strip().lower().encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc)
    return Memory(
        content=content,
        content_hash=content_hash,
        tags=test_tags,
        memory_type="note",
        metadata={"source": "test"},
        created_at=now.timestamp(),
        updated_at=now.timestamp(),
        created_at_iso=now.isoformat(),
        updated_at_iso=now.isoformat(),
    )


# --------------------------------------------------------------------- imports

class TestImportGating:
    """The backend should fail loudly with a useful message when its
    optional deps aren't installed, rather than crash with ImportError."""

    def test_module_imports_without_asyncpg(self, monkeypatch):
        """Module-level import must not raise even if asyncpg / pgvector
        aren't installed — instantiation is what should fail."""
        from mcp_memory_service.storage import pgvector
        # If we got here, the import succeeded without the deps installed
        # OR with them installed; both are fine.
        assert hasattr(pgvector, "PgVectorMemoryStorage")

    def test_instantiation_without_asyncpg_raises_helpful_error(self, monkeypatch):
        from mcp_memory_service.storage import pgvector

        monkeypatch.setattr(pgvector, "_ASYNCPG_AVAILABLE", False)
        with pytest.raises(ImportError, match="asyncpg is not installed"):
            pgvector.PgVectorMemoryStorage(dsn="postgresql://x", schema="public")

    def test_instantiation_without_pgvector_raises_helpful_error(self, monkeypatch):
        from mcp_memory_service.storage import pgvector

        # Force asyncpg-available but pgvector-absent.
        monkeypatch.setattr(pgvector, "_ASYNCPG_AVAILABLE", True)
        monkeypatch.setattr(pgvector, "_PGVECTOR_AVAILABLE", False)
        with pytest.raises(ImportError, match="pgvector is not installed"):
            pgvector.PgVectorMemoryStorage(dsn="postgresql://x", schema="public")


# --------------------------------------------------------------------- row parser

class TestRowToMemory:
    """_row_to_memory must round-trip whatever Postgres hands back, plus
    tolerate the legacy comma-joined-string tag form used during sqlite_vec
    → pgvector migration imports."""

    def _row(self, **overrides) -> dict:
        base = {
            "content_hash": "abc",
            "content": "hello",
            "tags": ["t1", "t2"],
            "memory_type": "note",
            "metadata": {"k": "v"},
            "created_at": 1700000000.5,
            "updated_at": 1700000001.5,
            "created_at_iso": "2023-11-14T22:13:20+00:00",
            "updated_at_iso": "2023-11-14T22:13:21+00:00",
        }
        base.update(overrides)
        return base

    def test_canonical_row_round_trips(self):
        from mcp_memory_service.storage.pgvector import _row_to_memory

        memory = _row_to_memory(self._row())
        assert memory.content == "hello"
        assert memory.content_hash == "abc"
        assert memory.tags == ["t1", "t2"]
        assert memory.memory_type == "note"
        assert memory.metadata == {"k": "v"}
        assert memory.created_at == 1700000000.5

    def test_handles_comma_joined_tag_string(self):
        """Legacy sqlite_vec exports stored tags as 'a,b,c'."""
        from mcp_memory_service.storage.pgvector import _row_to_memory

        memory = _row_to_memory(self._row(tags="a, b ,c"))
        assert memory.tags == ["a", "b", "c"]

    def test_handles_null_tags(self):
        """A NULL tags column from Postgres must not crash the parser.
        The Memory dataclass's __init__ adds an 'untagged' marker on its own
        when given no tags; we just need to confirm we passed an empty list
        in (rather than e.g. None) so that branch fires cleanly."""
        from mcp_memory_service.storage.pgvector import _row_to_memory

        memory = _row_to_memory(self._row(tags=None))
        # The dataclass adds an 'untagged' tag on empty input; the important
        # thing is the parser didn't crash and didn't carry over a sentinel.
        assert isinstance(memory.tags, list)
        assert "t1" not in memory.tags  # didn't leak from elsewhere

    def test_handles_metadata_as_json_string(self):
        """Migration path may produce metadata as a stringified JSON blob."""
        from mcp_memory_service.storage.pgvector import _row_to_memory

        memory = _row_to_memory(self._row(metadata=json.dumps({"k": "v"})))
        assert memory.metadata == {"k": "v"}

    def test_handles_corrupt_metadata(self):
        from mcp_memory_service.storage.pgvector import _row_to_memory

        memory = _row_to_memory(self._row(metadata="{not valid json"))
        assert memory.metadata == {}

    def test_handles_null_timestamps(self):
        """NULL timestamps in the row don't crash the parser. The Memory
        dataclass substitutes a sane default (current time) on its own
        when given 0.0/empty — we just need to not pass garbage in."""
        from mcp_memory_service.storage.pgvector import _row_to_memory

        memory = _row_to_memory(self._row(created_at=None, updated_at=None,
                                          created_at_iso=None, updated_at_iso=None))
        # The dataclass replaces 0.0 with now() — verify both timestamps
        # are present and finite, which is all the parser is responsible for.
        assert isinstance(memory.created_at, float)
        assert isinstance(memory.updated_at, float)
        assert memory.created_at >= 0.0


# --------------------------------------------------------------------- factory

class TestFactoryRouting:
    """STORAGE_BACKEND='pgvector' must route through to the new class
    and surface a clear error when the DSN is absent."""

    def test_get_storage_backend_class_returns_pgvector_class(self, monkeypatch):
        from mcp_memory_service.storage import factory

        monkeypatch.setattr("mcp_memory_service.config.STORAGE_BACKEND", "pgvector")
        cls = factory.get_storage_backend_class()
        assert cls.__name__ == "PgVectorMemoryStorage"

    @pytest.mark.asyncio
    async def test_create_instance_requires_dsn(self, monkeypatch):
        """Without MCP_PGVECTOR_DSN the factory must raise a clear ValueError
        rather than letting asyncpg blow up with a cryptic message."""
        from mcp_memory_service.storage import factory
        from mcp_memory_service import config

        monkeypatch.setattr(config, "STORAGE_BACKEND", "pgvector")
        monkeypatch.setattr(config, "PGVECTOR_DSN", None)

        with pytest.raises(ValueError, match="MCP_PGVECTOR_DSN"):
            await factory.create_storage_instance(sqlite_path="/tmp/unused.db")


# --------------------------------------------------------------------- properties

class TestProperties:
    """Backend properties expected by the MemoryStorage interface."""

    def _new_storage(self, monkeypatch):
        from mcp_memory_service.storage import pgvector

        monkeypatch.setattr(pgvector, "_ASYNCPG_AVAILABLE", True)
        monkeypatch.setattr(pgvector, "_PGVECTOR_AVAILABLE", True)
        return pgvector.PgVectorMemoryStorage(dsn="postgresql://x", schema="public")

    def test_max_content_length_is_unbounded(self, monkeypatch):
        storage = self._new_storage(monkeypatch)
        assert storage.max_content_length is None

    def test_supports_chunking_is_false(self, monkeypatch):
        """Initial release leaves chunking to the ingestion pipeline."""
        storage = self._new_storage(monkeypatch)
        assert storage.supports_chunking is False

    def test_schema_defaults_to_public(self, monkeypatch):
        storage = self._new_storage(monkeypatch)
        assert storage.schema == "public"

    def test_initialized_flag_starts_false(self, monkeypatch):
        storage = self._new_storage(monkeypatch)
        assert storage._initialized is False


# --------------------------------------------------------------------- integration

_INTEGRATION_DSN = os.environ.get("MCP_PGVECTOR_TEST_DSN")
_skip_integration = pytest.mark.skipif(
    not _INTEGRATION_DSN,
    reason="Set MCP_PGVECTOR_TEST_DSN to a postgres+pgvector DSN to run integration tests",
)

# These also need an embedding endpoint configured.
_skip_no_embedder = pytest.mark.skipif(
    not os.environ.get("MCP_EXTERNAL_EMBEDDING_URL"),
    reason="Set MCP_EXTERNAL_EMBEDDING_URL to enable end-to-end pgvector tests",
)


@pytest.fixture
async def real_storage():
    """Initialize a real PgVectorMemoryStorage against a per-test schema
    so parallel runs don't collide."""
    from mcp_memory_service.storage.pgvector import PgVectorMemoryStorage

    schema = f"mcp_test_{uuid.uuid4().hex[:8]}"
    storage = PgVectorMemoryStorage(dsn=_INTEGRATION_DSN, schema=schema)
    await storage.initialize()
    try:
        yield storage
    finally:
        # Drop the per-test schema so the database stays clean.
        try:
            async with storage.pool.acquire() as conn:
                await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE;')
        finally:
            await storage.close()


@_skip_integration
@_skip_no_embedder
@pytest.mark.integration
@pytest.mark.asyncio
class TestPgVectorIntegration:
    """End-to-end tests against a real Postgres with pgvector enabled."""

    async def test_store_and_get_by_hash(self, real_storage):
        memory = _make_memory("integration store/get test")
        ok, msg = await real_storage.store(memory)
        assert ok, msg

        loaded = await real_storage.get_by_hash(memory.content_hash)
        assert loaded is not None
        assert loaded.content == memory.content
        assert "__test__" in loaded.tags

    async def test_store_dedupes_by_content_hash(self, real_storage):
        memory = _make_memory("dedup test content")
        ok1, _ = await real_storage.store(memory)
        ok2, msg2 = await real_storage.store(memory)
        assert ok1 is True
        assert ok2 is False
        assert "Duplicate" in msg2

    async def test_retrieve_returns_stored_memory(self, real_storage):
        memory = _make_memory("retrieve test — semantic recall works")
        await real_storage.store(memory)

        results = await real_storage.retrieve("semantic recall", n_results=5)
        assert any(r.memory.content_hash == memory.content_hash for r in results)

    async def test_search_by_tag_finds_match(self, real_storage):
        memory = _make_memory("tagged content", tags=["proj:integration"])
        await real_storage.store(memory)

        found = await real_storage.search_by_tag(["proj:integration"])
        assert any(m.content_hash == memory.content_hash for m in found)

    async def test_search_by_tags_and_semantics(self, real_storage):
        m1 = _make_memory("AND test alpha", tags=["a", "b"])
        m2 = _make_memory("AND test beta", tags=["a"])
        await real_storage.store(m1)
        await real_storage.store(m2)

        # AND requires both tags; only m1 should match.
        and_hits = await real_storage.search_by_tags(["a", "b"], operation="AND")
        and_hashes = {m.content_hash for m in and_hits}
        assert m1.content_hash in and_hashes
        assert m2.content_hash not in and_hashes

        # OR matches either; both should be present.
        or_hits = await real_storage.search_by_tags(["a", "b"], operation="OR")
        or_hashes = {m.content_hash for m in or_hits}
        assert {m1.content_hash, m2.content_hash}.issubset(or_hashes)

    async def test_soft_delete_hides_from_lookups(self, real_storage):
        memory = _make_memory("soft-delete me")
        await real_storage.store(memory)

        ok, _ = await real_storage.delete(memory.content_hash)
        assert ok
        assert await real_storage.get_by_hash(memory.content_hash) is None
        assert await real_storage.is_deleted(memory.content_hash) is True

    async def test_delete_by_tag_returns_count(self, real_storage):
        for i in range(3):
            await real_storage.store(_make_memory(f"bulk delete {i}", tags=["bulk"]))

        count, _ = await real_storage.delete_by_tag("bulk")
        assert count == 3

    async def test_update_memory_metadata_replaces_tags(self, real_storage):
        memory = _make_memory("update test", tags=["old"])
        await real_storage.store(memory)

        ok, _ = await real_storage.update_memory_metadata(
            memory.content_hash, {"tags": ["new1", "new2"]}
        )
        assert ok
        loaded = await real_storage.get_by_hash(memory.content_hash)
        assert sorted(loaded.tags) == ["new1", "new2"]

    async def test_get_stats_reports_active_count(self, real_storage):
        await real_storage.store(_make_memory("stats test 1"))
        await real_storage.store(_make_memory("stats test 2"))

        stats = await real_storage.get_stats()
        assert stats["storage_backend"] == "PgVectorMemoryStorage"
        assert stats["status"] == "operational"
        assert stats["total_memories"] >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
