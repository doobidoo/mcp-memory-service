"""Tests for the SQLite-vec export in scripts/migration/migrate_to_cloudflare.py.

The export looped on get_recent_memories(100), which always returns the newest
100, so a database with more than 100 memories exported that first batch
repeatedly and never reached the older ones.
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

import mcp_memory_service.storage.sqlite_vec as sqlite_vec

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "migration" / "migrate_to_cloudflare.py"


@pytest.fixture(scope="module")
def migrator():
    spec = importlib.util.spec_from_file_location("migrate_to_cloudflare", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DataMigrator()


class _FakeStorage:
    """Same paging semantics as SqliteVecMemoryStorage: newest first."""

    def __init__(self, memories):
        self.memories = memories

    async def initialize(self):
        pass

    async def get_stats(self):
        return {"total_memories": len(self.memories)}

    async def get_recent_memories(self, n=10):
        return self.memories[:n]

    async def get_all_memories(self, limit=None, offset=0, store="default", **_):
        assert store is None, "the export must not drop memories outside the default store"
        end = None if limit is None else offset + limit
        return self.memories[offset:end]


def _memory(i):
    return SimpleNamespace(
        content=f"m{i}", content_hash=f"h{i}", tags=[], memory_type=None, metadata={},
        created_at=float(i), created_at_iso="", updated_at=float(i), updated_at_iso="")


@pytest.mark.parametrize("count", [0, 99, 100, 250])
def test_export_returns_every_memory_once(migrator, monkeypatch, count):
    fake = _FakeStorage([_memory(i) for i in range(count)])
    monkeypatch.setattr(sqlite_vec, "SqliteVecMemoryStorage", lambda path: fake)

    exported = asyncio.run(migrator.export_from_sqlite_vec("unused.db"))

    hashes = [m["content_hash"] for m in exported]
    assert len(hashes) == count
    assert set(hashes) == {f"h{i}" for i in range(count)}
