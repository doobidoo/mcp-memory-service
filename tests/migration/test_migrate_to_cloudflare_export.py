"""Tests for the SQLite-vec export in scripts/migration/migrate_to_cloudflare.py.

The export looped on get_recent_memories(100), which always returns the newest
100, so a database with more than 100 memories exported that first batch
repeatedly and never reached the older ones. Paging must also not drift when
get_all_memories() drops an unreadable row, and must not stop silently when a
page comes back empty.
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
    """Same paging semantics as SqliteVecMemoryStorage: newest first.

    ``rows`` are the table rows; ``None`` stands for a row that _row_to_memory()
    cannot convert, which get_all_memories() drops from the page. ``fail_at`` makes
    the page at that offset come back empty, as get_all_memories() does on error.
    """

    def __init__(self, rows, fail_at=None):
        self.rows = rows
        self.fail_at = fail_at

    async def initialize(self):
        pass

    async def count_all_memories(self, store="default", **_):
        assert store is None, "the count must cover every store"
        return len(self.rows)

    async def get_recent_memories(self, n=10):
        return [r for r in self.rows[:n] if r is not None]

    async def get_all_memories(self, limit=None, offset=0, store="default", **_):
        assert store is None, "the export must not drop memories outside the default store"
        if offset == self.fail_at:
            return []
        end = None if limit is None else offset + limit
        return [r for r in self.rows[offset:end] if r is not None]


def _memory(i):
    return SimpleNamespace(
        content=f"m{i}", content_hash=f"h{i}", tags=[], memory_type=None, metadata={},
        created_at=float(i), created_at_iso="", updated_at=float(i), updated_at_iso="")


def _export(migrator, monkeypatch, fake):
    monkeypatch.setattr(sqlite_vec, "SqliteVecMemoryStorage", lambda path: fake)
    return asyncio.run(migrator.export_from_sqlite_vec("unused.db"))


@pytest.mark.parametrize("count", [0, 99, 100, 250])
def test_export_returns_every_memory_once(migrator, monkeypatch, count):
    exported = _export(migrator, monkeypatch, _FakeStorage([_memory(i) for i in range(count)]))

    hashes = [m["content_hash"] for m in exported]
    assert len(hashes) == count
    assert set(hashes) == {f"h{i}" for i in range(count)}


def test_unreadable_rows_do_not_shift_the_next_page(migrator, monkeypatch):
    # Rows 50 and 150 cannot be converted. Advancing the offset by what came back
    # would re-read rows 100 and 101 (and 200, 201) on the following pages.
    rows = [None if i in (50, 150) else _memory(i) for i in range(250)]
    exported = _export(migrator, monkeypatch, _FakeStorage(rows))

    hashes = [m["content_hash"] for m in exported]
    assert len(hashes) == len(set(hashes)) == 248
    assert set(hashes) == {f"h{i}" for i in range(250)} - {"h50", "h150"}


def test_empty_page_mid_export_fails_instead_of_stopping(migrator, monkeypatch):
    fake = _FakeStorage([_memory(i) for i in range(250)], fail_at=100)
    with pytest.raises(RuntimeError, match="offset 100 of 250"):
        _export(migrator, monkeypatch, fake)
