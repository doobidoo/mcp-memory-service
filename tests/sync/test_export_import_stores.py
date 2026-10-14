"""The export/import round trip keeps every store and counts only what the storage accepted (#1520).

Before the fix, ``MemoryExporter`` read only the ``default`` store and ``MemoryImporter``
ignored the ``(success, message)`` that ``store()`` returns, so a round trip could lose
memories while reporting success.
"""
import hashlib
import json
import time
from pathlib import Path

import pytest_asyncio

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.sync.exporter import MemoryExporter
from mcp_memory_service.sync.importer import MemoryImporter


def _memory(content: str) -> Memory:
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return Memory(content=content, content_hash=content_hash, tags=["__test__"])


@pytest_asyncio.fixture
async def source(temp_db_path):
    storage = SqliteVecMemoryStorage(str(Path(temp_db_path) / "source.db"))
    await storage.initialize()
    try:
        yield storage
    finally:
        await storage.close()


@pytest_asyncio.fixture
async def target(temp_db_path):
    storage = SqliteVecMemoryStorage(str(Path(temp_db_path) / "target.db"))
    await storage.initialize()
    # The test environment turns semantic deduplication off (tests/conftest.py). Turn it on
    # for the importing database, with a threshold that makes every memory stored inside the
    # time window a duplicate of the one before it, whatever embedding backend this
    # environment has: the point is whether the importer bypasses the check, not how close
    # two sentences are for a given model.
    storage.semantic_dedup_enabled = True
    storage.semantic_dedup_threshold = -1.0
    try:
        yield storage
    finally:
        await storage.close()


async def test_round_trip_keeps_every_store_and_bypasses_semantic_dedup(source, target, temp_db_path):
    doc = _memory("Release checklist for the sync hub, kept in the docs store.")
    first = _memory("Restart the queue worker before the nightly export runs.")
    second = _memory("Restart the queue worker before the nightly export runs, then check the log.")

    ok, message = await source.store(doc, store="docs")
    assert ok, message
    ok, message = await source.store(first)
    assert ok, message
    ok, message = await source.store(second, skip_semantic_dedup=True)
    assert ok, message

    export_file = Path(temp_db_path) / "export.json"
    result = await MemoryExporter(source).export_to_json(export_file)
    assert result["exported_count"] == 3

    with open(export_file, encoding="utf-8") as f:
        records = json.load(f)["memories"]
    assert {r["content_hash"]: r["store"] for r in records} == {
        doc.content_hash: "docs",
        first.content_hash: "default",
        second.content_hash: "default",
    }

    stats = await MemoryImporter(target).import_from_json([export_file])
    assert stats["rejected"] == 0, stats["rejections"]
    assert stats["imported"] == 3

    docs_hashes = [m.content_hash for m in await target.get_all_memories(store="docs")]
    assert docs_hashes == [doc.content_hash]
    default_hashes = {m.content_hash for m in await target.get_all_memories(store="default")}
    assert default_hashes == {first.content_hash, second.content_hash}


async def test_import_counts_only_what_the_storage_accepted(target, temp_db_path):
    already_there = _memory("A memory the target database already holds.")
    ok, message = await target.store(already_there)
    assert ok, message
    doc = _memory("A memory that belongs in the docs store.")

    now = time.time()
    export_file = Path(temp_db_path) / "export.json"
    with open(export_file, "w", encoding="utf-8") as f:
        json.dump({
            "export_metadata": {"source_machine": "laptop"},
            "memories": [
                {
                    "content": memory.content, "content_hash": memory.content_hash,
                    "tags": ["__test__"], "created_at": now, "updated_at": now,
                    "memory_type": "note", "metadata": {}, "store": store,
                }
                for memory, store in ((already_there, "default"), (doc, "docs"))
            ],
        }, f)

    # deduplicate=False skips the importer's own hash check, so the duplicate reaches store(),
    # whose exact-hash check rejects it.
    stats = await MemoryImporter(target).import_from_json([export_file], deduplicate=False)

    assert stats["imported"] == 1
    assert stats["rejected"] == 1
    assert stats["rejections"][0]["content_hash"] == already_there.content_hash
    assert "Duplicate" in stats["rejections"][0]["reason"]
    assert stats["sources"]["laptop"]["rejected"] == 1
    assert [m.content_hash for m in await target.get_all_memories(store="docs")] == [doc.content_hash]
