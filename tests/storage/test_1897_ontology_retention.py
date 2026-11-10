"""Retention per type in the ontology overlay (#1897, phase 4).

A base type added through the overlay has no key in ``retention_periods``, so
its memories decayed on the 30-day fallback. ``set_retention`` stores a period
next to the type; the decay calculator reads it for types the consolidation
config does not name.
"""

import sqlite3
from datetime import datetime, timedelta

import pytest

from mcp_memory_service.consolidation.base import ConsolidationConfig
from mcp_memory_service.consolidation.decay import ExponentialDecayCalculator
from mcp_memory_service.models import ontology
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.ontology_overlay import upsert_retention
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


@pytest.fixture
async def storage(temp_db_path, monkeypatch):
    monkeypatch.delenv("MCP_CUSTOM_MEMORY_TYPES", raising=False)
    monkeypatch.setattr(ontology, "_OVERLAY_POLL_SECONDS", 0)
    s = SqliteVecMemoryStorage(f"{temp_db_path}/test_1897_retention.db")
    await s.initialize()
    await s.add_memory_type("legal")
    await s.add_memory_type("contract", parent="legal")
    yield s
    await s.close()
    ontology.clear_ontology_caches()


async def _retention_period(memory_type):
    now = datetime.now()
    created = now - timedelta(days=100)
    memory = Memory(
        content=f"{memory_type} memory",
        content_hash=f"hash-{memory_type}",
        tags=[],
        memory_type=memory_type,
        created_at=created.timestamp(),
        created_at_iso=created.isoformat() + "Z",
    )
    calc = ExponentialDecayCalculator(ConsolidationConfig())
    score = await calc._calculate_memory_relevance(memory, now, {}, {})
    return score.metadata["retention_period"]


async def test_overlay_base_type_and_its_subtypes_use_the_stored_period(storage):
    assert await _retention_period("legal") == 30  # the fallback this fixes

    assert await storage.set_retention("Legal", 400) == "legal"

    assert await _retention_period("legal") == 400
    assert await _retention_period("contract") == 400  # inherited from the parent


async def test_a_subtype_entry_overrides_its_parent_and_clear_falls_back(storage):
    await storage.set_retention("legal", 400)
    await storage.set_retention("contract", 60)
    await storage.set_retention("insight", 90)  # built-in subtype, no config key

    assert await _retention_period("contract") == 60
    assert await _retention_period("insight") == 90

    assert await storage.clear_retention("contract")
    assert not await storage.clear_retention("contract")
    assert await _retention_period("contract") == 400


async def test_removed_types_keep_decaying_on_their_stored_parent(storage):
    """Greptile on #1939: removing 'legal' left 'contract' memories without a
    parent, so they fell back to 30 days although 'legal' still has 400."""
    await storage.set_retention("legal", 400)
    await storage.add_memory_type("postmortem", parent="learning")

    assert await storage.remove_memory_type("legal")
    assert await storage.remove_memory_type("postmortem")
    assert ontology.get_parent_type("contract") is None  # no new writes with it

    assert await _retention_period("contract") == 400
    assert await _retention_period("postmortem") == 180  # built-in parent's period


@pytest.mark.parametrize(
    "memory_type, days",
    [
        ("nope", 90),        # not a memory type
        ("decision", 90),    # the config already has a period for it
        ("temporary", 90),   # legacy config key
        ("legal", 0),
        ("legal", 3651),
        ("legal", True),
        ("legal", "90"),
    ],
)
async def test_rejects_entries_that_would_not_apply(storage, memory_type, days):
    with pytest.raises(ValueError):
        await storage.set_retention(memory_type, days)


async def test_write_by_another_process_is_picked_up(storage):
    other = sqlite3.connect(storage.db_path)
    try:
        upsert_retention(other, "legal", 250)
    finally:
        other.close()

    assert await _retention_period("legal") == 250


async def test_malformed_rows_are_skipped(storage):
    other = sqlite3.connect(storage.db_path)
    try:
        with other:
            other.executemany(
                "INSERT INTO ontology_entries (kind, key, payload_json, updated_at) "
                "VALUES ('retention', ?, ?, 0)",
                [("legal", "not json"), ("contract", '{"days": -5}'), ("insight", '{"days": 45}')],
            )
            other.execute("UPDATE metadata SET value = value + 1 WHERE key = 'ontology_version'")
    finally:
        other.close()

    assert ontology.get_retention_overlay() == {"insight": 45}


async def test_in_memory_database_retention_loads(monkeypatch):
    monkeypatch.delenv("MCP_CUSTOM_MEMORY_TYPES", raising=False)
    monkeypatch.setattr(ontology, "_OVERLAY_POLL_SECONDS", 0)
    s = SqliteVecMemoryStorage(":memory:")
    await s.initialize()
    try:
        await s.add_memory_type("legal")
        await s.set_retention("legal", 400)
        assert await _retention_period("legal") == 400
    finally:
        await s.close()
        ontology.clear_ontology_caches()
