"""Ontology overlay in the memory DB (#1897, phase 1).

Memory types added through the storage live in ``ontology_entries``, are merged
after the built-ins and ``MCP_CUSTOM_MEMORY_TYPES``, and reach every process
that opens the same database through the ``ontology_version`` counter.
"""

import sqlite3

import pytest

from mcp_memory_service.models import ontology
from mcp_memory_service.storage.ontology_overlay import upsert_memory_type
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


@pytest.fixture
async def storage(temp_db_path, monkeypatch):
    monkeypatch.delenv("MCP_CUSTOM_MEMORY_TYPES", raising=False)
    # Poll on every read, so a write by "another process" shows without sleeping.
    monkeypatch.setattr(ontology, "_OVERLAY_POLL_SECONDS", 0)
    s = SqliteVecMemoryStorage(f"{temp_db_path}/test_1897.db")
    await s.initialize()
    yield s
    await s.close()
    ontology.clear_ontology_caches()


async def test_added_types_validate_and_resolve_their_parent(storage):
    assert await storage.add_memory_type("Legal") == "legal"
    await storage.add_memory_type("contract", parent="legal")
    await storage.add_memory_type("postmortem", parent="error")

    assert ontology.validate_memory_type("legal")
    assert ontology.get_parent_type("contract") == "legal"
    assert ontology.get_parent_type("postmortem") == "error"
    assert "legal" in ontology.get_base_types()
    assert ontology.coerce_memory_type("contract") == "contract"


@pytest.mark.parametrize(
    "name, parent",
    [
        ("decision", None),       # built-in base type
        ("code_edit", "error"),   # built-in subtype, would re-parent it
        ("thing", "nope"),        # unknown parent
        ("thing", "code_edit"),   # parent is a subtype, not a base type
        ("1bad", None),           # invalid name
    ],
)
async def test_rejects_names_that_would_shadow_or_dangle(storage, name, parent):
    with pytest.raises(ValueError):
        await storage.add_memory_type(name, parent=parent)


async def test_write_by_another_process_is_picked_up(storage):
    assert not ontology.validate_memory_type("runbook")

    # A second connection stands in for the HTTP server writing the same file.
    other = sqlite3.connect(storage.db_path)
    try:
        upsert_memory_type(other, "runbook", None)
    finally:
        other.close()

    assert ontology.validate_memory_type("runbook")


async def test_overlay_cannot_reparent_a_builtin_even_if_written_directly(storage):
    other = sqlite3.connect(storage.db_path)
    try:
        upsert_memory_type(other, "code_edit", "error")
    finally:
        other.close()

    assert ontology.get_parent_type("code_edit") == "observation"


async def test_remove_tombstones_overlay_types_only(storage):
    await storage.add_memory_type("legal")
    await storage.add_memory_type("contract", parent="legal")

    assert await storage.remove_memory_type("legal")
    assert not ontology.validate_memory_type("legal")
    # Its subtype has no base type left, so it is dropped as well.
    assert not ontology.validate_memory_type("contract")
    assert not await storage.remove_memory_type("decision")
    assert ontology.validate_memory_type("decision")

    # Re-adding revives the tombstoned row.
    await storage.add_memory_type("legal")
    assert ontology.validate_memory_type("contract")


async def test_close_unregisters_the_overlay(storage):
    await storage.add_memory_type("legal")
    await storage.close()
    assert not ontology.validate_memory_type("legal")
