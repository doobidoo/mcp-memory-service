"""MCP_CONSOLIDATION_AUTO_SUPERSEDE controls whether consolidation hides the
older memory of a pair relationship inference labels ``contradicts``.

With it off, the typed ``contradicts`` edge is still written to the graph;
only the supersession (which drops the older memory out of default
retrieval) is skipped. Runs on real SqliteVecMemoryStorage + GraphStorage;
only relationship inference is stubbed, so the test does not depend on the
heuristic classifying a particular pair of sentences.
"""

import importlib
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

import mcp_memory_service.config as config_package
import mcp_memory_service.config.graph as graph_config
import mcp_memory_service.consolidation.consolidator as consolidator_module
from mcp_memory_service.consolidation.base import ConsolidationConfig, MemoryAssociation
from mcp_memory_service.consolidation.consolidator import DreamInspiredConsolidator
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.graph import GraphStorage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash

_MOD = "mcp_memory_service.consolidation.consolidator"


async def _setup(temp_db_path, unique_content):
    db_path = os.path.join(temp_db_path, "test.db")
    storage = SqliteVecMemoryStorage(db_path)
    await storage.initialize()
    memories = []
    for base, ts in (("backups run nightly at two", 1735689600.0), ("backups now run nightly at three", 1746057600.0)):
        content = unique_content(base)
        m = Memory(content=content, content_hash=generate_content_hash(content), tags=["__test__"], created_at=ts)
        ok, msg = await storage.store(m)
        assert ok, msg
        memories.append(m)

    consolidator = DreamInspiredConsolidator(storage, ConsolidationConfig())
    consolidator.graph_storage = GraphStorage(db_path)
    consolidator.relationship_inference.infer_relationship_type = AsyncMock(return_value=("contradicts", 0.9))
    older, newer = memories
    association = MemoryAssociation(
        source_memory_hashes=[older.content_hash, newer.content_hash],
        similarity_score=0.6,
        connection_type="semantic",
        discovery_method="test",
        discovery_date=datetime.now(timezone.utc),
    )
    return storage, consolidator, older, newer, association


async def _visible(storage, content_hash):
    found = await storage.search_memories(query="backups nightly", limit=10)
    return content_hash in {m["content_hash"] for m in found["memories"]}


@pytest.mark.asyncio
@patch(f"{_MOD}.CONSOLIDATION_AUTO_SUPERSEDE", True)
async def test_default_supersedes_older_memory(temp_db_path, unique_content, monkeypatch):
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    storage, consolidator, older, newer, association = await _setup(temp_db_path, unique_content)
    try:
        await consolidator._store_associations_in_graph_table([association])
        assert not await _visible(storage, older.content_hash)
        assert await _visible(storage, newer.content_hash)
    finally:
        await consolidator.graph_storage.close()
        await storage.close()


@pytest.mark.asyncio
@patch(f"{_MOD}.CONSOLIDATION_AUTO_SUPERSEDE", False)
async def test_disabled_keeps_edge_and_both_memories(temp_db_path, unique_content, monkeypatch):
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    storage, consolidator, older, newer, association = await _setup(temp_db_path, unique_content)
    try:
        await consolidator._store_associations_in_graph_table([association])

        assert await _visible(storage, older.content_hash)
        assert await _visible(storage, newer.content_hash)
        rel_types = await consolidator.graph_storage.get_relationship_types(older.content_hash)
        assert rel_types.get("contradicts", 0) >= 1
    finally:
        await consolidator.graph_storage.close()
        await storage.close()


@pytest.fixture
def set_flag_from_environment(monkeypatch):
    """Set MCP_CONSOLIDATION_AUTO_SUPERSEDE in the environment and re-import the
    modules that read it, in import order: config.graph, the config package
    (which re-exports it), then the consolidator (which imports it by value)."""

    def _apply(value):
        if value is None:
            monkeypatch.delenv("MCP_CONSOLIDATION_AUTO_SUPERSEDE", raising=False)
        else:
            monkeypatch.setenv("MCP_CONSOLIDATION_AUTO_SUPERSEDE", value)
        importlib.reload(graph_config)
        importlib.reload(config_package)
        importlib.reload(consolidator_module)

    yield _apply
    monkeypatch.delenv("MCP_CONSOLIDATION_AUTO_SUPERSEDE", raising=False)
    importlib.reload(graph_config)
    importlib.reload(config_package)
    importlib.reload(consolidator_module)


@pytest.mark.asyncio
async def test_environment_false_reaches_consolidation(
    temp_db_path, unique_content, monkeypatch, set_flag_from_environment
):
    """No patching of the constant: the environment variable alone must keep
    both memories visible (the consolidator imports the setting by value)."""
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    set_flag_from_environment("false")
    storage, consolidator, older, newer, association = await _setup(temp_db_path, unique_content)
    try:
        await consolidator._store_associations_in_graph_table([association])
        assert await _visible(storage, older.content_hash)
        assert await _visible(storage, newer.content_hash)
    finally:
        await consolidator.graph_storage.close()
        await storage.close()


@pytest.mark.asyncio
async def test_environment_unset_still_supersedes(
    temp_db_path, unique_content, monkeypatch, set_flag_from_environment
):
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    set_flag_from_environment(None)
    storage, consolidator, older, newer, association = await _setup(temp_db_path, unique_content)
    try:
        await consolidator._store_associations_in_graph_table([association])
        assert not await _visible(storage, older.content_hash)
        assert await _visible(storage, newer.content_hash)
    finally:
        await consolidator.graph_storage.close()
        await storage.close()


def test_flag_reads_environment(monkeypatch):
    monkeypatch.setenv("MCP_CONSOLIDATION_AUTO_SUPERSEDE", "false")
    assert importlib.reload(graph_config).CONSOLIDATION_AUTO_SUPERSEDE is False
    monkeypatch.setenv("MCP_CONSOLIDATION_AUTO_SUPERSEDE", "true")
    assert importlib.reload(graph_config).CONSOLIDATION_AUTO_SUPERSEDE is True
    monkeypatch.delenv("MCP_CONSOLIDATION_AUTO_SUPERSEDE")
    assert importlib.reload(graph_config).CONSOLIDATION_AUTO_SUPERSEDE is True
