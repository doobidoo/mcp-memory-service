"""MCP_DECAY_ENABLED=false must stop relevance/decay scores being written (#1354).

With decay off, the relevance phase is skipped and no ``relevance_score`` /
``decay_factor`` / ``connection_boost`` / ``access_boost`` metadata reaches
storage. Forgetting still needs scores to pick low-relevance candidates, so it
computes them for itself and uses them without persisting them.

Runs a real consolidation pass on SqliteVecMemoryStorage and asserts on what
is stored afterwards.
"""

import dataclasses
import importlib
import os
import time

import pytest

from mcp_memory_service.config import consolidation as config_mod
from mcp_memory_service.consolidation.base import ConsolidationConfig
from mcp_memory_service.consolidation.consolidator import DreamInspiredConsolidator
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash

DECAY_KEYS = (
    "relevance_score",
    "relevance_calculated_at",
    "decay_factor",
    "connection_boost",
    "access_boost",
)


async def _store(storage, unique_content, base, created_at):
    content = unique_content(base)
    memory = Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=["__test__"],
        memory_type="decision",
        created_at=created_at,
    )
    ok, msg = await storage.store(memory)
    assert ok, msg
    return memory.content_hash


async def _decay_keys_written(storage, content_hash):
    stored = await storage.get_by_hash(content_hash)
    assert stored is not None
    return [k for k in DECAY_KEYS if k in (stored.metadata or {})]


def _config(consolidation_config, decay_enabled, forgetting_enabled=False):
    return dataclasses.replace(
        consolidation_config,
        decay_enabled=decay_enabled,
        clustering_enabled=False,
        associations_enabled=False,
        compression_enabled=False,
        forgetting_enabled=forgetting_enabled,
    )


def _config_from_env(decay_env):
    """A ConsolidationConfig built the way server_impl.py and web/app.py build it.

    CONSOLIDATION_CONFIG reads MCP_DECAY_ENABLED at import time, so the module is
    reloaded under the variable and again afterwards to restore it. A private
    MonkeyPatch keeps the reload from undoing the ``storage`` fixture's own patches.
    """
    with pytest.MonkeyPatch.context() as env:
        env.setenv("MCP_DECAY_ENABLED", decay_env)
        try:
            reloaded = importlib.reload(config_mod)
            built = ConsolidationConfig(**reloaded.CONSOLIDATION_CONFIG)
        finally:
            env.undo()
            importlib.reload(config_mod)
    return dataclasses.replace(
        built,
        clustering_enabled=False,
        associations_enabled=False,
        compression_enabled=False,
        forgetting_enabled=False,
    )


@pytest.fixture
async def storage(temp_db_path, monkeypatch):
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    s = SqliteVecMemoryStorage(os.path.join(temp_db_path, "test.db"))
    await s.initialize()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_decay_enabled_writes_relevance_metadata(storage, unique_content, consolidation_config):
    """Control: with the flag on, a pass does write the decay fields."""
    content_hash = await _store(storage, unique_content, "nightly backup moved to three", time.time() - 3600)
    consolidator = DreamInspiredConsolidator(storage, _config(consolidation_config, decay_enabled=True))

    report = await consolidator.consolidate("daily")

    assert report.memories_processed == 1
    assert await _decay_keys_written(storage, content_hash) == list(DECAY_KEYS)


@pytest.mark.asyncio
async def test_decay_disabled_writes_no_relevance_metadata(storage, unique_content, consolidation_config):
    content_hash = await _store(storage, unique_content, "nightly backup moved to three", time.time() - 3600)
    consolidator = DreamInspiredConsolidator(storage, _config(consolidation_config, decay_enabled=False))

    report = await consolidator.consolidate("daily")

    assert report.memories_processed == 1
    assert await _decay_keys_written(storage, content_hash) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "decay_env, expected_keys", [("false", []), ("true", list(DECAY_KEYS))], ids=["env-false", "env-true"]
)
async def test_env_flag_reaches_the_consolidator(storage, unique_content, decay_env, expected_keys):
    """MCP_DECAY_ENABLED -> CONSOLIDATION_CONFIG -> ConsolidationConfig -> consolidate()."""
    content_hash = await _store(storage, unique_content, "nightly backup moved to three", time.time() - 3600)
    consolidator = DreamInspiredConsolidator(storage, _config_from_env(decay_env))

    await consolidator.consolidate("daily")

    assert await _decay_keys_written(storage, content_hash) == expected_keys


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "decay_enabled, expected_keys", [(False, []), (True, list(DECAY_KEYS))], ids=["decay-off", "decay-on"]
)
async def test_forgetting_scores_persist_only_with_decay(
    storage, unique_content, consolidation_config, decay_enabled, expected_keys
):
    """Forgetting always gets relevance scores; they are written back only with decay on."""
    # A pass with an empty horizon window returns before any phase runs.
    await _store(storage, unique_content, "router firmware updated", time.time() - 3600)
    stale_hash = await _store(storage, unique_content, "old router firmware note", time.time() - 400 * 86400)
    consolidator = DreamInspiredConsolidator(
        storage, _config(consolidation_config, decay_enabled=decay_enabled, forgetting_enabled=True)
    )
    seen_scores = []
    real_process = consolidator.forgetting_engine.process

    async def spy(memories, relevance_scores, **kwargs):
        seen_scores.extend(relevance_scores)
        return await real_process(memories, relevance_scores, **kwargs)

    consolidator.forgetting_engine.process = spy

    await consolidator.consolidate("monthly")

    assert stale_hash in {s.memory_hash for s in seen_scores}
    assert await _decay_keys_written(storage, stale_hash) == expected_keys
