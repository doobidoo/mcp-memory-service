"""Semantic dedup supersede mode on sqlite-vec (#1938).

Embeddings are deterministic vectors at a chosen cosine. The four update
pairs are the ones reproduced against v12.7.0; the cosine is the measured
MiniLM score, not a fresh model call.
"""

import json
import math

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.base import RecreateRefused
from mcp_memory_service.storage.semantic_dedup import SemanticDedupResult
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash
from tests.unit.dedup_fixtures import XERJ_UPDATE_PAIRS


def _axis(dimension: int) -> list[float]:
    vector = [0.0] * dimension
    vector[0] = 1.0
    return vector


def _at_cosine(cosine: float, dimension: int) -> list[float]:
    cosine = max(-1.0, min(1.0, float(cosine)))
    sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    vector = [0.0] * dimension
    vector[0] = cosine
    if dimension > 1:
        vector[1] = sine
    return vector


@pytest.fixture
async def storage(temp_db_path, monkeypatch):
    backend = SqliteVecMemoryStorage(f"{temp_db_path}/dedup_supersede.db")
    await backend.initialize()
    backend.semantic_dedup_enabled = True
    backend.semantic_dedup_threshold = 0.85
    backend.semantic_dedup_near_identical_threshold = 0.97
    backend.semantic_dedup_mode = "reject"
    vectors: dict[str, list[float]] = {}

    def _embed(content: str) -> list[float]:
        try:
            return vectors[content]
        except KeyError as exc:
            raise AssertionError(f"no embedding fixture for {content!r}") from exc

    monkeypatch.setattr(backend, "_generate_embedding", _embed)
    backend._vectors = vectors  # type: ignore[attr-defined]
    yield backend
    await backend.close()


def _remember(storage, content: str, cosine: float | None) -> Memory:
    dimension = storage.embedding_dimension
    if cosine is None:
        storage._vectors[content] = _axis(dimension)
    else:
        storage._vectors[content] = _at_cosine(cosine, dimension)
    return Memory(content=content, content_hash=generate_content_hash(content))


def _lineage(storage, content_hash: str):
    return storage.conn.execute(
        "SELECT superseded_by, superseded_at, metadata, parent_id, version "
        "FROM memories WHERE content_hash = ?",
        (content_hash,),
    ).fetchone()


@pytest.mark.asyncio
@pytest.mark.parametrize("old,new,cosine", XERJ_UPDATE_PAIRS)
async def test_reject_mode_drops_the_reproduced_updates(storage, old, new, cosine):
    first = _remember(storage, old, None)
    ok, message = await storage.store(first)
    assert ok, message

    update = _remember(storage, new, cosine)
    ok, message = await storage.store(update)
    assert ok is False
    assert isinstance(message, SemanticDedupResult)
    assert message.outcome == "deduplicated"
    assert message.matched_hash == first.content_hash
    assert "semantically similar" in message
    assert _lineage(storage, update.content_hash) is None
    assert _lineage(storage, first.content_hash)[0] in (None, "")


@pytest.mark.asyncio
@pytest.mark.parametrize("old,new,cosine", XERJ_UPDATE_PAIRS)
async def test_supersede_mode_replaces_the_reproduced_updates(storage, old, new, cosine):
    storage.semantic_dedup_mode = "supersede"
    first = _remember(storage, old, None)
    ok, message = await storage.store(first)
    assert ok, message

    update = _remember(storage, new, cosine)
    ok, message = await storage.store(update)
    assert ok is True, message
    assert isinstance(message, SemanticDedupResult)
    assert message.outcome == "superseded"
    assert "conflict" not in str(message).lower()
    assert message.matched_hash == first.content_hash
    assert str(message).startswith("Memory stored successfully")

    old_row = _lineage(storage, first.content_hash)
    new_row = _lineage(storage, update.content_hash)
    assert old_row[0] == update.content_hash
    assert old_row[1] is not None
    old_meta = json.loads(old_row[2])
    assert old_meta["superseded_by"] == update.content_hash
    assert old_meta["evolution_reason"] == "semantic-dedup-supersede"
    assert new_row[3] == first.content_hash
    assert new_row[4] == 2

    found = await storage.retrieve(new, n_results=5)
    hashes = {item.memory.content_hash for item in found}
    assert update.content_hash in hashes
    assert first.content_hash not in hashes


@pytest.mark.asyncio
async def test_supersede_mode_still_rejects_the_near_identical_band(storage):
    storage.semantic_dedup_mode = "supersede"
    first = _remember(storage, "The cache TTL is 30 seconds.", None)
    assert (await storage.store(first))[0]
    update = _remember(storage, "The cache TTL is about 30 seconds.", 0.99)
    ok, message = await storage.store(update)
    assert ok is False
    assert isinstance(message, SemanticDedupResult)
    assert message.outcome == "deduplicated"
    assert _lineage(storage, update.content_hash) is None
    assert _lineage(storage, first.content_hash)[0] in (None, "")


@pytest.mark.asyncio
async def test_normalised_identical_text_rejects_inside_the_update_band(storage):
    storage.semantic_dedup_mode = "supersede"
    old = "The Lark API listens on port 8443 in production."
    new = "the   lark api listens on port 8443 in production."
    first = _remember(storage, old, None)
    assert (await storage.store(first))[0]
    update = _remember(storage, new, 0.90)
    ok, message = await storage.store(update)
    assert ok is False
    assert message.outcome == "deduplicated"
    assert message.matched_hash == first.content_hash


@pytest.mark.asyncio
async def test_below_threshold_stores_both_without_superseding(storage):
    storage.semantic_dedup_mode = "supersede"
    first = _remember(storage, "The office kettle is blue.", None)
    assert (await storage.store(first))[0]
    other = _remember(storage, "Q3 budget was approved on Monday.", 0.50)
    ok, message = await storage.store(other)
    assert ok is True, message
    assert not isinstance(message, SemanticDedupResult)
    assert _lineage(storage, first.content_hash)[0] in (None, "")
    assert _lineage(storage, other.content_hash)[0] in (None, "")


@pytest.mark.asyncio
async def test_supersede_stays_inside_the_target_store(storage):
    storage.semantic_dedup_mode = "supersede"
    old = "Dana's desk is on the 7th floor."
    new = "Dana's desk moved to the 3rd floor."
    first = _remember(storage, old, None)
    assert (await storage.store(first, store="docs"))[0]
    update = _remember(storage, new, 0.90)
    ok, message = await storage.store(update, store="default")
    assert ok is True, message
    assert not isinstance(message, SemanticDedupResult)
    assert _lineage(storage, first.content_hash)[0] in (None, "")


@pytest.mark.asyncio
async def test_skip_semantic_dedup_does_not_supersede(storage):
    storage.semantic_dedup_mode = "supersede"
    old, new, cosine = XERJ_UPDATE_PAIRS[0]
    first = _remember(storage, old, None)
    assert (await storage.store(first))[0]
    update = _remember(storage, new, cosine)
    ok, message = await storage.store(update, skip_semantic_dedup=True)
    assert ok is True, message
    assert not isinstance(message, SemanticDedupResult)
    assert _lineage(storage, first.content_hash)[0] in (None, "")


@pytest.mark.asyncio
async def test_exact_duplicate_is_not_a_semantic_outcome(storage):
    storage.semantic_dedup_mode = "supersede"
    content = "A fact stored twice, byte for byte."
    first = _remember(storage, content, None)
    assert (await storage.store(first))[0]
    again = _remember(storage, content, None)
    ok, message = await storage.store(again)
    assert ok is False
    assert not isinstance(message, SemanticDedupResult)
    assert "exact match" in message


@pytest.mark.asyncio
async def test_failed_supersede_retry_is_not_a_refused_recreate(storage, monkeypatch):
    """A failed link rolls the insert back. The retry is not a deleted-memory restore."""
    monkeypatch.setenv("MCP_SYNC_EVENTLOG", "on")
    storage.semantic_dedup_mode = "supersede"
    old, new, cosine = XERJ_UPDATE_PAIRS[0]
    first = _remember(storage, old, None)
    assert (await storage.store(first))[0]
    update = _remember(storage, new, cosine)
    original = storage._link_semantic_supersede_row

    def fail_once(_old_hash, _new_hash, _store_name):
        storage._link_semantic_supersede_row = original
        return False

    storage._link_semantic_supersede_row = fail_once
    ok, message = await storage.store(update)

    assert ok is False
    assert not isinstance(message, RecreateRefused)
    assert storage.conn.execute(
        "SELECT deleted_at FROM memories WHERE content_hash = ?",
        (update.content_hash,),
    ).fetchone() is None
    events = storage.conn.execute(
        "SELECT op FROM sync_events WHERE content_hash = ?",
        (update.content_hash,),
    ).fetchall()
    assert events == []

    result = await MemoryService(storage=storage).store_memory(content=new)
    assert result["success"] is True, result
    assert "refused_recreate" not in result
    row = storage.conn.execute(
        "SELECT deleted_at FROM memories WHERE content_hash = ?",
        (update.content_hash,),
    ).fetchone()
    assert row is not None and row[0] is None


@pytest.mark.asyncio
async def test_service_reports_both_outcomes(storage):
    old, new, cosine = XERJ_UPDATE_PAIRS[2]
    _remember(storage, old, None)
    _remember(storage, new, cosine)
    service = MemoryService(storage=storage)

    storage.semantic_dedup_mode = "reject"
    rejected = await service.store_memory(content=old)
    assert rejected["success"] is True
    assert "dedup_outcome" not in rejected
    missed = await service.store_memory(content=new)
    assert missed["success"] is False
    assert missed["dedup_outcome"] == "deduplicated"
    assert missed["matched_hash"] == generate_content_hash(old)
    assert "semantically similar" in missed["error"]

    storage.semantic_dedup_mode = "supersede"
    # The rejected update was not stored, so the same pair can supersede now.
    replaced = await service.store_memory(content=new)
    assert replaced["success"] is True
    assert replaced["dedup_outcome"] == "superseded"
    assert replaced["matched_hash"] == generate_content_hash(old)
    assert replaced["memory"]["content_hash"] == generate_content_hash(new)
