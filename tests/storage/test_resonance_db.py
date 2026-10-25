"""Tests for ResonanceDB scoring and the /api/memories/resonance-search endpoint (PR #1673)."""

import hashlib
import struct
from types import SimpleNamespace

import pytest
import pytest_asyncio
from fastapi import HTTPException

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.resonance_db import ResonanceDB, _embed_to_waveform, _phase_shift_score
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.web.api.memories import (
    ResonanceSearchRequest,
    _score_resonance_candidates,
    resonance_search,
)

DIM = 8
BASE = [0.9, -0.4, 0.3, -0.8, 0.1, 0.6, -0.2, 0.5]
NEAR = [0.8, -0.3, 0.35, -0.7, 0.05, 0.55, -0.25, 0.45]
OPPOSITE = [-v for v in BASE]


def _blob(values):
    """sqlite-vec stores raw little-endian float32 with no length prefix."""
    return struct.pack(f"<{len(values)}f", *values)


def test_phase_shift_is_zero_for_identical_waves():
    wave = _embed_to_waveform(BASE, DIM)
    assert _phase_shift_score(wave, wave) == 0.0
    near = _phase_shift_score(wave, _embed_to_waveform(NEAR, DIM))
    assert 0.0 < near < _phase_shift_score(wave, _embed_to_waveform(OPPOSITE, DIM)) <= 1.0


def test_exact_match_ranks_first_in_every_mode():
    db = ResonanceDB(dimension=DIM)
    db.add("exact", BASE)
    db.add("near", NEAR)
    db.add("opposite", OPPOSITE)
    for mode in ("resonance", "phase", "hybrid"):
        ranked = [r["content_hash"] for r in db.search(BASE, top_k=3, mode=mode)]
        assert ranked[0] == "exact", mode
        assert ranked[-1] == "opposite", mode


def test_candidates_decode_raw_float32_and_skip_wrong_dimension():
    rows = [("exact", _blob(BASE)), ("short", _blob(BASE[:4])), ("opposite", _blob(OPPOSITE))]
    results, scored = _score_resonance_candidates(rows, BASE, DIM, top_k=10, mode="hybrid")
    assert scored == 2
    assert [r["content_hash"] for r in results] == ["exact", "opposite"]


def test_top_k_survives_batching():
    # The best match sits after the first 1000-row batch; it must still be returned.
    rows = [(f"far{i}", _blob(OPPOSITE)) for i in range(1500)] + [("exact", _blob(BASE))]
    results, scored = _score_resonance_candidates(rows, BASE, DIM, top_k=1, mode="hybrid")
    assert scored == 1501
    assert results[0]["content_hash"] == "exact"


@pytest_asyncio.fixture
async def storage(tmp_path):
    s = SqliteVecMemoryStorage(str(tmp_path / "resonance.db"))
    await s.initialize()
    try:
        yield s
    finally:
        await s.close()


async def _put(storage, content, store="default"):
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    ok, msg = await storage.store(Memory(content=content, content_hash=content_hash), store=store)
    assert ok, msg
    return content_hash


@pytest.mark.asyncio
async def test_endpoint_scopes_store_hides_superseded_and_unwraps_hybrid(storage):
    current = await _put(storage, "The deploy target is the staging cluster in Frankfurt")
    replaced = await _put(storage, "The deploy target was the old cluster in Dublin")
    other_store = await _put(storage, "Unrelated note kept in a separate store", store="other")

    def _supersede():
        storage.conn.execute("UPDATE memories SET superseded_by = ? WHERE content_hash = ?", (current, replaced))
        storage.conn.commit()
    await storage._execute_with_retry(_supersede)

    request = ResonanceSearchRequest(query="deploy target cluster", top_k=10)
    # Hybrid storage exposes its SQLite backend as `primary`.
    response = await resonance_search(request, storage=SimpleNamespace(primary=storage), user=None)
    hashes = {r.content_hash for r in response.results}
    assert hashes == {current}
    assert response.total_stored == 1

    response_all = await resonance_search(
        ResonanceSearchRequest(query="deploy target cluster", store="all"), storage=storage, user=None
    )
    assert {r.content_hash for r in response_all.results} == {current, other_store}


@pytest.mark.asyncio
async def test_endpoint_rejects_backend_without_sqlite():
    with pytest.raises(HTTPException) as exc:
        await resonance_search(ResonanceSearchRequest(query="x"), storage=SimpleNamespace(), user=None)
    assert exc.value.status_code == 501
