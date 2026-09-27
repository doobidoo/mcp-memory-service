"""Quality model: separate computed_quality (machine) from user_rating (human) (#1312).

Before this change, a human rating was blended into `quality_score` as
`0.6*user + 0.4*old`, overwriting the machine's computed score in the same field
that retention/forgetting reads. Consequences reproduced here as RED-on-main:

1. rating a memory ERASES the machine `computed_quality` (no separate field kept);
2. `forgetting.py` reads the human-blended `quality_score`, so a thumbs-down can
   push a memory toward deletion — a rating is a search-ranking signal, not a
   keep/forget verdict (Henry, #1312);
3. rating normalization maps thumbs-down to 0.0 (floors the memory) instead of a
   de-ranking-not-flooring 0.25.

Design (Henry-approved on #1312): two origin fields in metadata
(`computed_quality` machine, `user_rating` human), the effective value
materialized into `quality_score`; human wins when present with -1/0/+1 mapped to
0.25/0.5/0.9, else computed, else 0.5; retention reads `computed_quality`.
"""

import os
import shutil
import tempfile

import pytest
import pytest_asyncio

from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest.fixture
def temp_storage_dir():
    d = tempfile.mkdtemp(prefix="mcp-test-qualmodel-")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest_asyncio.fixture
async def storage(temp_storage_dir):
    assert "mcp-test-" in temp_storage_dir
    db_path = os.path.join(temp_storage_dir, "test.db")
    _saved = {k: os.environ.get(k) for k in (
        "MCP_MEMORY_SQLITE_PATH", "MCP_MEMORY_STORAGE_BACKEND", "MCP_SEMANTIC_DEDUP_ENABLED"
    )}
    os.environ["MCP_MEMORY_SQLITE_PATH"] = db_path
    os.environ["MCP_MEMORY_STORAGE_BACKEND"] = "sqlite_vec"
    os.environ["MCP_SEMANTIC_DEDUP_ENABLED"] = "false"
    s = SqliteVecMemoryStorage(db_path)
    await s.initialize()
    yield s
    await s.close()
    # Restore the process env so a later test does not inherit a path to a
    # now-deleted temp database.
    for k, v in _saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# --- Pure composition function (the heart of the split) ---

def test_effective_quality_human_wins_with_normalization():
    """Human rating wins when present, mapped -1/0/+1 -> 0.25/0.5/0.9 (not 0/0.5/1)."""
    from mcp_memory_service.quality.config import effective_quality

    assert effective_quality(computed=0.8, user_rating=-1) == 0.25  # de-rank, not floor
    assert effective_quality(computed=0.8, user_rating=0) == 0.5
    assert effective_quality(computed=0.8, user_rating=1) == 0.9   # not overclaim 1.0


def test_effective_quality_falls_back_to_computed_then_default():
    """No rating -> computed; neither -> 0.5."""
    from mcp_memory_service.quality.config import effective_quality

    assert effective_quality(computed=0.72, user_rating=None) == 0.72
    assert effective_quality(computed=None, user_rating=None) == 0.5


# --- Storage behaviour ---

async def _store(storage, content, computed=None):
    m = Memory(content=content, content_hash=generate_content_hash(content), tags=["__test__"])
    if computed is not None:
        m.metadata["computed_quality"] = computed
        m.metadata["quality_score"] = computed
    ok, _ = await storage.store(m)
    assert ok
    return m.content_hash


def _meta(storage, h):
    import json
    cur = storage.conn.execute(
        "SELECT metadata FROM memories WHERE content_hash = ? AND deleted_at IS NULL", (h,)
    )
    row = cur.fetchone()
    return json.loads(row[0]) if row and row[0] else {}


@pytest.mark.asyncio
async def test_rating_preserves_computed_quality(storage):
    """Rating a memory must NOT erase the machine computed_quality."""
    from mcp_memory_service.server.handlers.quality import handle_rate_memory

    h = await _store(storage, "Backup runs at 02:00 nightly.", computed=0.8)

    class _Srv:
        async def _ensure_storage_initialized(self):
            return storage
    await handle_rate_memory(_Srv(), {"content_hash": h, "rating": -1})

    meta = _meta(storage, h)
    assert meta.get("computed_quality") == 0.8, "computed_quality must survive a human rating"
    assert meta.get("user_rating") == -1
    # effective materialized: human wins, -1 -> 0.25
    assert meta.get("quality_score") == 0.25


@pytest.mark.asyncio
async def test_forgetting_reads_computed_not_effective(storage):
    """A thumbs-down de-ranks search but must not change the retention band."""
    from mcp_memory_service.server.handlers.quality import handle_rate_memory

    h = await _store(storage, "High-value reference doc.", computed=0.9)

    class _Srv:
        async def _ensure_storage_initialized(self):
            return storage
    await handle_rate_memory(_Srv(), {"content_hash": h, "rating": -1})

    # After a down-vote: effective (search) dropped to 0.25, but retention must
    # still see the computed 0.9 (high-retention band), not 0.25 (aggressive archival).
    meta = _meta(storage, h)
    assert meta.get("quality_score") == 0.25   # search-facing
    assert meta.get("computed_quality") == 0.9  # retention-facing


@pytest.mark.asyncio
async def test_decay_relevance_ignores_human_downvote(storage):
    """Run the real decay/relevance engine: a down-vote must not lower relevance,
    because decay reads computed_quality, not the effective score (Greptile P1).

    This exercises the engine (not just stored metadata), so if the read ever
    reverts to memory.quality_score the down-vote would drop relevance and this
    test would fail.
    """
    from mcp_memory_service.consolidation.decay import ExponentialDecayCalculator
    from mcp_memory_service.consolidation.base import ConsolidationConfig
    from mcp_memory_service.server.handlers.quality import handle_rate_memory

    calc = ExponentialDecayCalculator(ConsolidationConfig())

    # Two identical high-computed memories; one will be down-voted.
    a = await _store(storage, "Relevance anchor memory alpha.", computed=0.9)
    b = await _store(storage, "Relevance anchor memory beta.", computed=0.9)

    ma = await storage.get_by_hash(a)
    mb = await storage.get_by_hash(b)
    rel_before = {r.memory_hash: r.total_score for r in await calc.process([ma, mb], connections={})}

    class _Srv:
        async def _ensure_storage_initialized(self):
            return storage
    await handle_rate_memory(_Srv(), {"content_hash": b, "rating": -1})

    ma2 = await storage.get_by_hash(a)
    mb2 = await storage.get_by_hash(b)
    rel_after = {r.memory_hash: r.total_score for r in await calc.process([ma2, mb2], connections={})}

    # The down-voted memory's relevance is unchanged (decay used computed_quality 0.9).
    assert rel_after[b] == pytest.approx(rel_before[b]), (
        "a human down-vote must not lower decay relevance — it reads computed_quality"
    )


@pytest.mark.asyncio
async def test_association_boost_does_not_overwrite_human_downvote(storage):
    """The consolidation association boost is a RETENTION signal, not a search score.

    A well-connected memory that a human down-voted must keep its effective
    quality_score at 0.25 after the boost runs — the boost must not resurrect the
    machine score into the search-facing field (Henry, #1349 review, decay.py).
    """
    from mcp_memory_service.consolidation.decay import ExponentialDecayCalculator
    from mcp_memory_service.consolidation.base import ConsolidationConfig
    from mcp_memory_service.server.handlers.quality import handle_rate_memory

    calc = ExponentialDecayCalculator(ConsolidationConfig())

    # computed 0.8, human thumbs-down -> effective quality_score 0.25.
    h = await _store(storage, "Well-connected but down-voted memory.", computed=0.8)

    class _Srv:
        async def _ensure_storage_initialized(self):
            return storage
    await handle_rate_memory(_Srv(), {"content_hash": h, "rating": -1})

    m = await storage.get_by_hash(h)
    assert m.metadata.get("quality_score") == 0.25  # precondition: down-voted

    # Enough connections (>= MCP_CONSOLIDATION_MIN_CONNECTIONS_FOR_BOOST=5) to fire
    # the association boost, then persist relevance metadata.
    scores = await calc.process([m], connections={h: 5})
    m = await calc.update_memory_relevance_metadata(m, scores[0])

    # The boost may raise retention (relevance_score) but must NOT overwrite the
    # human-effective search score.
    assert m.metadata.get("quality_score") == 0.25, (
        "association boost overwrote the human down-vote in quality_score"
    )
