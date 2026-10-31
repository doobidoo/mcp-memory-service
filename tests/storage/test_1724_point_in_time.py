# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for issue #1724: point-in-time (as_of) queries and status-flagged history.

Scenario (all timestamps UTC):
- M1 "The project deadline is October 1st" created 2026-09-01
- M2 "The project deadline is November 1st" created 2026-10-01, supersedes M1 at 2026-10-15

Expected (knowledge-time semantics):
- as_of=2026-09-15 -> M1 visible (it was current then), M2 invisible (not yet created)
- as_of=2026-10-20 -> M2 visible, M1 hidden (superseded by then)
- no as_of         -> M2 visible, M1 hidden (unchanged default behaviour)
- include_superseded=true -> both visible; M1 flagged superseded (+superseded_by/+superseded_at),
  M2 flagged current
"""

import hashlib
from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest
import pytest_asyncio

from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.utils.time_parser import parse_as_of


T1 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc).timestamp()   # M1 created
T2 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc).timestamp()  # M2 created
T3 = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc).timestamp()  # M1 superseded by M2


def _ts(y, m, d, hh=12):
    return datetime(y, m, d, hh, 0, tzinfo=timezone.utc).timestamp()


def _make_memory(content: str, created_at: float) -> Memory:
    return Memory(
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        tags=["deadline"],
        created_at=created_at,
    )


@pytest_asyncio.fixture
async def storage(tmp_path):
    """SqliteVec storage with a stub embedding model (all vectors identical).

    The embedding-model init is no-op'd so the fixture is hermetic with and
    without the ML extras installed: otherwise initialize() loads a real
    384-dim SentenceTransformer in the ML CI job (vec table sized FLOAT[384],
    embedding_dimension overwritten to 384) while the stub below returns
    8-dim vectors — every store() then fails with a dimension mismatch and
    all tests fail. With the no-op, the table is FLOAT[8] everywhere.
    """
    db_path = tmp_path / "test_1724.db"
    s = SqliteVecMemoryStorage(str(db_path))
    s.enable_cache = False
    s.embedding_dimension = 8  # must precede initialize(): the vec table is sized here

    async def _noop_model_init():
        return None

    s._initialize_embedding_model = _noop_model_init
    await s.initialize()
    mock_model = MagicMock()

    def _encode(texts, **kwargs):
        return np.ones((len(texts), 8), dtype=np.float32)

    mock_model.encode.side_effect = _encode
    s.embedding_model = mock_model
    try:
        yield s
    finally:
        await s.close()


@pytest_asyncio.fixture
async def superseded_pair(storage):
    """Store M1, then M2 which supersedes M1 at T3. Returns (m1_hash, m2_hash)."""
    m1 = _make_memory("The project deadline is October 1st", T1)
    m2 = _make_memory("The project deadline is November 1st", T2)
    # skip_semantic_dedup: the stub model returns identical vectors for every
    # text, which would otherwise trip the 0.85 dedup threshold.
    assert await storage.store(m1, skip_semantic_dedup=True)
    assert await storage.store(m2, skip_semantic_dedup=True)
    # Pin deterministic created_at (store() keeps the Memory's own timestamps,
    # but be explicit so the test does not depend on that).
    def _pin_times():
        storage.conn.execute(
            "UPDATE memories SET created_at = ? WHERE content_hash = ?", (T1, m1.content_hash)
        )
        storage.conn.execute(
            "UPDATE memories SET created_at = ? WHERE content_hash = ?", (T2, m2.content_hash)
        )
        storage.conn.commit()

    await storage._execute_with_retry(_pin_times)
    # Supersede M1 -> M2, backdating the supersede moment to T3.
    marked = await storage.mark_superseded_batch([(m2.content_hash, m1.content_hash)])
    assert marked == 1

    def _backdate_supersede():
        # Tolerate the pre-fix schema (no superseded_at column yet): the RED
        # phase must fail on the missing feature, not on fixture setup.
        try:
            storage.conn.execute(
                "UPDATE memories SET superseded_at = ? WHERE content_hash = ?",
                (T3, m1.content_hash),
            )
            storage.conn.commit()
        except Exception:
            storage.conn.rollback()

    await storage._execute_with_retry(_backdate_supersede)
    return m1.content_hash, m2.content_hash


def _hashes(result):
    return {m["content_hash"] for m in result["memories"]}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_as_of_returns_past_knowledge(storage, superseded_pair):
    """as_of between M1's creation and its supersede must show M1, not M2."""
    m1_hash, m2_hash = superseded_pair
    result = await storage.search_memories(query="project deadline", as_of="2026-09-15", limit=10)
    assert "error" not in result, result.get("error")
    hashes = _hashes(result)
    assert m1_hash in hashes, "M1 was current on 2026-09-15 and must be returned"
    assert m2_hash not in hashes, "M2 did not exist on 2026-09-15"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_as_of_after_supersede_returns_current(storage, superseded_pair):
    """as_of after the supersede must show M2 and hide M1 (default)."""
    m1_hash, m2_hash = superseded_pair
    result = await storage.search_memories(query="project deadline", as_of="2026-10-20", limit=10)
    assert "error" not in result, result.get("error")
    hashes = _hashes(result)
    assert m2_hash in hashes
    assert m1_hash not in hashes


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_as_of_behaviour_unchanged(storage, superseded_pair):
    """Without as_of, default search still hides superseded memories."""
    m1_hash, m2_hash = superseded_pair
    result = await storage.search_memories(query="project deadline", limit=10)
    assert "error" not in result, result.get("error")
    hashes = _hashes(result)
    assert m2_hash in hashes
    assert m1_hash not in hashes


@pytest.mark.unit
@pytest.mark.asyncio
async def test_include_superseded_carries_status_flags(storage, superseded_pair):
    """include_superseded=true must flag each result current/superseded."""
    m1_hash, m2_hash = superseded_pair
    result = await storage.search_memories(
        query="project deadline", include_superseded=True, limit=10
    )
    assert "error" not in result, result.get("error")
    by_hash = {m["content_hash"]: m for m in result["memories"]}
    assert m1_hash in by_hash and m2_hash in by_hash

    m1 = by_hash[m1_hash]
    assert m1.get("status") == "superseded", f"M1 must be flagged superseded, got {m1.get('status')}"
    assert m1.get("superseded_by") == m2_hash
    assert m1.get("superseded_at") is not None, "replacement time must be recorded"

    m2 = by_hash[m2_hash]
    assert m2.get("status") == "current", f"M2 must be flagged current, got {m2.get('status')}"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_as_of_with_include_superseded_shows_historical_status(storage, superseded_pair):
    """as_of + include_superseded: M1 was current at 2026-09-15, so flagged current there."""
    m1_hash, _ = superseded_pair
    result = await storage.search_memories(
        query="project deadline", as_of="2026-09-15", include_superseded=True, limit=10
    )
    assert "error" not in result, result.get("error")
    by_hash = {m["content_hash"]: m for m in result["memories"]}
    assert m1_hash in by_hash
    # At 2026-09-15 the supersede had not happened yet -> M1 was current then.
    assert by_hash[m1_hash].get("status") == "current"


@pytest.mark.unit
def test_parse_as_of_shorthands():
    """Year/month shorthands resolve to the last day of the period (Carrel-style).

    Bare dates resolve against the host's local day, matching parse_boundary;
    naive datetimes below therefore express the same host-local convention.
    """
    end_2024 = datetime(2024, 12, 31, 23, 59, 59).timestamp()
    assert parse_as_of("2024") == pytest.approx(end_2024, abs=2)

    end_feb_2024 = datetime(2024, 2, 29, 23, 59, 59).timestamp()
    assert parse_as_of("2024-02") == pytest.approx(end_feb_2024, abs=2)

    end_sep1 = datetime(2026, 9, 1, 23, 59, 59).timestamp()
    assert parse_as_of("2026-09-01") == pytest.approx(end_sep1, abs=2)

    # Shorthands order correctly: year > month > day within it.
    assert parse_as_of("2024") > parse_as_of("2024-06") > parse_as_of("2024-06-15")

    # Full ISO datetime with offset is an exact instant (tz-independent).
    assert parse_as_of("2026-09-15T12:00:00+00:00") == pytest.approx(
        datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc).timestamp(), abs=1
    )

    # Garbage raises ValueError.
    with pytest.raises(ValueError):
        parse_as_of("not-a-date")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mark_superseded_batch_records_superseded_at(storage):
    """mark_superseded_batch must write superseded_at in the same UPDATE."""
    m1 = _make_memory("old fact", T1)
    m2 = _make_memory("new fact", T2)
    await storage.store(m1, skip_semantic_dedup=True)
    await storage.store(m2, skip_semantic_dedup=True)
    before = datetime.now(timezone.utc).timestamp()
    marked = await storage.mark_superseded_batch([(m2.content_hash, m1.content_hash)])
    assert marked == 1

    def _read():
        cur = storage.conn.execute(
            "SELECT superseded_by, superseded_at FROM memories WHERE content_hash = ?",
            (m1.content_hash,),
        )
        return cur.fetchone()

    row = await storage._execute_with_retry(_read)
    assert row[0] == m2.content_hash
    assert row[1] is not None, "superseded_at must be written by mark_superseded_batch"
    assert row[1] >= before - 1, "superseded_at must be ~now"


# ---------------------------------------------------------------------------
# Issue #1724 (4): reasoning/temporal.py must not be dead code — the graph
# layer wires it up via get_temporal_edges() and find_connected(as_of=...).
# ---------------------------------------------------------------------------

T4 = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc).timestamp()  # later probe


@pytest.mark.asyncio
async def test_temporal_edges_wired_into_graph(graph_storage):
    """store_temporal_association -> get_temporal_edges(as_of) filters by validity."""
    from mcp_memory_service.reasoning.temporal import store_temporal_association

    h1, h2, h3 = "hash-aaa", "hash-bbb", "hash-ccc"
    # Edge A valid Sep-01 .. Oct-10 ; edge B valid from Oct-20 onward.
    oct10 = _ts(2026, 10, 10)
    oct20 = _ts(2026, 10, 20)
    assert await store_temporal_association(
        graph_storage, h1, h2, 0.9, ["temporal"], "causes",
        valid_from=T1, valid_until=oct10,
    )
    assert await store_temporal_association(
        graph_storage, h1, h3, 0.8, ["temporal"], "causes",
        valid_from=oct20,
    )

    # No as_of: both edges present.
    edges = await graph_storage.get_temporal_edges(h1)
    assert {e.target for e in edges} == {h2, h3}

    # as_of Sep-15: only A was valid then.
    edges = await graph_storage.get_temporal_edges(h1, as_of=_ts(2026, 9, 15))
    assert {e.target for e in edges} == {h2}

    # as_of Oct-25: only B was valid then.
    edges = await graph_storage.get_temporal_edges(h1, as_of=_ts(2026, 10, 25))
    assert {e.target for e in edges} == {h3}


@pytest.mark.asyncio
async def test_find_connected_respects_as_of(graph_storage):
    """find_connected(as_of=...) only traverses edges valid at as_of."""
    from mcp_memory_service.reasoning.temporal import store_temporal_association

    h1, h2, h3 = "hash-ddd", "hash-eee", "hash-fff"
    oct10 = _ts(2026, 10, 10)
    oct20 = _ts(2026, 10, 20)
    assert await store_temporal_association(
        graph_storage, h1, h2, 0.9, ["temporal"], "causes",
        valid_from=T1, valid_until=oct10,
    )
    assert await store_temporal_association(
        graph_storage, h1, h3, 0.8, ["temporal"], "causes",
        valid_from=oct20,
    )

    connected = await graph_storage.find_connected(h1, as_of=_ts(2026, 9, 15))
    assert {h for h, _ in connected} == {h2}
    connected = await graph_storage.find_connected(h1, as_of=_ts(2026, 10, 25))
    assert {h for h, _ in connected} == {h3}
    # Untimed edges still pass: no as_of keeps old behaviour.
    connected = await graph_storage.find_connected(h1)
    assert {h for h, _ in connected} == {h2, h3}


# ---------------------------------------------------------------------------
# Review follow-up (filhocf, 2026-10-31): boundary cases, non-retrieve modes,
# superseded_at idempotency, and the RRF fusion path.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_as_of_future_equals_present(storage, superseded_pair):
    """as_of in the future must equal the present result set (review #4)."""
    m1_hash, m2_hash = superseded_pair
    present = await storage.search_memories(query="project deadline", limit=10)
    future = await storage.search_memories(
        query="project deadline", as_of="2030-01-01", limit=10
    )
    assert "error" not in future, future.get("error")
    assert _hashes(future) == _hashes(present)
    assert m2_hash in _hashes(future)
    assert m1_hash not in _hashes(future)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_as_of_before_oldest_memory_returns_empty(storage, superseded_pair):
    """as_of before the oldest memory must return empty, not an error (review #4)."""
    result = await storage.search_memories(
        query="project deadline", as_of="2020-01-01", limit=10
    )
    assert "error" not in result, result.get("error")
    assert result["memories"] == []
    assert result["total"] == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_as_of_exact_mode(storage, superseded_pair):
    """Exact mode must honour as_of (review #1): created/superseded bounds apply."""
    m1_hash, m2_hash = superseded_pair
    result = await storage.search_memories(
        query="deadline", mode="exact", as_of="2026-10-20", limit=10
    )
    assert "error" not in result, result.get("error")
    hashes = _hashes(result)
    assert m2_hash in hashes
    assert m1_hash not in hashes, "M1 was superseded 2026-10-15, before as_of"

    result = await storage.search_memories(
        query="deadline", mode="exact", as_of="2026-10-10", limit=10
    )
    assert "error" not in result, result.get("error")
    hashes = _hashes(result)
    assert m1_hash in hashes, "M1 was still current on 2026-10-10"
    assert m2_hash in hashes


@pytest.mark.unit
@pytest.mark.asyncio
async def test_as_of_tag_only_mode(storage, superseded_pair):
    """Tag-only path must honour as_of (review #1)."""
    m1_hash, m2_hash = superseded_pair
    result = await storage.search_memories(
        tags=["deadline"], as_of="2026-10-20", limit=10
    )
    assert "error" not in result, result.get("error")
    hashes = _hashes(result)
    assert m2_hash in hashes
    assert m1_hash not in hashes, "M1 was superseded before as_of"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_exact_mode_status_flags(storage, superseded_pair):
    """Exact mode with include_superseded must label lineage (review #1)."""
    m1_hash, m2_hash = superseded_pair
    result = await storage.search_memories(
        query="deadline", mode="exact", include_superseded=True, limit=10
    )
    assert "error" not in result, result.get("error")
    by_hash = {m["content_hash"]: m for m in result["memories"]}
    assert m1_hash in by_hash and m2_hash in by_hash
    assert by_hash[m1_hash].get("status") == "superseded"
    assert by_hash[m1_hash].get("superseded_by") == m2_hash
    assert by_hash[m2_hash].get("status") == "current"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_superseded_at_not_restamped_on_retry(storage, superseded_pair):
    """Retried supersede operations must keep the original superseded_at (review #2)."""

    async def _read_sup_at(content_hash):
        def _q():
            row = storage.conn.execute(
                "SELECT superseded_at FROM memories WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
            return row[0] if row else None

        return await storage._execute_with_retry(_q)

    m1_hash, m2_hash = superseded_pair
    assert await _read_sup_at(m1_hash) == T3

    # Retry the same batch: the timestamp must not move.
    assert await storage.mark_superseded_batch([(m2_hash, m1_hash)]) == 1
    assert await _read_sup_at(m1_hash) == T3, "batch retry must preserve superseded_at"

    # Explicit metadata update with the same winner: preserved too.
    ok, _ = await storage.update_memory_metadata(m1_hash, {"superseded_by": m2_hash})
    assert ok
    assert await _read_sup_at(m1_hash) == T3, "metadata re-point must preserve superseded_at"

    # Clearing the supersession clears the timestamp.
    ok, _ = await storage.update_memory_metadata(m1_hash, {"superseded_by": None})
    assert ok
    assert await _read_sup_at(m1_hash) is None

    # Re-superseding after a clear stamps a fresh (current) time.
    ok, _ = await storage.update_memory_metadata(m1_hash, {"superseded_by": m2_hash})
    assert ok
    new_ts = await _read_sup_at(m1_hash)
    assert new_ts is not None and new_ts > T3


@pytest_asyncio.fixture
async def rrf_corpus(storage):
    """40-memory corpus for the RRF test: superseded pair + future memory + fillers."""
    t_fill = _ts(2026, 9, 15)
    t_future = _ts(2026, 10, 25)  # M3 is created after as_of
    m1 = _make_memory("The project deadline is October 1st", T1)
    m2 = _make_memory("The project deadline is November 1st", T2)
    m3 = _make_memory("The project deadline moved to December", t_future)
    assert await storage.store(m1, skip_semantic_dedup=True)
    assert await storage.store(m2, skip_semantic_dedup=True)
    assert await storage.store(m3, skip_semantic_dedup=True)
    for i in range(37):
        f = _make_memory(f"filler note {i} about project deadline logistics", t_fill)
        assert await storage.store(f, skip_semantic_dedup=True)
    assert await storage.mark_superseded_batch([(m2.content_hash, m1.content_hash)]) == 1

    def _backdate():
        storage.conn.execute(
            "UPDATE memories SET superseded_at = ? WHERE content_hash = ?",
            (T3, m1.content_hash),
        )
        storage.conn.commit()

    await storage._execute_with_retry(_backdate)
    return m1.content_hash, m2.content_hash, m3.content_hash


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rrf_fusion_honors_as_of(storage, rrf_corpus, monkeypatch):
    """RRF fusion path must apply the as_of bound at 40 candidates (review #3)."""
    m1_hash, m2_hash, m3_hash = rrf_corpus
    monkeypatch.setattr("mcp_memory_service.config.MCP_HYBRID_FUSION_METHOD", "rrf")

    result = await storage.search_memories(
        query="project deadline", mode="hybrid", as_of="2026-10-20", limit=50
    )
    assert "error" not in result, result.get("error")
    hashes = _hashes(result)
    assert len(hashes) == 38, f"expected 38 live-at-as_of memories, got {len(hashes)}"
    assert m1_hash not in hashes, "M1 was superseded 2026-10-15, before as_of"
    assert m2_hash in hashes, "M2 was current on 2026-10-20"
    assert m3_hash not in hashes, "M3 was created after as_of"

    # Sanity: without as_of the present-day path still hides M1 and shows M3.
    result_now = await storage.search_memories(
        query="project deadline", mode="hybrid", limit=50
    )
    assert "error" not in result_now, result_now.get("error")
    hashes_now = _hashes(result_now)
    assert m1_hash not in hashes_now
    assert m2_hash in hashes_now
    assert m3_hash in hashes_now
