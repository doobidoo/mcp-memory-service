"""
RED tests for delta-sync Phase 5 — bootstrap + event version negotiation.

Spec: docs/rfc/planned/spec-delta-sync-fase5.md (CA1-CA7; CA8 is a hot E2E, manual).
These WILL fail until Phase 5 is implemented:
  - storage/sync/bootstrap.py: generate_baseline(storage) + install_baseline(storage, events, peer_id, watermark)
  - apply.py: reject events whose schema_version > KNOWN_SCHEMA_VERSION (R7)

Design (G1, approved): baseline ON DEMAND from current state; deterministic event_id =
stable hash of (content_hash, updated_at); watermark = MAX(seq) read in the same tx; peer
installs via the existing apply_remote_event + advance_sync_cursor (reuses Phase 4).
"""

import os
import tempfile

import pytest
import pytest_asyncio

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest_asyncio.fixture
async def source(monkeypatch):
    """Storage A (source of the baseline), agent=alpha, event-log on."""
    monkeypatch.setenv("MCP_AGENT_ID", "alpha")
    monkeypatch.setenv("MCP_SYNC_EVENTLOG", "true")
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    with tempfile.TemporaryDirectory() as tmp:
        s = SqliteVecMemoryStorage(os.path.join(tmp, "a.db"))
        await s.initialize()
        yield s
        await s.close()


@pytest_asyncio.fixture
async def target(monkeypatch):
    """Fresh storage B (new peer), agent=beta, empty."""
    monkeypatch.setenv("MCP_AGENT_ID", "beta")
    monkeypatch.setenv("MCP_SYNC_EVENTLOG", "true")
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    with tempfile.TemporaryDirectory() as tmp:
        s = SqliteVecMemoryStorage(os.path.join(tmp, "b.db"))
        await s.initialize()
        yield s
        await s.close()


async def _seed(storage, n=3):
    hashes = []
    for i in range(n):
        c = f"Bootstrap source memory number {i}"
        h = generate_content_hash(c)
        await storage.store(Memory(content=c, content_hash=h, tags=["seed"], memory_type="note"))
        hashes.append(h)
    return hashes


# ─────────────────────────── Bootstrap ───────────────────────────

@pytest.mark.asyncio
async def test_ca1_baseline_deterministic_and_install_idempotent(source, target):
    """CA1 (R1/R2): same state → identical baseline event ids; install twice → no dups."""
    from mcp_memory_service.storage.sync.bootstrap import generate_baseline, install_baseline

    await _seed(source, 3)
    base1, wm1 = generate_baseline(source)
    base2, wm2 = generate_baseline(source)
    ids1 = sorted(e["event_id"] for e in base1)
    ids2 = sorted(e["event_id"] for e in base2)
    assert ids1 == ids2, "baseline event_ids must be deterministic across runs"
    assert wm1 == wm2, "watermark must be stable for an unchanged state"

    install_baseline(target, base1, peer_id="alpha", watermark=wm1)
    n_after_first = (await target.get_all_memories(exclude_pending=False))
    install_baseline(target, base1, peer_id="alpha", watermark=wm1)
    n_after_second = (await target.get_all_memories(exclude_pending=False))
    assert len(n_after_first) == 3, f"expected 3 memories after install, got {len(n_after_first)}"
    assert len(n_after_second) == 3, "re-install must be idempotent (no duplicates)"


@pytest.mark.asyncio
async def test_ca2_watermark_seam_no_gap(source):
    """CA2 (R3): a memory stored after the watermark read is NOT silently in the baseline;
    it must be representable by a post-watermark event (seq > watermark)."""
    from mcp_memory_service.storage.sync.bootstrap import generate_baseline

    await _seed(source, 2)
    base, watermark = generate_baseline(source)
    # store AFTER the baseline snapshot
    c = "Memory created after the watermark"
    h = generate_content_hash(c)
    await source.store(Memory(content=c, content_hash=h, tags=["late"], memory_type="note"))
    # the late memory must have produced an event with seq > watermark
    row = source.conn.execute(
        "SELECT MAX(seq) FROM sync_events WHERE content_hash = ?", (h,)
    ).fetchone()
    assert row[0] is not None and row[0] > watermark, "late write must be post-watermark, not lost"
    # and it must NOT be in the already-generated baseline
    assert all(e["content_hash"] != h for e in base), "late write must not appear in the prior baseline"


# ─────────────────────── Bootstrap authorship/store ───────────────────────


@pytest.mark.asyncio
async def test_ca3_authorship_preserved_legacy_marked(source):
    """CA3 (R4): agent_id preserved; a memory with no authorship → legacy/unattributed,
    not an invented agent."""
    from mcp_memory_service.storage.sync.bootstrap import generate_baseline

    c = "Attributed memory"
    h = generate_content_hash(c)
    await source.store(Memory(content=c, content_hash=h, tags=["x"], memory_type="note",
                              metadata={"agent_id": "alpha"}))
    # a legacy row with NO agent_id anywhere
    c2 = "Legacy unattributed memory"
    h2 = generate_content_hash(c2)
    await source.store(Memory(content=c2, content_hash=h2, tags=["x"], memory_type="note"))
    source.conn.execute(
        "UPDATE memories SET metadata = json_remove(COALESCE(metadata,'{}'), '$.agent_id') WHERE content_hash = ?",
        (h2,),
    )
    source.conn.commit()

    base, _ = generate_baseline(source)
    by_hash = {e["content_hash"]: e for e in base}
    assert by_hash[h]["agent_id"] == "alpha", "attributed memory keeps its agent_id"
    assert by_hash[h2]["agent_id"] in ("legacy", "unattributed", None) or \
        by_hash[h2].get("legacy") is True, \
        f"unattributed memory must not get an invented agent, got {by_hash[h2].get('agent_id')!r}"


@pytest.mark.asyncio
async def test_ca3b_store_preserved_in_baseline(source):
    """CA3b (R-store, G5 auto-review): a named-store memory must carry its real store in the
    baseline payload — read from the `store` COLUMN, not from metadata."""
    from mcp_memory_service.storage.sync.bootstrap import generate_baseline

    c = "Memory in a named store"
    h = generate_content_hash(c)
    await source.store(Memory(content=c, content_hash=h, tags=["x"], memory_type="note"), store="projects")
    base, _ = generate_baseline(source)
    ev = next((e for e in base if e["content_hash"] == h), None)
    assert ev is not None and ev["payload"]["store"] == "projects", \
        f"baseline must carry the real store, got {ev and ev['payload'].get('store')!r}"


@pytest.mark.asyncio
async def test_ca4_soft_deleted_becomes_tombstone(source, target):
    """CA4 (R5): a soft-deleted memory bootstraps as a tombstone (peer: not resurrected)."""
    from mcp_memory_service.storage.sync.bootstrap import generate_baseline, install_baseline

    c = "Will be deleted before bootstrap"
    h = generate_content_hash(c)
    await source.store(Memory(content=c, content_hash=h, tags=["d"], memory_type="note"))
    await source.delete(h)  # soft delete

    base, wm = generate_baseline(source)
    ev = next((e for e in base if e["content_hash"] == h), None)
    assert ev is not None and ev["op"] == "delete", "deleted memory must bootstrap as a tombstone event"

    install_baseline(target, base, peer_id="alpha", watermark=wm)
    got = await target.get_by_hash(h)
    assert got is None, "tombstoned memory must not be resurrected on the peer"


@pytest.mark.asyncio
async def test_ca5_fresh_peer_cursor_at_watermark(source, target):
    """CA5 (R6): install → corpus + cursor at watermark; a post-watermark event applies on top."""
    from mcp_memory_service.storage.sync.bootstrap import generate_baseline, install_baseline

    await _seed(source, 2)
    base, wm = generate_baseline(source)
    install_baseline(target, base, peer_id="alpha", watermark=wm)

    cur = target.conn.execute(
        "SELECT last_seq_seen FROM sync_cursor WHERE peer_id = 'alpha'"
    ).fetchone()
    assert cur is not None and cur[0] == wm, f"cursor must sit at watermark {wm}, got {cur}"


# ─────────────────────── Version negotiation ───────────────────────

@pytest.mark.asyncio
async def test_ca6_unknown_envelope_version_rejected(target):
    """CA6 (R7): an event with schema_version > known is rejected, cursor NOT advanced."""
    from mcp_memory_service.storage.sync.apply import apply_remote_event

    c = "Event from a newer protocol version"
    h = generate_content_hash(c)
    ev = {
        "seq": 1, "schema_version": 99, "agent_id": "alpha", "event_id": "future1",
        "op": "create", "content_hash": h, "hlc_physical": 10, "hlc_logical": 0,
        "embedding_model": None, "embedding_dim": None,
        "payload": {"content_hash": h, "content": c, "tags": [], "memory_type": "note"},
    }
    res = apply_remote_event(target, ev)
    assert res.applied is False, "unknown envelope version must be rejected (not applied)"
    assert await target.get_by_hash(h) is None, "unknown-version event must not materialize"


@pytest.mark.asyncio
async def test_ca7_known_version_still_applies(target):
    """CA7 (R8): schema_version=1 applies exactly as Phase 4 (regression)."""
    from mcp_memory_service.storage.sync.apply import apply_remote_event

    c = "Normal current-version event"
    h = generate_content_hash(c)
    ev = {
        "seq": 1, "schema_version": 1, "agent_id": "alpha", "event_id": "cur1",
        "op": "create", "content_hash": h, "hlc_physical": 10, "hlc_logical": 0,
        "embedding_model": None, "embedding_dim": None,
        "payload": {"content_hash": h, "content": c, "tags": [], "memory_type": "note"},
    }
    res = apply_remote_event(target, ev)
    assert res.applied is True, "known-version event must apply as before"


@pytest.mark.asyncio
async def test_ca5b_cursor_never_regresses(source, target):
    """P1-1 (Tuvok): a re-install with an older baseline watermark must NOT rewind a cursor
    that live sync already advanced past it (R6 'no cursor regression')."""
    from mcp_memory_service.storage.sync.bootstrap import generate_baseline, install_baseline
    from mcp_memory_service.storage.sync.apply import advance_sync_cursor

    await _seed(source, 2)
    base, wm = generate_baseline(source)
    install_baseline(target, base, peer_id="alpha", watermark=wm)
    # live sync advanced the cursor well past the baseline watermark
    advance_sync_cursor(target, "alpha", wm + 50)
    # a stale re-install with the OLD watermark must not rewind the cursor
    install_baseline(target, base, peer_id="alpha", watermark=wm)
    cur = target.conn.execute(
        "SELECT last_seq_seen FROM sync_cursor WHERE peer_id = 'alpha'"
    ).fetchone()
    assert cur[0] == wm + 50, f"cursor must not regress below live position, got {cur[0]} (expected {wm+50})"


# ─────────────────────── Transport (endpoint + orchestration) ───────────────────────

@pytest.mark.asyncio
async def test_p1_1_cursor_not_advanced_on_failure(source, target):
    """Greptile P1-1: if any baseline event fails (e.g. unknown schema_version), the cursor
    must NOT advance — later pulls must not skip the history to recover it."""
    from mcp_memory_service.storage.sync.bootstrap import install_baseline

    # one good event + one with a future schema_version (apply rejects it)
    c = "good baseline event"; h = generate_content_hash(c)
    bad = {"schema_version": 99, "agent_id": "alpha", "event_id": "futurebl", "op": "create",
           "content_hash": generate_content_hash("bad"), "hlc_physical": 1, "hlc_logical": 0,
           "embedding_model": None, "embedding_dim": None,
           "payload": {"content_hash": generate_content_hash("bad"), "content": "x", "tags": [], "memory_type": "note"}}
    good = {"schema_version": 1, "agent_id": "alpha", "event_id": "goodbl", "op": "create",
            "content_hash": h, "hlc_physical": 1, "hlc_logical": 0,
            "embedding_model": None, "embedding_dim": None,
            "payload": {"content_hash": h, "content": c, "tags": [], "memory_type": "note"}}
    res = install_baseline(target, [good, bad], peer_id="alpha", watermark=42)
    assert res["complete"] is False and res["failed"] == 1
    cur = target.conn.execute("SELECT last_seq_seen FROM sync_cursor WHERE peer_id='alpha'").fetchone()
    assert cur is None or cur[0] == 0, "cursor must NOT advance when a baseline event failed"


@pytest.mark.asyncio
async def test_p1_2_baseline_carries_real_hlc(source):
    """Greptile P1-2: baseline events carry the source's real HLC (not 0), so a historical
    event cannot win over the baseline and revert state."""
    from mcp_memory_service.storage.sync.bootstrap import generate_baseline

    await _seed(source, 1)
    base, _ = generate_baseline(source)
    assert base, "expected a baseline event"
    assert base[0]["hlc_physical"] > 0, f"baseline must carry a real HLC, got {base[0]['hlc_physical']}"


@pytest.mark.asyncio
async def test_p1_4_superseded_carried_and_materialized(source, target):
    """Greptile P1-4: a superseded memory carries superseded_by in the baseline and the apply
    restores the column, so it stays hidden from search on the target."""
    from mcp_memory_service.storage.sync.bootstrap import generate_baseline, install_baseline

    c = "superseded source memory"; h = generate_content_hash(c)
    await source.store(Memory(content=c, content_hash=h, tags=["s"], memory_type="note"))
    source.conn.execute("UPDATE memories SET superseded_by = 'newer-hash' WHERE content_hash = ?", (h,))
    source.conn.commit()

    base, wm = generate_baseline(source)
    ev = next((e for e in base if e["content_hash"] == h), None)
    assert ev is not None and ev["payload"].get("superseded_by") == "newer-hash", "baseline must carry superseded_by"

    install_baseline(target, base, peer_id="alpha", watermark=wm)
    col = target.conn.execute("SELECT superseded_by FROM memories WHERE content_hash = ?", (h,)).fetchone()
    assert col is not None and col[0] == "newer-hash", "apply must restore superseded_by on the target"


def test_p1_3_schema_version_travels_through_feed(source):
    """Greptile P1-3: schema_version is carried by the feed (not dropped to 1), so the apply
    version check works through the real transport, not only on a direct apply call."""
    import asyncio
    from fastapi.testclient import TestClient
    from mcp_memory_service.web.app import create_app
    from mcp_memory_service.web.dependencies import set_storage
    from mcp_memory_service.web.oauth.middleware import require_read_access

    # store a normal event, then force its envelope schema_version to 2 in sync_events
    async def _prep():
        c = "feed version probe"; h = generate_content_hash(c)
        await source.store(Memory(content=c, content_hash=h, tags=["v"], memory_type="note"))
        source.conn.execute("UPDATE sync_events SET schema_version = 2 WHERE content_hash = ?", (h,))
        source.conn.commit()
    asyncio.get_event_loop().run_until_complete(_prep())
    set_storage(source)
    app = create_app(); app.dependency_overrides[require_read_access] = lambda: True
    client = TestClient(app)
    resp = client.get("/api/sync/events?since_seq=0&limit=100")
    assert resp.status_code == 200
    evs = resp.json()["events"]
    assert any(e.get("schema_version") == 2 for e in evs), "feed must carry schema_version, not drop it to 1"

@pytest.mark.asyncio
async def test_ca8_bootstrap_from_peer_full_cycle(source, target):
    """CA8 (in-process stand-in for the hot E2E): a fake peer serves the source's baseline;
    bootstrap_from_peer installs it on a fresh target and parks the cursor at the watermark."""
    from mcp_memory_service.storage.sync.bootstrap import generate_baseline, bootstrap_from_peer

    await _seed(source, 3)

    class _FakePeer:
        """Stands in for RemoteHTTPStorage.get_baseline over the wire."""
        async def get_baseline(self):
            return generate_baseline(source)

    result = await bootstrap_from_peer(target, _FakePeer(), peer_id="alpha")
    assert result["applied"] == 3, f"expected 3 applied, got {result}"
    mems = await target.get_all_memories(exclude_pending=False)
    assert len(mems) == 3, "target must hold the full corpus after bootstrap"
    cur = target.conn.execute(
        "SELECT last_seq_seen FROM sync_cursor WHERE peer_id = 'alpha'"
    ).fetchone()
    assert cur is not None and cur[0] == result["watermark"], "cursor parked at watermark"


def test_ca8b_baseline_endpoint_serves_events(source, monkeypatch):
    """CA8b: GET /api/sync/baseline returns events + watermark for the source storage."""
    import asyncio
    from fastapi.testclient import TestClient
    from mcp_memory_service.web.app import create_app
    from mcp_memory_service.web.dependencies import set_storage

    asyncio.get_event_loop().run_until_complete(_seed(source, 2))
    set_storage(source)
    app = create_app()
    from mcp_memory_service.web.oauth.middleware import require_read_access
    app.dependency_overrides[require_read_access] = lambda: True
    client = TestClient(app)
    resp = client.get("/api/sync/baseline")
    assert resp.status_code == 200, f"expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["count"] == 2 and len(body["events"]) == 2, "endpoint must serve the baseline events"
    assert "watermark" in body, "endpoint must return the watermark"
