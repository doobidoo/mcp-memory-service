"""
RED tests (Gate G3 / TDD) for the sync_events LOG COMPACTION job — the fix for the
unbounded sync_events growth where ~97% of rows are idempotent op='update_metadata'
recomputes.

Target under test (DOES NOT EXIST YET — collection fails with ImportError, valid RED):

    from mcp_memory_service.storage.sync.compaction import compact_update_metadata
    result = compact_update_metadata(storage, watermark=None)
    # -> {"events_removed": int, "bytes_reclaimed": int | None}

API / NAME DECISIONS (documented for G4, see subagent_response too):
  - Module: mcp_memory_service/storage/sync/compaction.py (sibling of apply.py/bootstrap.py,
    matches the package convention: resolver/apply/bootstrap/orchestrator/protocol).
  - Public fn: compact_update_metadata(storage, watermark=None) -> dict. Name kept from the
    task; it is precise (it compacts ONLY the update_metadata chains, never create/delete).
  - watermark semantics (T4): optional int. When None the function MUST compute a safe
    internal watermark = MIN(last_seq_pushed) across push_cursor (and never exceed the
    live-event frontier). An explicit watermark lets the caller cap how far compaction may
    touch. NO event with seq > watermark may be removed or rewritten.
  - Return keys: events_removed (int, count of sync_events rows deleted) and bytes_reclaimed
    (int | None; None allowed when the backend cannot cheaply measure, e.g. no VACUUM).

WHY EACH TEST IS RED TODAY: compaction.py does not exist, so the module import at the top
fails at COLLECTION time → every test in the file errors (ImportError). That is a legitimate
RED for a not-yet-written function. Each test ALSO specifies real post-compaction BEHAVIOR
(not just "import works") so it stays meaningful and turns GREEN only when G4 implements the
contract — and would turn RED again on a regression.

Reuses the REAL resolution (never re-implements LWW in the test):
  - apply._load_live_events + apply._resolve_md_fields  (field-level LWW winners)
  - apply._materialize_update_metadata                  (materialized row state)
  - bootstrap._read_snapshot                            (HLC pair election per hash)
  - store._recreate_of_for / recreate_of SELECT op='delete'  (delete dominance)
"""

import os
import json
import tempfile
import time

import pytest
import pytest_asyncio

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash
from mcp_memory_service.storage.sync.apply import (
    apply_remote_event,
    advance_sync_cursor,
    _load_live_events,
    _resolve_md_fields,
    _materialize_update_metadata,
    _sqlite,
)
from mcp_memory_service.storage.sync.bootstrap import _read_snapshot

# The module under test — DOES NOT EXIST YET. This import fails at collection → RED.
from mcp_memory_service.storage.sync.compaction import compact_update_metadata


# ─────────────────────────────── fixtures / helpers ───────────────────────────────

@pytest_asyncio.fixture
async def store(monkeypatch):
    monkeypatch.setenv("MCP_AGENT_ID", "beta")
    monkeypatch.setenv("MCP_SYNC_EVENTLOG", "true")
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    with tempfile.TemporaryDirectory() as tmp:
        s = SqliteVecMemoryStorage(os.path.join(tmp, "c.db"))
        await s.initialize()
        yield s
        await s.close()


async def _mk_store(name="x.db"):
    d = tempfile.mkdtemp()
    s = SqliteVecMemoryStorage(os.path.join(d, name))
    await s.initialize()
    return s


def _create_event(content_hash, *, content, tags=None, store="default",
                  created_at=1000.0, updated_at=1000.0, agent="alpha", event_id="c1",
                  hlc_physical=10, hlc_logical=0, memory_type="note", metadata=None):
    return {
        "agent_id": agent, "event_id": event_id, "op": "create",
        "content_hash": content_hash, "hlc_physical": hlc_physical, "hlc_logical": hlc_logical,
        "embedding_model": None, "embedding_dim": None,
        "payload": {
            "content_hash": content_hash, "content": content, "memory_type": memory_type,
            "tags": tags or [], "created_at": created_at, "updated_at": updated_at,
            "metadata": metadata or {}, "store": store,
        },
    }


def _update_metadata_event(content_hash, *, updates, event_id, agent="alpha",
                           hlc_physical, hlc_logical=0, updated_at=2000.0):
    return {
        "agent_id": agent, "event_id": event_id, "op": "update_metadata",
        "content_hash": content_hash, "hlc_physical": hlc_physical, "hlc_logical": hlc_logical,
        "embedding_model": None, "embedding_dim": None,
        "payload": {"content_hash": content_hash, "updates": updates, "updated_at": updated_at},
    }


def _delete_event(content_hash, *, event_id, agent="alpha", hlc_physical, hlc_logical=0,
                  deleted_at=1500.0):
    return {
        "agent_id": agent, "event_id": event_id, "op": "delete",
        "content_hash": content_hash, "hlc_physical": hlc_physical, "hlc_logical": hlc_logical,
        "embedding_model": None, "embedding_dim": None,
        "payload": {"content_hash": content_hash, "deleted_at": deleted_at},
    }


def _materialized_fields(s, content_hash):
    """Resolve the live incarnation via the REAL resolver (not a re-implementation)."""
    parsed, create_key = _load_live_events(s, content_hash)
    best, meta_best, create_md_keys, latest_updated_at = _resolve_md_fields(
        parsed, create_key, content_hash)
    return {
        "best": best,
        "meta_best": meta_best,
        "create_md_keys": sorted(create_md_keys),
        "latest_updated_at": latest_updated_at,
    }


def _row_state(s, content_hash):
    """Materialized memories-row snapshot (tags/type/metadata/updated_at/deleted_at)."""
    row = s.conn.execute(
        "SELECT tags, memory_type, metadata, updated_at, deleted_at "
        "FROM memories WHERE content_hash = ?", (content_hash,)
    ).fetchone()
    if row is None:
        return None
    return {
        "tags": set((row[0] or "").split(",")) - {""},
        "memory_type": row[1],
        "metadata": json.loads(row[2]) if row[2] else {},
        "updated_at": row[3],
        "deleted_at": row[4],
    }


def _count_update_metadata(s, content_hash):
    return s.conn.execute(
        "SELECT COUNT(*) FROM sync_events WHERE content_hash = ? AND op = 'update_metadata'",
        (content_hash,)
    ).fetchone()[0]


def _all_events(s, content_hash=None):
    q = ("SELECT seq, agent_id, event_id, op, content_hash, hlc_physical, hlc_logical, payload "
         "FROM sync_events")
    args = ()
    if content_hash is not None:
        q += " WHERE content_hash = ?"
        args = (content_hash,)
    q += " ORDER BY seq"
    return s.conn.execute(q, args).fetchall()


# ───────────────────────────────────── T1 ─────────────────────────────────────

@pytest.mark.asyncio
async def test_t1_lww_multifield_parity_via_real_resolver(store):
    """T1 — multi-field LWW parity. Different fields win in different update_metadata events
    (tags in A, memory_type in B, a custom metadata key in C). After compaction the LIVE
    incarnation resolved by the REAL resolver (_load_live_events + _resolve_md_fields) and the
    materialized memories row must be IDENTICAL to the pre-compaction state. We never
    re-implement LWW in the test — we compare the resolver's own output before vs after.

    RED: compact_update_metadata does not exist (collection ImportError). After G4 it must
    collapse the chain yet leave the resolved/materialized state byte-identical."""
    content = "multi-field LWW memory"
    h = generate_content_hash(content)
    apply_remote_event(store, _create_event(h, content=content, tags=["base"],
                                            memory_type="note", event_id="c1", hlc_physical=10))
    # A: tags wins here (hlc 20)
    apply_remote_event(store, _update_metadata_event(h, updates={"tags": ["base", "work"]},
                                                     event_id="A", hlc_physical=20))
    # B: memory_type wins here (hlc 30)
    apply_remote_event(store, _update_metadata_event(h, updates={"memory_type": "reference"},
                                                     event_id="B", hlc_physical=30))
    # C: a custom metadata key wins here (hlc 40)
    apply_remote_event(store, _update_metadata_event(
        h, updates={"metadata": {"project": "mir"}}, event_id="C", hlc_physical=40))

    s = _sqlite(store)
    before_resolved = _materialized_fields(s, h)
    before_row = _row_state(s, h)
    assert _count_update_metadata(s, h) == 3, "precondition: three update_metadata events"

    result = compact_update_metadata(store, watermark=None)
    assert isinstance(result, dict) and "events_removed" in result
    assert result["events_removed"] >= 1, "compaction must remove collapsed update_metadata rows"

    # Re-materialize from the (now compacted) log and compare against the resolver's own
    # pre-compaction verdict — identical means LWW parity held.
    _materialize_update_metadata(s, h, {"content_hash": h, "updates": {}})
    after_resolved = _materialized_fields(s, h)
    after_row = _row_state(s, h)

    assert after_resolved["best"] == before_resolved["best"], (
        f"LWW top-level fields changed: {before_resolved['best']} -> {after_resolved['best']}")
    assert after_resolved["meta_best"] == before_resolved["meta_best"], (
        f"LWW metadata winners changed: {before_resolved['meta_best']} -> {after_resolved['meta_best']}")
    assert after_row == before_row, f"materialized row diverged: {before_row} -> {after_row}"


# ───────────────────────────────────── T2 ─────────────────────────────────────

@pytest.mark.asyncio
async def test_t2_hlc_pair_does_not_regress(store):
    """T2 — the consolidated event must carry the HLC PAIR of the WINNING event (not MAX per
    column). bootstrap._read_snapshot must elect the SAME (hlc_physical, hlc_logical) pair per
    hash after compaction as before; and the per-column MAX used by
    _seed_last_hlc_on_first_boot must not shrink (no HLC regression).

    Crossed clocks: event with the larger physical has the smaller logical. A naive
    MAX-per-column consolidation would invent (2000,100); the correct consolidation keeps the
    winning event's real pair (2000,0).

    RED: compaction.py missing (collection ImportError). After G4 the elected pair must match."""
    content = "hlc pair memory"
    h = generate_content_hash(content)
    apply_remote_event(store, _create_event(h, content=content, event_id="c1", hlc_physical=10))
    # crossed clocks on the update_metadata chain
    apply_remote_event(store, _update_metadata_event(h, updates={"tags": ["lo"]},
                                                     event_id="lo", hlc_physical=1000, hlc_logical=100))
    apply_remote_event(store, _update_metadata_event(h, updates={"tags": ["hi"]},
                                                     event_id="hi", hlc_physical=2000, hlc_logical=0))
    s = _sqlite(store)

    rows_before, _, hlc_before = _read_snapshot(store)
    pair_before = hlc_before.get(h)
    max_phys_before = s.conn.execute("SELECT MAX(hlc_physical) FROM sync_events").fetchone()[0]
    max_log_before = s.conn.execute("SELECT MAX(hlc_logical) FROM sync_events").fetchone()[0]

    compact_update_metadata(store, watermark=None)

    rows_after, _, hlc_after = _read_snapshot(store)
    pair_after = hlc_after.get(h)
    max_phys_after = s.conn.execute("SELECT MAX(hlc_physical) FROM sync_events").fetchone()[0]
    max_log_after = s.conn.execute("SELECT MAX(hlc_logical) FROM sync_events").fetchone()[0]

    assert pair_after == pair_before, (
        f"bootstrap must elect the SAME HLC pair; before={pair_before} after={pair_after}")
    # The consolidated event must NOT raise the per-column MAX beyond what already existed
    # (no fabricated clock → _seed_last_hlc_on_first_boot cannot regress a future event).
    assert max_phys_after <= max_phys_before, "compaction must not inflate MAX(hlc_physical)"
    assert max_log_after <= max_log_before, "compaction must not inflate MAX(hlc_logical)"
    # And specifically: the winning pair is the real (2000,0), never the invented (2000,100).
    assert pair_after == (2000, 0), f"winning pair must be the real (2000,0), got {pair_after}"


# ───────────────────────────────────── T3 ─────────────────────────────────────

@pytest.mark.asyncio
async def test_t3_delete_events_untouched_recreate_of_stable(store):
    """T3 — op='delete' events are NEVER touched by compaction. The recreate_of SELECT
    (store.py: WHERE op='delete') must elect the SAME dominant delete after compaction.

    RED: compaction.py missing. After G4 delete rows and their count/identity must be intact."""
    content = "deletable memory"
    h = generate_content_hash(content)
    apply_remote_event(store, _create_event(h, content=content, event_id="c1", hlc_physical=10))
    apply_remote_event(store, _update_metadata_event(h, updates={"tags": ["x"]},
                                                     event_id="u1", hlc_physical=20))
    apply_remote_event(store, _delete_event(h, event_id="d1", hlc_physical=30))
    s = _sqlite(store)

    def _delete_rows():
        return s.conn.execute(
            "SELECT agent_id, event_id, hlc_physical, hlc_logical FROM sync_events "
            "WHERE content_hash = ? AND op = 'delete' ORDER BY event_id", (h,)
        ).fetchall()

    deletes_before = _delete_rows()
    # dominant delete the recreate_of lookup would pick (same query store.py uses)
    dom_before = s.conn.execute(
        "SELECT event_id FROM sync_events WHERE content_hash = ? AND op='delete' "
        "ORDER BY hlc_physical DESC, hlc_logical DESC LIMIT 1", (h,)).fetchone()

    compact_update_metadata(store, watermark=None)

    deletes_after = _delete_rows()
    dom_after = s.conn.execute(
        "SELECT event_id FROM sync_events WHERE content_hash = ? AND op='delete' "
        "ORDER BY hlc_physical DESC, hlc_logical DESC LIMIT 1", (h,)).fetchone()

    assert deletes_after == deletes_before, "delete events must be untouched by compaction"
    assert dom_after == dom_before, "dominant delete (recreate_of) must not change"


# ───────────────────────────────────── T4 ─────────────────────────────────────

@pytest.mark.asyncio
async def test_t4_watermark_respects_push_cursor_and_leaves_cursors(store):
    """T4 — no event with seq > MIN(last_seq_pushed) across peers may be removed/rewritten,
    and both sync_cursor and push_cursor must be byte-identical after compaction.

    Setup: a long update_metadata chain; push_cursor for two peers so MIN(last_seq_pushed)
    sits in the MIDDLE of the chain. Compaction may only collapse events at or below that
    frontier; every event above it must survive verbatim.

    RED: compaction.py missing. After G4 the frontier and cursors must be respected."""
    content = "watermark memory"
    h = generate_content_hash(content)
    apply_remote_event(store, _create_event(h, content=content, event_id="c1", hlc_physical=10))
    for i in range(2, 7):  # u2..u6
        apply_remote_event(store, _update_metadata_event(
            h, updates={"tags": [f"t{i}"]}, event_id=f"u{i}", hlc_physical=10 * i))
    s = _sqlite(store)

    # frontier = MIN(last_seq_pushed) across peers → set peers so it lands mid-chain
    all_seqs = [r[0] for r in _all_events(s, h)]
    mid_seq = all_seqs[len(all_seqs) // 2]
    from mcp_memory_service.storage.sync.orchestrator import advance_push_cursor
    advance_push_cursor(store, "peerA", mid_seq)
    advance_push_cursor(store, "peerB", mid_seq + 1)  # MIN is peerA's mid_seq
    advance_sync_cursor(store, "peerA", 3)

    push_before = s.conn.execute(
        "SELECT peer_id, last_seq_pushed FROM push_cursor ORDER BY peer_id").fetchall()
    sync_before = s.conn.execute(
        "SELECT peer_id, last_seq_seen FROM sync_cursor ORDER BY peer_id").fetchall()
    above_before = [r for r in _all_events(s, h) if r[0] > mid_seq]

    compact_update_metadata(store, watermark=None)

    push_after = s.conn.execute(
        "SELECT peer_id, last_seq_pushed FROM push_cursor ORDER BY peer_id").fetchall()
    sync_after = s.conn.execute(
        "SELECT peer_id, last_seq_seen FROM sync_cursor ORDER BY peer_id").fetchall()
    above_after = [r for r in _all_events(s, h) if r[0] > mid_seq]

    assert push_after == push_before, "push_cursor must be unchanged by compaction"
    assert sync_after == sync_before, "sync_cursor must be unchanged by compaction"
    # every event strictly above the min-pushed frontier must survive byte-identical
    assert {r[2] for r in above_after} == {r[2] for r in above_before}, (
        "events above MIN(last_seq_pushed) must not be removed")
    assert above_after == above_before, "events above the frontier must not be rewritten"


# ───────────────────────────────────── T5 ─────────────────────────────────────

@pytest.mark.asyncio
async def test_t5_recreation_boundary_not_merged_across_incarnations(store):
    """T5 — create-after-delete (two incarnations of the same hash). Compaction must NOT merge
    the DEAD incarnation's update_metadata fields into the LIVE one. We compare the resolver's
    live-incarnation verdict before vs after; the dead edit must stay excluded.

    RED: compaction.py missing. After G4 the recreation boundary must hold."""
    content = "recreated memory"
    h = generate_content_hash(content)
    # incarnation 1
    apply_remote_event(store, _create_event(h, content=content, tags=["v1"],
                                            memory_type="note", event_id="c1", hlc_physical=10))
    apply_remote_event(store, _update_metadata_event(h, updates={"memory_type": "DEAD"},
                                                     event_id="dead-edit", hlc_physical=15))
    apply_remote_event(store, _delete_event(h, event_id="d1", hlc_physical=20))
    # incarnation 2 (v1 bare create → revive)
    apply_remote_event(store, _create_event(h, content=content, tags=["v2"],
                                            memory_type="reference", event_id="c2", hlc_physical=30))
    apply_remote_event(store, _update_metadata_event(h, updates={"tags": ["v2", "fresh"]},
                                                     event_id="live-edit", hlc_physical=40))
    s = _sqlite(store)

    before = _materialized_fields(s, h)
    assert before["best"].get("memory_type") == "reference", (
        "precondition: live incarnation type is 'reference' (dead 'DEAD' excluded)")

    compact_update_metadata(store, watermark=None)

    after = _materialized_fields(s, h)
    assert after["best"].get("memory_type") == "reference", (
        f"dead incarnation's memory_type must NOT be merged into the live one, got {after['best']}")
    assert after["best"].get("tags") == before["best"].get("tags"), (
        f"live tags must be preserved across compaction: {before['best']} -> {after['best']}")
    assert "DEAD" not in json.dumps(after["best"]), "dead incarnation field must not resurface"


# ───────────────────────────────────── T6 ─────────────────────────────────────

@pytest.mark.asyncio
async def test_t6_idempotent_failclosed_deterministic_id(store):
    """T6 — (a) a 2nd immediate compaction is a no-op (events_removed == 0); (b) the
    consolidated event_id is DETERMINISTIC (same inputs → same id); (c) fail-closed: if the
    transaction fails mid-way the log is byte-IDENTICAL (no partial removal).

    RED: compaction.py missing. After G4 all three properties must hold."""
    content = "idempotent memory"
    h = generate_content_hash(content)
    apply_remote_event(store, _create_event(h, content=content, event_id="c1", hlc_physical=10))
    for i in range(2, 6):
        apply_remote_event(store, _update_metadata_event(
            h, updates={"tags": [f"t{i}"]}, event_id=f"u{i}", hlc_physical=10 * i))
    s = _sqlite(store)

    # (b) determinism: capture consolidated event_id of the first run
    first = compact_update_metadata(store, watermark=None)
    assert first["events_removed"] >= 1
    consolidated_1 = s.conn.execute(
        "SELECT event_id FROM sync_events WHERE content_hash = ? AND op = 'update_metadata' "
        "ORDER BY seq", (h,)).fetchall()

    # (a) idempotence: second immediate run removes nothing
    second = compact_update_metadata(store, watermark=None)
    assert second["events_removed"] == 0, (
        f"2nd compaction must be a no-op, removed {second['events_removed']}")
    consolidated_2 = s.conn.execute(
        "SELECT event_id FROM sync_events WHERE content_hash = ? AND op = 'update_metadata' "
        "ORDER BY seq", (h,)).fetchall()
    assert consolidated_2 == consolidated_1, "consolidated event_id must be deterministic/stable"

    # (c) fail-closed: force the commit to blow up mid-transaction and assert nothing changed.
    content2 = "failclosed memory"
    h2 = generate_content_hash(content2)
    apply_remote_event(store, _create_event(h2, content=content2, event_id="c1b", hlc_physical=10))
    for i in range(2, 6):
        apply_remote_event(store, _update_metadata_event(
            h2, updates={"tags": [f"q{i}"]}, event_id=f"v{i}", hlc_physical=10 * i))
    before_log = _all_events(s)

    # Inject a failure in the MIDDLE of the REAL _apply_plan transaction: patch json.dumps (used
    # to serialize the consolidated payload for the INSERT, which runs AFTER the DELETE) so the
    # real function performs its DELETE, then raises during the INSERT. This exercises the REAL
    # fail-closed path (the function's own SAVEPOINT/ROLLBACK), not a mock of it. We patch
    # json.dumps rather than sqlite3.Connection.commit, which is a read-only C attribute on
    # Python 3.11 and cannot be monkeypatched on a raw connection.
    import mcp_memory_service.storage.sync.compaction as _compaction_mod

    orig_dumps = _compaction_mod.json.dumps

    def _boom_dumps(*a, **k):
        raise RuntimeError("injected failure during consolidated INSERT serialization")

    _compaction_mod.json.dumps = _boom_dumps
    try:
        with pytest.raises(Exception):
            compact_update_metadata(store, watermark=None)
    finally:
        _compaction_mod.json.dumps = orig_dumps

    after_log = _all_events(s)
    assert after_log == before_log, (
        "fail-closed: a failed compaction must leave the event log byte-identical "
        "(no partial removal)")


# ───────────────────────────────────── T7 ─────────────────────────────────────

@pytest.mark.asyncio
async def test_t7_two_instances_converge_after_one_compacts(monkeypatch):
    """T7 — convergence (in-memory integration, two storage instances, no HTTP). Apply the
    SAME event set to peers A and B, compact ONLY A, then cross-apply each peer's current
    sync_events to the other and assert both converge to the IDENTICAL materialized state.

    RED: compaction.py missing. After G4 compaction must be convergence-preserving."""
    monkeypatch.setenv("MCP_AGENT_ID", "beta")
    monkeypatch.setenv("MCP_SYNC_EVENTLOG", "true")
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")

    content = "converging memory"
    h = generate_content_hash(content)
    create = _create_event(h, content=content, tags=["base"], memory_type="note",
                           event_id="c1", hlc_physical=10)
    edits = [
        _update_metadata_event(h, updates={"tags": ["base", "work"]}, event_id="e-tags",
                               hlc_physical=20),
        _update_metadata_event(h, updates={"memory_type": "reference"}, event_id="e-type",
                               hlc_physical=30),
        _update_metadata_event(h, updates={"metadata": {"project": "mir"}}, event_id="e-md",
                               hlc_physical=40),
    ]

    sa = await _mk_store("a.db")
    sb = await _mk_store("b.db")
    try:
        for ev in [create, *edits]:
            apply_remote_event(sa, dict(ev))
            apply_remote_event(sb, dict(ev))

        sa_sql, sb_sql = _sqlite(sa), _sqlite(sb)
        before_a = _row_state(sa_sql, h)
        before_b = _row_state(sb_sql, h)
        assert before_a == before_b, "precondition: peers identical before compaction"

        # Compact ONLY peer A.
        compact_update_metadata(sa, watermark=None)

        # Cross-apply each peer's CURRENT sync_events to the other (simulate a sync round
        # after A compacted — B pulls A's now-consolidated log and vice-versa).
        def _export(s):
            out = []
            for seq, agent, eid, op, ch, hp, hl, payload in _all_events(s, h):
                out.append({
                    "agent_id": agent, "event_id": eid, "op": op, "content_hash": ch,
                    "hlc_physical": hp, "hlc_logical": hl,
                    "embedding_model": None, "embedding_dim": None,
                    "payload": json.loads(payload) if payload else {},
                })
            return out

        for ev in _export(sa_sql):
            apply_remote_event(sb, ev)
        for ev in _export(sb_sql):
            apply_remote_event(sa, ev)

        after_a = _row_state(sa_sql, h)
        after_b = _row_state(sb_sql, h)
        assert after_a == after_b, (
            f"peers must converge after one compacts: A={after_a} B={after_b}")
        # and convergence must match the pre-compaction materialized state (no field lost)
        assert after_a["tags"] == before_a["tags"], "tags lost through compaction+sync"
        assert after_a["memory_type"] == before_a["memory_type"], "type lost through compaction+sync"
        assert after_a["metadata"].get("project") == "mir", "custom metadata lost through compaction+sync"
    finally:
        await sa.close()
        await sb.close()


# ───────────────────────────────────── T8 (G5 P2 hardening) ─────────────────────────────────────

@pytest.mark.asyncio
async def test_t8_insert_collision_fails_closed_no_silent_loss(store):
    """T8 — G5/Tuvok P2: a collision on the consolidated event's identity must NOT silently drop
    the row after the DELETE already ran (that was the OR IGNORE anti-pattern → silent data loss).
    With a raw INSERT, the collision raises and the single SAVEPOINT rolls the whole batch back:
    the event log stays byte-identical (fail-closed), nothing is lost.

    We seed a surviving decoy row carrying the SAME (agent_id='compaction', event_id) the
    consolidation would mint, then assert compaction raises AND the log is unchanged.

    RED-on-main proof: with `INSERT OR IGNORE` this test FAILS (the chain is deleted, the decoy
    survives, no exception, data lost); with the raw `INSERT` fix it PASSES."""
    from mcp_memory_service.storage.sync.compaction import (
        _deterministic_consolidated_event_id, _compaction_agent_id,
    )

    content = "collision memory"
    h = generate_content_hash(content)
    apply_remote_event(store, _create_event(h, content=content, event_id="c1", hlc_physical=10))
    # chain whose winner is the edit at hlc 50 (largest physical, logical 0)
    for i in range(2, 6):
        apply_remote_event(store, _update_metadata_event(
            h, updates={"tags": [f"t{i}"]}, event_id=f"u{i}", hlc_physical=10 * i))
    s = _sqlite(store)

    # the id compaction WILL mint for this hash (winner pair = (50, 0))
    collide_id = _deterministic_consolidated_event_id(h, 50, 0)
    # seed a DECOY surviving row the compaction will NOT touch: a NON-update_metadata op
    # (op='create') so it is never in the collapsible set, carrying the SAME (agent_id, event_id)
    # the consolidation will mint. The consolidated INSERT must then collide on
    # UNIQUE(agent_id, event_id) → raise → SAVEPOINT rolls the whole batch back (fail-closed).
    decoy_hash = generate_content_hash("decoy")
    s.conn.execute(
        "INSERT INTO sync_events (schema_version, agent_id, event_id, op, content_hash, "
        "payload, created_at, hlc_physical, hlc_logical) VALUES (1,?,?,?,?,?,?,?,?)",
        (_compaction_agent_id(), collide_id, "create", decoy_hash,
         json.dumps({"content_hash": decoy_hash}), time.time(), 999, 0),
    )
    s.conn.commit()

    before_log = _all_events(s)

    with pytest.raises(Exception):
        compact_update_metadata(store, watermark=None)

    after_log = _all_events(s)
    assert after_log == before_log, (
        "fail-closed on identity collision: the chain must NOT be deleted when the consolidated "
        "INSERT cannot land (no silent data loss)")
