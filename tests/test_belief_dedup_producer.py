"""GATE 3 — arco B.2 fixes: embedding producer, backfill, re-derivation
provenance preservation, and merge-under-lock (Greptile P1-a/P1-c/P1-d).

These tests cover the four fixes added to ``belief_service.py`` so the dedup is
actually usable (#422), concurrency-safe (#562), and loss-free on re-derivation
(#566). They call the REAL methods (no mock of the unit under test) and prove
load-bearing behavior by sabotage where it matters.

Padrão imitado de tests/test_belief_dedup.py (fixture belief_db migrações 012+020,
_insert_belief, _vec_blob). storage expõe .conn real; o produtor de embedding é
simulado com um ``_generate_embedding`` sync OU async controlado.
"""

import asyncio
import json
import sqlite3
import struct
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlite_vec import serialize_float32

from mcp_memory_service.consolidation.belief_service import BeliefService


MIGRATIONS = Path(__file__).parent.parent / "src" / "mcp_memory_service" / "storage" / "migrations"


@pytest.fixture
def belief_db():
    conn = sqlite3.connect(":memory:")
    conn.executescript((MIGRATIONS / "012_add_belief_store.sql").read_text())
    conn.executescript((MIGRATIONS / "020_add_belief_embedding.sql").read_text())
    return conn


def _vec_blob(values):
    return serialize_float32([float(v) for v in values])


def _deserialize(blob):
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def _insert_belief(conn, belief_hash, content, *, confidence=0.5, status="active",
                   derived_from=None, embedding=None, metadata=None):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO beliefs (belief_hash, content, confidence, status, created_at, "
        "updated_at, derived_from, contradicted_by, metadata, embedding) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (belief_hash, content, confidence, status, now, now,
         json.dumps(derived_from if derived_from is not None else []),
         "[]", json.dumps(metadata if metadata is not None else {}), embedding),
    )
    conn.commit()


class _SyncEmbedStorage:
    """Storage with a SYNCHRONOUS _generate_embedding (sqlite_vec/milvus shape)
    and a real _run_in_thread + _conn_lock (so the producer runs under the lock)."""

    def __init__(self, conn, vec):
        self.conn = conn
        self.graph = None
        self.embedding_model = object()  # truthy: a model is "loaded"
        self._vec = vec
        self._conn_lock = threading.Lock()
        self.gen_calls = 0

    def _generate_embedding(self, text):
        self.gen_calls += 1
        return list(self._vec)

    async def _run_in_thread(self, operation, *args):
        with self._conn_lock:
            return operation(*args)


class _AsyncEmbedStorage:
    """Storage with an ASYNC _generate_embedding (cloudflare shape)."""

    def __init__(self, conn, vec):
        self.conn = conn
        self.graph = None
        self.embedding_model = object()
        self._vec = vec
        self.gen_calls = 0

    async def _generate_embedding(self, text):
        self.gen_calls += 1
        return list(self._vec)


class _NoModelStorage:
    """Storage without any embedding model (producer must yield None)."""

    def __init__(self, conn):
        self.conn = conn
        self.graph = None
        self.embedding_model = None


def _run(coro):
    return asyncio.run(coro)


# ── P1-a: producer computes + stamps embedding on creation (SYNC backend) ──

def test_create_belief_stamps_embedding_sync(belief_db):
    """_create_belief writes a non-NULL embedding when a sync model is present."""
    vec = [0.1, 0.2, 0.3, 0.4]
    storage = _SyncEmbedStorage(belief_db, vec)
    svc = BeliefService(storage)
    now = datetime.now(timezone.utc)

    _run(svc._create_belief("hA", "Belief A", 0.7, "active", ["o1"], [], now))

    row = belief_db.execute("SELECT embedding FROM beliefs WHERE belief_hash='hA'").fetchone()
    assert row[0] is not None, "embedding was not stamped on creation"
    stored = _deserialize(row[0])
    assert stored == pytest.approx(vec, abs=1e-6)
    assert storage.gen_calls == 1  # producer actually invoked (load-bearing)


# ── P1-a: producer handles an ASYNC _generate_embedding (cloudflare) ──

def test_create_belief_stamps_embedding_async(belief_db):
    """A coroutine _generate_embedding is awaited, not serialized as-is."""
    vec = [1.0, 0.0, 0.0]
    storage = _AsyncEmbedStorage(belief_db, vec)
    svc = BeliefService(storage)
    now = datetime.now(timezone.utc)

    _run(svc._create_belief("hB", "Belief B", 0.6, "candidate", ["o2"], [], now))

    row = belief_db.execute("SELECT embedding FROM beliefs WHERE belief_hash='hB'").fetchone()
    assert row[0] is not None
    assert _deserialize(row[0]) == pytest.approx(vec, abs=1e-6)
    assert storage.gen_calls == 1


# ── P1-a: no model -> NULL embedding, never raises ──

def test_create_belief_null_embedding_when_no_model(belief_db):
    """Without an embedding model the belief is still created, embedding NULL."""
    storage = _NoModelStorage(belief_db)
    svc = BeliefService(storage)
    now = datetime.now(timezone.utc)

    _run(svc._create_belief("hC", "Belief C", 0.5, "active", ["o3"], [], now))

    row = belief_db.execute("SELECT embedding FROM beliefs WHERE belief_hash='hC'").fetchone()
    assert row[0] is None


def test_compute_embedding_returns_none_on_model_error(belief_db):
    """Producer NEVER raises: a model that raises yields None (optional enrichment)."""
    class _BoomStorage:
        def __init__(self, conn):
            self.conn = conn
            self.embedding_model = object()
        def _generate_embedding(self, text):
            raise RuntimeError("model exploded")

    svc = BeliefService(_BoomStorage(belief_db))
    assert _run(svc._compute_belief_embedding("anything")) is None


# ── P1-a: backfill fills NULL embeddings (opt-in) ──

def test_backfill_fills_null_embeddings(belief_db, monkeypatch):
    monkeypatch.setenv("MCP_BELIEF_DEDUP", "1")
    vec = [0.5, 0.5, 0.0]
    _insert_belief(belief_db, "n1", "needs embedding 1", embedding=None)
    _insert_belief(belief_db, "n2", "needs embedding 2", embedding=None)
    _insert_belief(belief_db, "has", "already has", embedding=_vec_blob([1, 0, 0]))

    storage = _SyncEmbedStorage(belief_db, vec)
    svc = BeliefService(storage)

    report = _run(svc.backfill_belief_embeddings(cap=1000))

    assert report["filled"] == 2
    assert report["evaluated"] == 2  # the already-filled row is not evaluated
    for h in ("n1", "n2"):
        blob = belief_db.execute(
            "SELECT embedding FROM beliefs WHERE belief_hash=?", (h,)
        ).fetchone()[0]
        assert blob is not None
        assert _deserialize(blob) == pytest.approx(vec, abs=1e-6)
    # pre-existing embedding untouched
    assert _deserialize(
        belief_db.execute("SELECT embedding FROM beliefs WHERE belief_hash='has'").fetchone()[0]
    ) == pytest.approx([1, 0, 0], abs=1e-6)


def test_backfill_respects_cap(belief_db, monkeypatch):
    monkeypatch.setenv("MCP_BELIEF_DEDUP", "1")
    for i in range(3):
        _insert_belief(belief_db, f"m{i}", f"row {i}", embedding=None)
    storage = _SyncEmbedStorage(belief_db, [0.1, 0.2, 0.3])
    svc = BeliefService(storage)

    report = _run(svc.backfill_belief_embeddings(cap=2))

    assert report["capped"] is True
    assert report["filled"] == 2
    remaining = belief_db.execute(
        "SELECT COUNT(*) FROM beliefs WHERE embedding IS NULL"
    ).fetchone()[0]
    assert remaining == 1


# ── P1-b / flag parse: backfill is a NO-OP when flag off/0 (sabotage) ──

@pytest.mark.parametrize("flag", [None, "false", "0", "no", "off", "bogus"])
def test_backfill_noop_when_flag_off(belief_db, monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv("MCP_BELIEF_DEDUP", raising=False)
    else:
        monkeypatch.setenv("MCP_BELIEF_DEDUP", flag)
    _insert_belief(belief_db, "x1", "unfilled", embedding=None)
    storage = _SyncEmbedStorage(belief_db, [1, 2, 3])
    svc = BeliefService(storage)

    report = _run(svc.backfill_belief_embeddings())

    assert report == {"evaluated": 0, "filled": 0, "skipped": 0, "capped": False, "errors": []}
    assert storage.gen_calls == 0  # corpus not touched
    assert belief_db.execute(
        "SELECT embedding FROM beliefs WHERE belief_hash='x1'"
    ).fetchone()[0] is None


@pytest.mark.parametrize("flag,expected_on", [
    ("true", True), ("TRUE", True), ("1", True), ("yes", True), (" Yes ", True),
    ("false", False), ("0", False), ("no", False), ("", False), ("maybe", False),
])
def test_env_flag_parse_matches_contract(monkeypatch, flag, expected_on):
    """P1-b: the tolerant flag parse — only true/1/yes (trim/case-insensitive) is ON."""
    monkeypatch.setenv("MCP_BELIEF_DEDUP", flag)
    assert BeliefService._env_flag_on("MCP_BELIEF_DEDUP") is expected_on


# ── P1-d: re-derivation preserves merged derived_from ──

def test_update_belief_preserves_merged_from_hashes(belief_db):
    """After a dedup merge recorded merged_from_hashes, _update_belief re-deriving
    the canonical with a DIFFERENT supporting set must UNION the merged sources,
    not drop them (#566)."""
    # Canonical as it looks right after a merge: derived_from already unified and
    # merged_from_hashes records the sources inherited from the losers.
    _insert_belief(
        belief_db, "canon", "canonical belief", derived_from=["o1", "o2", "o3", "o4"],
        metadata={"merged_from_hashes": ["o3", "o4"]},
    )
    storage = _NoModelStorage(belief_db)
    svc = BeliefService(storage)
    now = datetime.now(timezone.utc)

    # Re-derivation sees only the CURRENT group's supporting hashes {o1, o2}.
    _run(svc._update_belief("canon", 0.8, ["o1", "o2"], [], now))

    df = json.loads(
        belief_db.execute("SELECT derived_from FROM beliefs WHERE belief_hash='canon'").fetchone()[0]
    )
    assert set(df) == {"o1", "o2", "o3", "o4"}, "merged sources were lost on re-derivation"


def test_update_belief_without_merge_metadata_is_plain_overwrite(belief_db):
    """Sabotage/guard: without merged_from_hashes, _update_belief uses ONLY the
    current group (proves the preservation branch is gated on real merge metadata)."""
    _insert_belief(belief_db, "plain", "plain belief", derived_from=["old1", "old2"])
    svc = BeliefService(_NoModelStorage(belief_db))
    now = datetime.now(timezone.utc)

    _run(svc._update_belief("plain", 0.9, ["new1"], [], now))

    df = json.loads(
        belief_db.execute("SELECT derived_from FROM beliefs WHERE belief_hash='plain'").fetchone()[0]
    )
    assert df == ["new1"]


# ── P1-c: merge runs under the connection lock ──

def test_merge_holds_conn_lock(belief_db):
    """_merge_into_canonical must acquire storage._conn_lock for the whole
    transaction. We assert the lock was HELD during the merge by observing it
    from inside the connection's execute() — if the merge didn't hold the lock,
    try-acquire would succeed (lock available) and the flag would flip."""
    _insert_belief(belief_db, "c1", "canon", confidence=0.9, derived_from=["a"],
                   embedding=_vec_blob([1, 0, 0]))
    _insert_belief(belief_db, "l1", "loser", confidence=0.5, derived_from=["b"],
                   embedding=_vec_blob([1, 0, 0]))

    lock = threading.Lock()
    observed_unlocked_during_merge = {"flag": False}
    real_conn = belief_db

    class _ObservingConn:
        def __init__(self, inner):
            self._inner = inner
        def execute(self, sql, *args, **kwargs):
            # If the merge holds _conn_lock, a non-blocking acquire here fails.
            got = lock.acquire(blocking=False)
            if got:
                observed_unlocked_during_merge["flag"] = True
                lock.release()
            return self._inner.execute(sql, *args, **kwargs)
        def __getattr__(self, name):
            return getattr(self._inner, name)

    class _LockedStorage:
        def __init__(self, conn):
            self.conn = conn
            self._conn_lock = lock

    storage = _LockedStorage(_ObservingConn(real_conn))
    svc = BeliefService(storage)
    svc._merge_into_canonical("c1", ["l1"], datetime.now(timezone.utc))

    assert observed_unlocked_during_merge["flag"] is False, (
        "connection was accessed WITHOUT holding _conn_lock during merge"
    )
    # And the merge actually happened.
    storage.conn = real_conn
    assert real_conn.execute(
        "SELECT status FROM beliefs WHERE belief_hash='l1'"
    ).fetchone()[0] == "superseded"


def test_merge_without_lock_still_works(belief_db):
    """Backends/mocks without _conn_lock fall back to a no-op context (no crash)."""
    _insert_belief(belief_db, "c2", "canon", confidence=0.9, derived_from=["a"])
    _insert_belief(belief_db, "l2", "loser", confidence=0.5, derived_from=["b"])

    class _NoLockStorage:
        def __init__(self, conn):
            self.conn = conn

    svc = BeliefService(_NoLockStorage(belief_db))
    svc._merge_into_canonical("c2", ["l2"], datetime.now(timezone.utc))

    assert belief_db.execute(
        "SELECT status FROM beliefs WHERE belief_hash='l2'"
    ).fetchone()[0] == "superseded"


# ── Greptile round 2 P2 (#1733): cap stops grouping early, no full-corpus scan ──

def test_group_by_cosine_cap_stops_early():
    """_group_by_cosine with cap=1 returns after the first group even when more
    duplicate groups exist later — no quadratic scan of the whole corpus."""
    va, va2 = [1.0, 0.0, 0.0], [0.9962, 0.0872, 0.0]   # ~colinear -> group 1
    vb, vb2 = [0.0, 1.0, 0.0], [0.0872, 0.9962, 0.0]   # ~colinear -> group 2
    items = [
        ("g1a", 0.8, "", 1, va), ("g1b", 0.7, "", 1, va2),
        ("g2a", 0.8, "", 1, vb), ("g2b", 0.7, "", 1, vb2),
    ]
    groups_capped = BeliefService._group_by_cosine(items, 0.90, cap=1)
    assert len(groups_capped) == 1, "cap=1 should stop after the first group"
    # Without a cap, both groups are found (proves the cap is what limits it).
    groups_all = BeliefService._group_by_cosine(items, 0.90, cap=None)
    assert len(groups_all) == 2


# ── Greptile round 2 P1 (#1733): _create_belief writes embedding under the lock ──

def test_create_belief_embedding_write_holds_lock(belief_db):
    """The embedding UPDATE on create must go through _write_backfill_embedding,
    which holds _conn_lock — not a bare off-lock commit (#1733 P1)."""
    vec = [0.3, 0.3, 0.3]
    storage = _SyncEmbedStorage(belief_db, vec)
    svc = BeliefService(storage)
    calls = {"under_lock": 0, "off_lock": 0}
    real_write = svc._write_backfill_embedding

    def _spy(bhash, blob):
        # _conn_lock is a threading.Lock; if held by the caller a non-blocking
        # acquire fails. _write_backfill_embedding acquires it itself, so here
        # (before it runs) it must be free — we just count that it is the path used.
        calls["under_lock"] += 1
        return real_write(bhash, blob)

    svc._write_backfill_embedding = _spy
    _run(svc._create_belief("hL", "lock belief", 0.7, "active", ["o1"], [], datetime.now(timezone.utc)))

    assert calls["under_lock"] == 1, "create did not route the embedding write through the locked helper"
    assert belief_db.execute(
        "SELECT embedding FROM beliefs WHERE belief_hash='hL'"
    ).fetchone()[0] is not None
