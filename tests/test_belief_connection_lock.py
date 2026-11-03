"""Real shared-connection interleavings between derivation and dedup merges."""

import asyncio
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlite_vec import serialize_float32

from mcp_memory_service.consolidation.belief_service import BeliefService, _hash_content
from mcp_memory_service.models.memory import Memory

MIGRATIONS = (
    Path(__file__).resolve().parents[1] / "src/mcp_memory_service/storage/migrations"
)


@pytest.fixture
def belief_connection():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    for migration in ("012_add_belief_store.sql", "020_add_belief_embedding.sql"):
        conn.executescript((MIGRATIONS / migration).read_text())
    for name, confidence, count in (
        ("canon", 0.9, 10),
        ("loser", 0.5, 5),
        ("other", 0.3, 0),
    ):
        conn.execute(
            "INSERT INTO beliefs (belief_hash, content, confidence, status, created_at, "
            "updated_at, derived_from, contradicted_by, metadata, embedding) "
            "VALUES (?, ?, ?, 'active', '2026-01-01', '2026-01-01', ?, '[]', ?, ?)",
            (
                name,
                name,
                confidence,
                json.dumps([name + "-source"]),
                json.dumps({"injected_used": count}),
                serialize_float32([1.0, 0.0]),
            ),
        )
    conn.commit()
    yield conn
    conn.close()


class _PausedConnectionLock:
    def __init__(self, state):
        self.state = state

    def __enter__(self):
        if threading.get_ident() != self.state.worker.ident:
            self.state.progressed.set()
        self.state.real_lock.acquire()

    def __exit__(self, *_):
        self.state.real_lock.release()


class _PausedCursor:
    def __init__(self, inner, state):
        self.inner, self.state = inner, state

    def _signal_read(self):
        if self.state.reading and threading.get_ident() != self.state.worker.ident:
            self.state.progressed.set()

    def fetchone(self):
        row = self.inner.fetchone()
        self._signal_read()
        return row

    def fetchall(self):
        rows = self.inner.fetchall()
        self._signal_read()
        return rows

    def __getattr__(self, name):
        return getattr(self.inner, name)


class _PausedConnection:
    def __init__(self, state):
        self.state = state

    def execute(self, sql, args=()):
        state = self.state
        worker = threading.get_ident() == state.worker.ident
        if worker and sql.startswith("UPDATE beliefs SET status"):
            raise RuntimeError("forced loser update failure")
        cursor = state.conn.execute(sql, args)
        if worker and sql.startswith("UPDATE beliefs SET derived_from"):
            state.paused.set()
            if not state.resume.wait(5):
                raise RuntimeError("merge was not released")
        return _PausedCursor(cursor, state)

    def commit(self):
        self.state.conn.commit()
        if threading.get_ident() != self.state.worker.ident:
            self.state.progressed.set()

    def __getattr__(self, name):
        return getattr(self.state.conn, name)


def _run_paused_merge(state):
    try:
        state.service._merge_into_canonical("canon", ["loser"], state.now)
    except Exception as error:
        state.merge_errors.append(str(error))


def _release_paused_merge(state):
    # Correct operations signal BEFORE waiting for the connection lock.
    # Unguarded writers signal only AFTER committing the shared transaction.
    if not state.progressed.wait(5):
        state.coordination_errors.append("derivation made no progress")
    state.resume.set()


def _finish_paused_merge(state):
    if state.finished:
        return
    state.resume.set()
    state.worker.join(5)
    if state.controller.ident is not None:
        state.controller.join(5)
    state.finished = True
    assert not state.worker.is_alive()
    assert not state.controller.is_alive()
    assert not state.coordination_errors
    assert state.merge_errors == ["forced loser update failure"]


@pytest.fixture
def paused_merge(belief_connection):
    state = SimpleNamespace(
        conn=belief_connection,
        main_thread=threading.get_ident(),
        paused=threading.Event(),
        progressed=threading.Event(),
        resume=threading.Event(),
        real_lock=threading.Lock(),
        merge_errors=[],
        coordination_errors=[],
        reading=False,
        finished=False,
        now=datetime.now(timezone.utc),
    )
    state.service = BeliefService(
        SimpleNamespace(
            conn=_PausedConnection(state),
            _conn_lock=_PausedConnectionLock(state),
            embedding_model=None,
        )
    )
    state.worker = threading.Thread(target=_run_paused_merge, args=(state,))
    state.controller = threading.Thread(target=_release_paused_merge, args=(state,))
    state.finish = lambda: _finish_paused_merge(state)
    state.worker.start()
    try:
        assert state.paused.wait(5), "merge did not reach its partial write"
        state.controller.start()
        yield state
    finally:
        state.finish()


def _assert_merge_rolled_back(conn):
    row = conn.execute(
        "SELECT derived_from, metadata FROM beliefs WHERE belief_hash='canon'"
    ).fetchone()
    assert json.loads(row[0]) == [
        "canon-source"
    ], "derivation committed a partial merge"
    assert json.loads(row[1])["injected_used"] == 10
    assert (
        conn.execute("SELECT status FROM beliefs WHERE belief_hash='loser'").fetchone()[
            0
        ]
        == "active"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "update", "status"])
async def test_derivation_writes_cannot_commit_a_partial_merge(
    belief_connection, paused_merge, operation
):
    service, now = paused_merge.service, paused_merge.now
    if operation == "create":
        await service._create_belief(
            "new", "new belief", 0.8, "active", ["new-source"], [], now
        )
    elif operation == "update":
        await service._update_belief("other", 0.8, ["fresh-source"], [], now)
    else:
        await service._set_belief_status("other", "disputed", now)
    paused_merge.finish()
    _assert_merge_rolled_back(belief_connection)
    query, expected = {
        "create": ("SELECT content FROM beliefs WHERE belief_hash='new'", "new belief"),
        "update": ("SELECT confidence FROM beliefs WHERE belief_hash='other'", 0.8),
        "status": ("SELECT status FROM beliefs WHERE belief_hash='other'", "disputed"),
    }[operation]
    assert belief_connection.execute(query).fetchone()[0] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["get", "list", "candidates"])
async def test_belief_reads_do_not_observe_a_partial_merge(
    belief_connection, paused_merge, operation
):
    paused_merge.reading = True
    service = paused_merge.service
    if operation == "get":
        result = json.loads((await service._get_belief("canon"))["derived_from"])
    elif operation == "list":
        result = (await service.get_beliefs(min_confidence=0.0))[0]["derived_from"]
    else:
        result = service._load_dedup_candidates({"evaluated": 0, "skipped": 0})[0][3]
    paused_merge.finish()
    _assert_merge_rolled_back(belief_connection)
    assert (
        result
        == {"get": ["canon-source"], "list": ["canon-source"], "candidates": 1}[
            operation
        ]
    )


class _MetadataLock:
    def __init__(self, state):
        self.state = state

    def __enter__(self):
        state = self.state
        if threading.get_ident() == state.worker.ident and state.real_lock.locked():
            state.continue_update.set()
        state.real_lock.acquire()

    def __exit__(self, *_):
        self.state.real_lock.release()


class _MetadataCursor:
    def __init__(self, inner, state):
        self.inner, self.state = inner, state

    def fetchone(self):
        row = self.inner.fetchone()
        self.state.metadata_read.set()
        assert self.state.continue_update.wait(
            5
        ), "merge did not reach the competing update"
        return row


class _MetadataConnection:
    def __init__(self, state):
        self.state = state

    def execute(self, sql, args=()):
        cursor = self.state.conn.execute(sql, args)
        if (
            sql.startswith("SELECT metadata")
            and threading.get_ident() != self.state.worker.ident
        ):
            return _MetadataCursor(cursor, self.state)
        return cursor

    def __getattr__(self, name):
        return getattr(self.state.conn, name)


def _merge_after_metadata_read(state):
    try:
        assert state.metadata_read.wait(5)
        state.service._merge_into_canonical("canon", ["loser"], state.now)
    except Exception as error:
        state.errors.append(error)
    finally:
        state.continue_update.set()


def _finish_rederivation(state):
    if state.finished:
        return
    state.continue_update.set()
    state.worker.join(5)
    state.finished = True
    assert not state.worker.is_alive()
    assert not state.errors


@pytest.fixture
def overlapping_rederivation(belief_connection):
    state = SimpleNamespace(
        conn=belief_connection,
        metadata_read=threading.Event(),
        continue_update=threading.Event(),
        real_lock=threading.Lock(),
        main_thread=threading.get_ident(),
        errors=[],
        now=datetime.now(timezone.utc),
        finished=False,
    )
    state.service = BeliefService(
        SimpleNamespace(
            conn=_MetadataConnection(state),
            _conn_lock=_MetadataLock(state),
        )
    )
    state.worker = threading.Thread(target=_merge_after_metadata_read, args=(state,))
    state.finish = lambda: _finish_rederivation(state)
    state.worker.start()
    try:
        yield state
    finally:
        state.finish()


@pytest.mark.asyncio
async def test_rederivation_preserves_sources_from_an_overlapping_merge(
    belief_connection, overlapping_rederivation
):
    state = overlapping_rederivation
    await state.service._update_belief("canon", 0.9, ["fresh-source"], [], state.now)
    state.finish()
    row = belief_connection.execute(
        "SELECT derived_from FROM beliefs WHERE belief_hash='canon'"
    ).fetchone()
    assert set(json.loads(row[0])) == {"fresh-source", "loser-source"}


class _SupersedingReader(BeliefService):
    async def _get_belief(self, belief_hash):
        snapshot = await super()._get_belief(belief_hash)
        self._merge_into_canonical("canon", [belief_hash], datetime.now(timezone.utc))
        return snapshot


@pytest.fixture
def superseded_candidate(belief_connection, monkeypatch):
    content = "Use parameterized database queries when processing requests received from external clients."
    belief_hash = _hash_content(content)
    belief_connection.execute(
        "UPDATE beliefs SET belief_hash=?, content=?, status='candidate' WHERE belief_hash='loser'",
        (belief_hash, content),
    )
    belief_connection.commit()
    observations = [
        Memory(
            content=content, content_hash=f"fresh-source-{i}", memory_type="observation"
        )
        for i in range(5)
    ]
    storage = SimpleNamespace(
        conn=belief_connection,
        _conn_lock=threading.Lock(),
        embedding_model=None,
        get_all_memories=AsyncMock(return_value=observations),
        retrieve=AsyncMock(return_value=[]),
    )
    # Confidence is fixed so this case specifically reaches candidate promotion.
    monkeypatch.setattr(
        "mcp_memory_service.consolidation.belief_service.derive_confidence",
        lambda **_: 0.9,
    )
    return _SupersedingReader(storage), belief_hash


@pytest.mark.asyncio
async def test_derivation_does_not_reactivate_a_concurrently_superseded_loser(
    belief_connection, superseded_candidate
):
    service, belief_hash = superseded_candidate
    stats = await service.derive_beliefs()
    assert not stats["errors"]
    assert stats["updated"] == 1
    row = belief_connection.execute(
        "SELECT status FROM beliefs WHERE belief_hash=?", (belief_hash,)
    ).fetchone()
    assert row[0] == "superseded"
    assert stats["promoted"] == 0


class _ObservedLock:
    def __init__(self):
        self.lock = threading.Lock()
        self.attempted = threading.Event()

    def __enter__(self):
        self.attempted.set()
        self.lock.acquire()

    def __exit__(self, *_):
        self.lock.release()


def _hold_connection_lock(lock, held, release):
    with lock.lock:
        held.set()
        release.wait(2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation", ["get", "list", "create", "update", "status", "backfill"]
)
async def test_waiting_for_connection_does_not_freeze_event_loop(
    belief_connection, monkeypatch, operation
):
    lock = _ObservedLock()
    held, release = threading.Event(), threading.Event()
    worker = threading.Thread(target=_hold_connection_lock, args=(lock, held, release))
    storage = SimpleNamespace(
        conn=belief_connection, _conn_lock=lock, embedding_model=None
    )
    service = BeliefService(storage)
    now = datetime.now(timezone.utc)
    monkeypatch.setenv("MCP_BELIEF_DEDUP", "true")
    worker.start()
    assert held.wait(2)
    operations = {
        "get": lambda: service._get_belief("canon"),
        "list": service.get_beliefs,
        "create": lambda: service._create_belief(
            "new", "new belief", 0.8, "active", [], [], now
        ),
        "update": lambda: service._update_belief("other", 0.8, [], [], now),
        "status": lambda: service._set_belief_status("other", "disputed", now),
        "backfill": service.backfill_belief_embeddings,
    }
    task = asyncio.create_task(operations[operation]())
    try:
        assert await asyncio.to_thread(lock.attempted.wait, 1)
        assert worker.is_alive(), "waiting for the lock blocked the event loop"
        assert not task.done()
    finally:
        release.set()
        await task
        worker.join(2)
