"""Beliefs and the NLI on-store check respect the claimed store (#1877, part of #1683).

Beliefs are not stored per store. A belief is visible in store S when none of its
live source or contradicting memories sits in another store. Deleted memories
don't count: retention deletes all the time, and a deleted memory leaks nothing.
``None`` and ``"all"`` are unscoped.

The on-store check compares a new memory only with beliefs visible in the store
it was written to. ``"default"`` is an ordinary store here, so a single-store user
(everything in ``"default"``) sees every belief and keeps challenging them. PR
#1714 treated only ``None`` as global and switched the challenge off for everyone,
because ``handle_store_memory`` defaults ``store`` to ``"default"``.
"""

import asyncio
import os
import shutil
import tempfile
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
import pytest_asyncio

from mcp_memory_service.consolidation import quarantine as quarantine_mod
from mcp_memory_service.consolidation.belief_service import BeliefService
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.reasoning.nli import NLIResult
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash

os.environ["MCP_MEMORY_ALLOW_HASH_EMBEDDINGS"] = "1"


@pytest_asyncio.fixture
async def storage():
    d = tempfile.mkdtemp(prefix="mcp-test-store-scoped-beliefs-")
    s = SqliteVecMemoryStorage(os.path.join(d, "test.db"))
    await s.initialize()
    yield s
    await s.close()
    shutil.rmtree(d, ignore_errors=True)


async def _mem(storage, content, store):
    m = Memory(content=content, content_hash=generate_content_hash(content), tags=["t"])
    ok, msg = await storage.store(m, skip_semantic_dedup=True, store=store)
    assert ok, msg
    return m.content_hash


async def _belief(storage, name, sources, contradicting=()):
    await BeliefService(storage)._create_belief(
        name, f"belief {name}", 0.9, "active", list(sources), list(contradicting),
        datetime.now(timezone.utc),
    )


async def _visible(storage, store):
    beliefs = await BeliefService(storage).get_beliefs(store=store)
    return {b["belief_hash"] for b in beliefs}


@pytest_asyncio.fixture
async def beliefs(storage):
    a1 = await _mem(storage, "Store A deploys with blue-green.", "a")
    a2 = await _mem(storage, "Store A keeps two replicas.", "a")
    b1 = await _mem(storage, "Store B deploys with canaries.", "b")
    b_gone = await _mem(storage, "Store B once used rolling deploys.", "b")
    await storage.delete(b_gone)

    await _belief(storage, "from-a", [a1, a2])
    await _belief(storage, "from-b", [b1])
    await _belief(storage, "mixed", [a1, b1])
    await _belief(storage, "no-sources", [])
    await _belief(storage, "a-plus-deleted-b", [a1, b_gone])
    await _belief(storage, "contradicted-from-b", [a1], contradicting=[b1])
    return storage


@pytest.mark.asyncio
async def test_store_sees_only_beliefs_grounded_in_it(beliefs):
    assert await _visible(beliefs, "a") == {"from-a", "no-sources", "a-plus-deleted-b"}
    assert await _visible(beliefs, "b") == {"from-b", "no-sources"}


@pytest.mark.asyncio
@pytest.mark.parametrize("store", [None, "all"])
async def test_unscoped_sees_every_belief(beliefs, store):
    assert await _visible(beliefs, store) == {
        "from-a", "from-b", "mixed", "no-sources", "a-plus-deleted-b", "contradicted-from-b",
    }


@pytest.mark.asyncio
async def test_single_store_user_keeps_beliefs_after_source_deletion(storage):
    d1 = await _mem(storage, "The build runs on Linux runners.", "default")
    d2 = await _mem(storage, "The build also runs on macOS runners.", "default")
    await _belief(storage, "builds", [d1, d2])

    await storage.delete(d2)

    assert await _visible(storage, "default") == {"builds"}


class _Contradicts:
    backend = "cascade"

    async def classify(self, premise, hypothesis):
        return NLIResult(label="contradiction", confidence=0.95)

    def max_achievable_confidence(self):
        return 0.95


@pytest.mark.asyncio
async def test_on_store_check_compares_only_beliefs_visible_in_the_store(beliefs, monkeypatch):
    """A memory written to store A is never quarantined against store B's belief."""
    monkeypatch.setenv("MCP_NLI_ON_STORE_BUDGET_S", "5")
    seen = []
    svc = BeliefService(beliefs)
    original = svc.get_beliefs

    async def recording(*args, **kwargs):
        result = await original(*args, **kwargs)
        seen.extend(b["belief_hash"] for b in result)
        return result

    monkeypatch.setattr(svc, "get_beliefs", recording)
    h = await _mem(beliefs, "Store A deploys straight to production.", "a")

    with patch("mcp_memory_service.reasoning.nli.NLIClassifier", return_value=_Contradicts()):
        result = await quarantine_mod.check_beliefs_on_store(
            beliefs, svc, "Store A deploys straight to production.", h, store="a"
        )

    assert result and result["status"] == "quarantined"
    assert "from-b" not in seen and "mixed" not in seen
    assert set(seen) <= {"from-a", "no-sources", "a-plus-deleted-b"}


class _Server:
    def __init__(self, storage):
        from mcp_memory_service.services.memory_service import MemoryService

        self.storage = storage
        self.memory_service = MemoryService(storage)

    async def _ensure_storage_initialized(self):
        return self.storage


@pytest.mark.asyncio
async def test_default_store_still_challenges_beliefs_through_the_handler(storage, monkeypatch):
    """Three contradicting memories stored without a store argument challenge the
    belief, as on main. This is the path #1714 broke."""
    from mcp_memory_service.server.handlers.memory import handle_store_memory

    monkeypatch.setenv("MCP_NLI_ON_STORE", "true")
    monkeypatch.delenv("MCP_NLI_ON_STORE_ASYNC", raising=False)
    monkeypatch.setenv("MCP_NLI_ON_STORE_BUDGET_S", "5")
    monkeypatch.setattr(quarantine_mod, "CONTRADICTION_THRESHOLD", 3)

    src = await _mem(storage, "Releases ship on Tuesdays.", "default")
    await _belief(storage, "release-day", [src])

    server = _Server(storage)
    with patch("mcp_memory_service.reasoning.nli.NLIClassifier", return_value=_Contradicts()):
        for content in (
            "Releases ship on Fridays now.",
            "We moved releases to Thursdays.",
            "Releases no longer ship on Tuesdays.",
        ):
            await handle_store_memory(server, {"content": content, "tags": ["t"]})

    # challenge_belief() marks the belief disputed and re-derives it; with one
    # source it lands at candidate. Either way it is no longer active.
    row = storage.conn.execute(
        "SELECT status FROM beliefs WHERE belief_hash = 'release-day'"
    ).fetchone()
    assert row[0] != "active"


@pytest.mark.asyncio
async def test_background_check_receives_the_store(monkeypatch):
    seen = {}

    async def fake_check(storage, belief_service, content, content_hash, store=None):
        seen["store"] = store
        return None

    async def no_pairs(*args, **kwargs):
        return {"pairs_detected": 0}

    monkeypatch.setattr(quarantine_mod, "check_beliefs_on_store", fake_check)
    monkeypatch.setattr("mcp_memory_service.reasoning.nli.detect_contradictions_nli", no_pairs)

    assert quarantine_mod.schedule_on_store_checks(object(), "text", "hash1234", store="work")
    await asyncio.gather(*list(quarantine_mod._background_checks))

    assert seen["store"] == "work"


@pytest.mark.asyncio
async def test_include_beliefs_no_longer_needs_the_global_grant(storage):
    from mcp_memory_service.server_impl import MemoryServer

    server = MemoryServer(storage=storage)
    target = await server.store_grant_target(
        "memory_search", {"query": "x", "store": "a", "include_beliefs": True}
    )
    assert target == "a"


# --- NLI contradiction check on store (GHSA-989p-hfjf-q97q) -----------------
#
# detect_contradictions_nli() searched without a store, so a memory written to
# store "a" was compared with the default store's memories and linked to them
# with a "contradicts" edge, and the caller was told how many it had hit.

_NEW = "The service runs on Postgres 16."
_OLD = "The service runs on Postgres 12."


class _Graph:
    """Entity graph stand-in: the real one is not partitioned by store."""

    def __init__(self, hashes):
        self.hashes = hashes
        self.edges = []

    async def find_memories_by_entity(self, entity):
        return list(self.hashes)

    async def store_association(self, source_hash, target_hash, **kwargs):
        self.edges.append((source_hash, target_hash))


async def _nli_setup(storage, monkeypatch, other_store):
    """An older statement in ``other_store``, similar enough to reach the classifier."""
    monkeypatch.setenv("MCP_NLI_ENABLED", "true")
    dim = storage.embedding_dimension
    # Cosine 0.2 scores 0.6, inside the 0.4-0.75 band the classifier is asked about.
    vectors = {_NEW: [1.0, 0.0], _OLD: [0.2, 0.98]}
    monkeypatch.setattr(
        storage, "_generate_embedding",
        lambda text: vectors.get(text, [0.0, 1.0]) + [0.0] * (dim - 2),
    )
    old = Memory(
        content=_OLD, content_hash=generate_content_hash(_OLD), tags=["t"],
        metadata={"entities": ["Postgres"]},
    )
    ok, msg = await storage.store(old, skip_semantic_dedup=True, store=other_store)
    assert ok, msg
    graph = _Graph([old.content_hash])

    async def get_graph():
        return graph

    monkeypatch.setattr("mcp_memory_service.server.handlers.graph.get_graph_storage", get_graph)
    return graph


@pytest.mark.asyncio
@pytest.mark.parametrize("other_store, edges", [("a", 1), ("default", 0)])
async def test_nli_on_store_stays_in_the_store_through_the_handler(
    storage, monkeypatch, other_store, edges
):
    from mcp_memory_service.server.handlers.memory import handle_store_memory

    monkeypatch.setenv("MCP_NLI_ON_STORE", "true")
    monkeypatch.delenv("MCP_NLI_ON_STORE_ASYNC", raising=False)
    graph = await _nli_setup(storage, monkeypatch, other_store)

    with patch("mcp_memory_service.reasoning.nli.NLIClassifier", return_value=_Contradicts()):
        out = await handle_store_memory(_Server(storage), {
            "content": _NEW, "tags": ["t"], "store": "a",
            "metadata": {"entities": ["Postgres"]},
        })

    assert len(graph.edges) == edges
    assert ("Contradiction detected" in out[0].text) == bool(edges)


@pytest.mark.asyncio
@pytest.mark.parametrize("other_store, edges", [("a", 1), ("default", 0)])
async def test_nli_background_check_stays_in_the_store(storage, monkeypatch, other_store, edges):
    graph = await _nli_setup(storage, monkeypatch, other_store)
    new = Memory(
        content=_NEW, content_hash=generate_content_hash(_NEW), tags=["t"],
        metadata={"entities": ["Postgres"]},
    )
    ok, msg = await storage.store(new, skip_semantic_dedup=True, store="a")
    assert ok, msg

    with patch("mcp_memory_service.reasoning.nli.NLIClassifier", return_value=_Contradicts()):
        await quarantine_mod._run_on_store_checks(storage, _NEW, new.content_hash, "a")

    assert len(graph.edges) == edges
