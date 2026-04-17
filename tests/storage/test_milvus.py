"""Tests for the Milvus storage backend.

All tests use Milvus Lite (a single local file, no server required) via
``tmp_path`` so each test gets its own isolated collection.
"""

import asyncio
import os
import uuid

import pytest
import pytest_asyncio

# Skip the whole module if pymilvus / milvus-lite are not installed.
pymilvus = pytest.importorskip("pymilvus")
milvus_lite = pytest.importorskip("milvus_lite")

from src.mcp_memory_service.models.memory import Memory  # noqa: E402
from src.mcp_memory_service.storage.milvus import MilvusMemoryStorage  # noqa: E402
from src.mcp_memory_service.utils.hashing import generate_content_hash  # noqa: E402


@pytest.fixture(autouse=True)
def _offline_model_env(monkeypatch):
    """Force Hugging Face offline mode for the duration of every Milvus test.

    Without this, running the Milvus suite as part of the full pytest session
    can pick up network access attempts left in the environment by other
    suites — when the sandbox machine has no network, that fails the fixture
    setup even though the embedding model is already cached on disk.
    """
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


# Share one Milvus Lite DB file across the module. Each test gets its own
# collection inside that file so the tests stay isolated without spawning a
# fresh milvus-lite daemon process for every test.
@pytest.fixture(scope="module")
def milvus_db_path(tmp_path_factory):
    return tmp_path_factory.mktemp("milvus_backend") / "milvus.db"


@pytest_asyncio.fixture
async def storage(milvus_db_path):
    """Per-test storage using a fresh collection inside a shared DB file."""
    collection_name = f"mcp_memory_{uuid.uuid4().hex[:12]}"
    instance = MilvusMemoryStorage(
        uri=str(milvus_db_path),
        collection_name=collection_name,
    )
    await instance.initialize()
    try:
        yield instance
    finally:
        try:
            if instance.client is not None and instance.client.has_collection(collection_name):
                instance.client.drop_collection(collection_name)
        except Exception:
            pass
        await instance.close()


@pytest.fixture
def sample_memory():
    content = "Milvus backend smoke test — Python FastAPI handlers."
    return Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=["milvus", "fastapi"],
        memory_type="note",
        metadata={"priority": "medium", "source": "unit-test"},
    )


# -- Initialization ---------------------------------------------------------


@pytest.mark.asyncio
async def test_initialization_creates_collection(milvus_db_path):
    name = f"init_collection_{uuid.uuid4().hex[:8]}"
    storage = MilvusMemoryStorage(uri=str(milvus_db_path), collection_name=name)
    try:
        await storage.initialize()
        assert storage.client is not None
        assert storage.client.has_collection(collection_name=name)
        assert storage.embedding_dimension > 0
    finally:
        try:
            if storage.client is not None and storage.client.has_collection(name):
                storage.client.drop_collection(name)
        except Exception:
            pass
        await storage.close()


@pytest.mark.asyncio
async def test_initialize_is_idempotent(storage):
    # Calling initialize() twice must not raise or re-create the collection.
    await storage.initialize()
    await storage.initialize()
    assert storage.client is not None


# -- Store / retrieve -------------------------------------------------------


@pytest.mark.asyncio
async def test_store_memory(storage, sample_memory):
    ok, msg = await storage.store(sample_memory)
    assert ok, msg
    assert "success" in msg.lower()

    retrieved = await storage.get_by_hash(sample_memory.content_hash)
    assert retrieved is not None
    assert retrieved.content == sample_memory.content
    assert set(retrieved.tags) == set(sample_memory.tags)
    assert retrieved.memory_type == "note"


@pytest.mark.asyncio
async def test_store_duplicate_rejected(storage, sample_memory):
    ok, _ = await storage.store(sample_memory)
    assert ok
    ok2, msg2 = await storage.store(sample_memory)
    assert not ok2
    assert "duplicate" in msg2.lower()


@pytest.mark.asyncio
async def test_retrieve_semantic_search(storage, sample_memory):
    await storage.store(sample_memory)
    results = await storage.retrieve("python fastapi", n_results=3)
    assert len(results) >= 1
    assert results[0].memory.content_hash == sample_memory.content_hash
    assert 0.0 <= results[0].relevance_score <= 1.0
    assert results[0].debug_info["backend"] == "milvus"


@pytest.mark.asyncio
async def test_retrieve_ranks_more_similar_higher(storage):
    close_content = "Python type hints make dataclasses more self-documenting."
    far_content = "The Saturn V rocket launched Apollo 11 in 1969."
    close_mem = Memory(
        content=close_content,
        content_hash=generate_content_hash(close_content),
        tags=["python"],
    )
    far_mem = Memory(
        content=far_content,
        content_hash=generate_content_hash(far_content),
        tags=["space"],
    )
    await storage.store(close_mem, skip_semantic_dedup=True)
    await storage.store(far_mem, skip_semantic_dedup=True)

    results = await storage.retrieve("python typing", n_results=5)
    ranks = {r.memory.content_hash: i for i, r in enumerate(results)}
    assert close_mem.content_hash in ranks
    if far_mem.content_hash in ranks:
        assert ranks[close_mem.content_hash] < ranks[far_mem.content_hash]


@pytest.mark.asyncio
async def test_retrieve_with_tag_filter(storage):
    # Two memories, different tags.
    a = Memory(
        content="Alpha memory about databases.",
        content_hash=generate_content_hash("Alpha memory about databases."),
        tags=["alpha"],
    )
    b = Memory(
        content="Beta memory about databases.",
        content_hash=generate_content_hash("Beta memory about databases."),
        tags=["beta"],
    )
    await storage.store(a, skip_semantic_dedup=True)
    await storage.store(b, skip_semantic_dedup=True)

    results = await storage.retrieve("database", n_results=5, tags=["alpha"])
    hashes = [r.memory.content_hash for r in results]
    assert a.content_hash in hashes
    assert b.content_hash not in hashes


# -- store_batch ------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_batch(storage):
    memories = [
        Memory(
            content=f"Batch memory {i} about distinct subject {i}",
            content_hash=generate_content_hash(f"Batch memory {i} about distinct subject {i}"),
            tags=["batch", f"item-{i}"],
        )
        for i in range(5)
    ]
    results = await storage.store_batch(memories)
    assert len(results) == 5
    assert all(ok for ok, _ in results)

    assert await storage.count_all_memories() == 5

    # Re-running the same batch should see every item as a duplicate.
    second = await storage.store_batch(memories)
    assert all((not ok) and "duplicate" in msg.lower() for ok, msg in second)


# -- Tag search -------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_by_tag(storage, sample_memory):
    await storage.store(sample_memory)
    hits = await storage.search_by_tag(["milvus"])
    assert len(hits) == 1
    assert hits[0].content_hash == sample_memory.content_hash

    assert await storage.search_by_tag(["nope"]) == []
    assert await storage.search_by_tag([]) == []


@pytest.mark.asyncio
async def test_search_by_tag_exact_not_substring(storage):
    """Regression: tag matching must not treat stored tags as substrings."""
    mem_exact = Memory(
        content="Exact tag memory",
        content_hash=generate_content_hash("Exact tag memory"),
        tags=["test"],
    )
    mem_longer = Memory(
        content="Longer tag memory",
        content_hash=generate_content_hash("Longer tag memory"),
        tags=["testing"],
    )
    mem_hyphen = Memory(
        content="Hyphenated tag memory",
        content_hash=generate_content_hash("Hyphenated tag memory"),
        tags=["my-test-tag"],
    )
    for m in (mem_exact, mem_longer, mem_hyphen):
        await storage.store(m, skip_semantic_dedup=True)

    hashes = {m.content_hash for m in await storage.search_by_tag(["test"])}
    assert mem_exact.content_hash in hashes
    assert mem_longer.content_hash not in hashes
    assert mem_hyphen.content_hash not in hashes


@pytest.mark.asyncio
async def test_search_by_tags_and_or(storage):
    a = Memory(
        content="Memory A", content_hash=generate_content_hash("Memory A"),
        tags=["alpha", "shared"],
    )
    b = Memory(
        content="Memory B", content_hash=generate_content_hash("Memory B"),
        tags=["beta", "shared"],
    )
    c = Memory(
        content="Memory C", content_hash=generate_content_hash("Memory C"),
        tags=["gamma"],
    )
    for m in (a, b, c):
        await storage.store(m, skip_semantic_dedup=True)

    # OR — any of the tags matches.
    or_hits = {m.content_hash for m in await storage.search_by_tags(["alpha", "gamma"], operation="OR")}
    assert or_hits == {a.content_hash, c.content_hash}

    # AND — all tags must match.
    and_hits = {m.content_hash for m in await storage.search_by_tags(["alpha", "shared"], operation="AND")}
    assert and_hits == {a.content_hash}


# -- Delete -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete(storage, sample_memory):
    await storage.store(sample_memory)
    ok, msg = await storage.delete(sample_memory.content_hash)
    assert ok
    assert sample_memory.content_hash in msg

    assert await storage.get_by_hash(sample_memory.content_hash) is None


@pytest.mark.asyncio
async def test_delete_nonexistent(storage):
    ok, msg = await storage.delete("nonexistent-hash-0001")
    assert not ok
    assert "not found" in msg.lower()


@pytest.mark.asyncio
async def test_delete_by_tag(storage):
    shared = Memory(
        content="Shared tag memory",
        content_hash=generate_content_hash("Shared tag memory"),
        tags=["shared"],
    )
    other = Memory(
        content="Keeper memory",
        content_hash=generate_content_hash("Keeper memory"),
        tags=["keep"],
    )
    await storage.store(shared, skip_semantic_dedup=True)
    await storage.store(other, skip_semantic_dedup=True)

    count, msg = await storage.delete_by_tag("shared")
    assert count == 1
    assert "1 memories" in msg

    remaining = {m.content_hash for m in await storage.search_by_tag(["keep"])}
    assert remaining == {other.content_hash}

    # Unknown tag — no delete, no crash.
    zero, _ = await storage.delete_by_tag("does-not-exist")
    assert zero == 0


@pytest.mark.asyncio
async def test_delete_by_tags(storage):
    a = Memory(content="A", content_hash=generate_content_hash("A"), tags=["t1"])
    b = Memory(content="B", content_hash=generate_content_hash("B"), tags=["t2"])
    c = Memory(content="C", content_hash=generate_content_hash("C"), tags=["t3"])
    for m in (a, b, c):
        await storage.store(m, skip_semantic_dedup=True)

    count, _, hashes = await storage.delete_by_tags(["t1", "t2"])
    assert count == 2
    assert set(hashes) == {a.content_hash, b.content_hash}
    remaining = {m.content_hash for m in await storage.get_all_memories()}
    assert remaining == {c.content_hash}


# -- Read helpers -----------------------------------------------------------


@pytest.mark.asyncio
async def test_get_by_exact_content_is_case_insensitive(storage):
    content = "Mixed-Case content for Exact lookup."
    m = Memory(content=content, content_hash=generate_content_hash(content), tags=["exact"])
    await storage.store(m, skip_semantic_dedup=True)
    hits = await storage.get_by_exact_content("mixed-case content")
    assert len(hits) == 1
    assert hits[0].content_hash == m.content_hash

    assert await storage.get_by_exact_content("no such substring") == []


@pytest.mark.asyncio
async def test_get_all_memories_and_count(storage):
    for i in range(3):
        content = f"Distinct memory {i} — {chr(65 + i)}"
        m = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=[f"g-{i}"],
            memory_type="note",
        )
        await storage.store(m, skip_semantic_dedup=True)

    all_mems = await storage.get_all_memories()
    assert len(all_mems) == 3

    assert await storage.count_all_memories() == 3
    assert await storage.count_all_memories(tags=["g-1"]) == 1
    assert await storage.count_all_memories(memory_type="note") == 3
    assert await storage.count_all_memories(memory_type="reminder") == 0


@pytest.mark.asyncio
async def test_get_all_tags_deduplicates(storage):
    a = Memory(content="one", content_hash=generate_content_hash("one"), tags=["x", "y"])
    b = Memory(content="two", content_hash=generate_content_hash("two"), tags=["y", "z"])
    await storage.store(a, skip_semantic_dedup=True)
    await storage.store(b, skip_semantic_dedup=True)
    tags = await storage.get_all_tags()
    assert set(tags) == {"x", "y", "z"}


# -- Update / stats ---------------------------------------------------------


@pytest.mark.asyncio
async def test_update_memory_metadata(storage, sample_memory):
    await storage.store(sample_memory)
    ok, msg = await storage.update_memory_metadata(
        sample_memory.content_hash,
        updates={
            "tags": ["milvus", "updated"],
            "memory_type": "decision",
            "metadata": {"priority": "high"},
            "new_field": "extra",
        },
    )
    assert ok, msg

    refreshed = await storage.get_by_hash(sample_memory.content_hash)
    assert refreshed is not None
    assert set(refreshed.tags) == {"milvus", "updated"}
    assert refreshed.memory_type == "decision"
    assert refreshed.metadata["priority"] == "high"
    assert refreshed.metadata["new_field"] == "extra"


@pytest.mark.asyncio
async def test_update_rejects_bad_tag_type(storage, sample_memory):
    await storage.store(sample_memory)
    ok, msg = await storage.update_memory_metadata(
        sample_memory.content_hash, updates={"tags": "not-a-list"}
    )
    assert not ok
    assert "list of strings" in msg.lower()


@pytest.mark.asyncio
async def test_get_stats(storage, sample_memory):
    await storage.store(sample_memory)
    stats = await storage.get_stats()
    assert stats["backend"] == "milvus"
    assert stats["total_memories"] >= 1
    assert stats["embedding_dimension"] > 0
    assert stats["collection"] == storage.collection_name


# -- Semantic dedup ---------------------------------------------------------


@pytest.mark.asyncio
async def test_semantic_dedup_blocks_near_duplicate(milvus_db_path, monkeypatch):
    # Force dedup on for this test — the module-level conftest disables it.
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "true")
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_THRESHOLD", "0.85")
    name = f"mcp_memory_dedup_{uuid.uuid4().hex[:8]}"
    storage = MilvusMemoryStorage(
        uri=str(milvus_db_path),
        collection_name=name,
    )
    await storage.initialize()
    try:
        first = Memory(
            content="Claude Code is a powerful CLI tool for software engineering.",
            content_hash=generate_content_hash(
                "Claude Code is a powerful CLI tool for software engineering."
            ),
            tags=["tool"],
        )
        ok, _ = await storage.store(first)
        assert ok

        near = Memory(
            content="The Claude Code CLI is an excellent software development tool.",
            content_hash=generate_content_hash(
                "The Claude Code CLI is an excellent software development tool."
            ),
            tags=["tool"],
        )
        ok2, msg2 = await storage.store(near)
        assert not ok2
        assert "semantically similar" in msg2.lower()

        # With skip flag, the same near-duplicate lands.
        ok3, _ = await storage.store(near, skip_semantic_dedup=True)
        assert ok3
    finally:
        try:
            if storage.client is not None and storage.client.has_collection(name):
                storage.client.drop_collection(name)
        except Exception:
            pass
        await storage.close()


# -- cleanup_duplicates ------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_duplicates_noop(storage, sample_memory):
    # Milvus uses content_hash as the primary key, so duplicates can't exist.
    await storage.store(sample_memory)
    count, msg = await storage.cleanup_duplicates()
    assert count == 0
    assert "no duplicate" in msg.lower()


# -- Regression: channel survives across asyncio.run boundaries -------------


def test_store_survives_cross_loop_init(milvus_db_path):
    """Regression: the gRPC channel underpinning MilvusClient must survive
    being built in one ``asyncio.run`` call and reused in another.

    The MCP server builds storage from a stdio lifespan loop and then handles
    each tool call from a fresh ``asyncio.run``-style context. If pymilvus's
    sync channel were tied to a specific event loop (or torn down by garbage
    collection between loops) the second call would surface as ``Cannot
    invoke RPC on closed channel!``. We exercise that exact pattern here.
    """
    import uuid as _uuid
    from src.mcp_memory_service.utils.hashing import generate_content_hash as _ch

    name = f"mcp_xloop_{_uuid.uuid4().hex[:8]}"
    storage = MilvusMemoryStorage(uri=str(milvus_db_path), collection_name=name)
    try:
        import asyncio as _asyncio
        _asyncio.run(storage.initialize())

        for i in range(3):
            content = f"cross-loop store attempt {i}: fresh asyncio.run each time"

            async def _op(c=content):
                m = Memory(content=c, content_hash=_ch(c), tags=["xloop"])
                return await storage.store(m, skip_semantic_dedup=True)

            ok, msg = _asyncio.run(_op())
            assert ok, f"iteration {i} failed after channel rebuild: {msg}"
    finally:
        try:
            if storage.client is not None and storage.client.has_collection(name):
                storage.client.drop_collection(name)
        except Exception:
            pass
        import asyncio as _asyncio
        _asyncio.run(storage.close())


@pytest.mark.asyncio
async def test_store_reports_failure_on_closed_client(storage):
    """Regression: if the client has been explicitly closed, ``store`` must
    return ``(False, <message>)`` — never a silent fake success.

    Milvus Lite's daemon lifecycle is tied to the MilvusClient, so we do
    *not* recreate the client automatically. A closed/torn client is a
    visible failure mode the caller must see.
    """
    from src.mcp_memory_service.utils.hashing import generate_content_hash as _ch

    assert storage.client is not None
    storage.client.close()  # emulate a torn channel

    content = "this store must fail loudly, not silently succeed"
    m = Memory(content=content, content_hash=_ch(content), tags=["fail-loud"])
    ok, msg = await storage.store(m, skip_semantic_dedup=True)
    assert ok is False, f"closed-client store must fail, got ok={ok} msg={msg!r}"
    assert msg  # non-empty message


# -- Regression: MCP-style request spacing ---------------------------------


@pytest.mark.asyncio
async def test_three_calls_with_500ms_gaps(storage):
    """Regression: three storage operations separated by 500ms gaps — exactly
    the pattern exposed by the MCP stdio loop when each tool call arrives as
    a separate JSON-RPC request. All three must succeed; the client must
    remain live throughout; reads must see every write.
    """
    from src.mcp_memory_service.utils.hashing import generate_content_hash as _ch

    stored_hashes = []
    for idx in range(3):
        content = f"spaced-call {idx}: {uuid.uuid4().hex}"
        m = Memory(content=content, content_hash=_ch(content), tags=["spaced"])
        ok, msg = await storage.store(m, skip_semantic_dedup=True)
        assert ok, f"iteration {idx} failed: {msg}"
        assert storage.client is not None, f"client went None after iteration {idx}"
        stored_hashes.append(m.content_hash)
        await asyncio.sleep(0.5)

    # Every store must be readable after the gaps.
    for h in stored_hashes:
        got = await storage.get_by_hash(h)
        assert got is not None, f"lost memory {h} after 500ms gap reads"


@pytest.mark.asyncio
async def test_ten_stores_interleaved_with_sleep(storage):
    """Regression: 10 consecutive stores with a 100ms pause between each
    must all be persisted and all retrievable. Guards against silent data
    loss where ``store`` reports success but the write never lands.
    """
    from src.mcp_memory_service.utils.hashing import generate_content_hash as _ch

    planned = []
    for idx in range(10):
        content = f"interleaved-store-{idx} unique-token-{uuid.uuid4().hex[:6]}"
        planned.append((content, _ch(content)))

    for content, content_hash in planned:
        m = Memory(content=content, content_hash=content_hash, tags=["interleave"])
        ok, msg = await storage.store(m, skip_semantic_dedup=True)
        assert ok, f"store reported failure for '{content[:40]}': {msg}"
        await asyncio.sleep(0.1)

    # Now verify every single one is retrievable. count_all_memories must
    # match, and every hash must be fetchable.
    count = await storage.count_all_memories()
    assert count == len(planned), (
        f"silent data loss: stored {len(planned)} but count_all_memories reports {count}"
    )
    for _, content_hash in planned:
        got = await storage.get_by_hash(content_hash)
        assert got is not None, f"silent data loss: {content_hash} missing after store"


# -- Regression: end-to-end MCP invocation pattern --------------------------


@pytest.mark.asyncio
async def test_persists_across_sequential_calls(storage):
    """Regression: back-to-back stores in the same loop persist and are
    visible via retrieve.

    Matches the exact failure mode observed when driving the real MCP
    server — two stores would return success hashes but ``memory_search``
    reported 0 memories because the Milvus Lite daemon had been silently
    respawned between calls. With one ``MilvusClient`` per storage instance
    for life, two sequential stores must both land in the same daemon.
    """
    from src.mcp_memory_service.utils.hashing import generate_content_hash as _ch

    planned = ["probe A", "probe B"]
    for idx, content in enumerate(planned):
        m = Memory(
            content=content,
            content_hash=_ch(content),
            tags=["t"],
            memory_type="note",
        )
        ok, msg = await storage.store(m, skip_semantic_dedup=True)
        assert ok, f"store {idx} failed: {msg}"
        await asyncio.sleep(0.05)

    count = await storage.count_all_memories()
    assert count == len(planned), (
        f"silent data loss: expected {len(planned)} memories, got {count}"
    )

    results = await storage.retrieve("probe", n_results=3)
    assert len(results) == len(planned), (
        f"retrieve returned {len(results)} hits, expected {len(planned)}"
    )


@pytest.mark.asyncio
async def test_client_not_recreated_between_calls(storage):
    """Regression: the ``MilvusClient`` object identity must be stable
    across CRUD calls. Recreating the client mid-lifetime would spawn a
    fresh Milvus Lite daemon and orphan previously-stored data.
    """
    from src.mcp_memory_service.utils.hashing import generate_content_hash as _ch

    assert storage.client is not None
    initial_id = id(storage.client)

    content_a = "identity-check memory A"
    m = Memory(content=content_a, content_hash=_ch(content_a), tags=["id"])
    ok, _ = await storage.store(m, skip_semantic_dedup=True)
    assert ok
    assert id(storage.client) == initial_id, (
        "client identity changed after first store — a new MilvusClient was "
        "created and the Milvus Lite daemon was respawned"
    )

    got = await storage.get_by_hash(m.content_hash)
    assert got is not None
    assert id(storage.client) == initial_id, (
        "client identity changed after get_by_hash"
    )

    results = await storage.retrieve("identity", n_results=2)
    assert len(results) == 1
    assert id(storage.client) == initial_id, (
        "client identity changed after retrieve"
    )


@pytest.mark.asyncio
async def test_call_client_raises_when_not_initialized(milvus_db_path):
    """Regression: calling a CRUD method before ``initialize`` must raise a
    clear ``RuntimeError``, not silently create a new client."""
    name = f"not_init_{uuid.uuid4().hex[:8]}"
    storage = MilvusMemoryStorage(uri=str(milvus_db_path), collection_name=name)

    # Never called initialize(), so self.client is None.
    assert storage.client is None

    with pytest.raises(RuntimeError, match="not initialized"):
        await storage._call_client("has_collection", collection_name=name)


# -- Regression: Lite reconnect-on-dead-channel (upstream issue #334) -------


class _FlakyLiteClient:
    """Mock Milvus client that raises the exact closed-channel ValueError the
    first time any method is called, then behaves normally.

    Matches the error pymilvus / grpc surface when Milvus Lite's daemon
    subprocess has exited after idle (upstream milvus-lite #334).
    """

    def __init__(self, fail_once_with=None):
        self._fail_once = fail_once_with or ValueError(
            "Cannot invoke RPC on closed channel!"
        )
        self._raised = False
        self.call_count = 0

    def has_collection(self, **kwargs):
        self.call_count += 1
        if not self._raised:
            self._raised = True
            raise self._fail_once
        return True


@pytest.mark.asyncio
async def test_lite_reconnects_after_channel_death(milvus_db_path):
    """White-box: on Milvus Lite, a single "closed channel" error on an RPC
    must be swallowed by one reconnect + one retry. The second call runs
    against the freshly-built ``MilvusClient``, and ``id(storage.client)``
    must have changed exactly once.
    """
    name = f"lite_recon_{uuid.uuid4().hex[:8]}"
    storage = MilvusMemoryStorage(uri=str(milvus_db_path), collection_name=name)
    await storage.initialize()
    try:
        assert storage._is_lite is True

        # The newly-built client will be asked whether our collection exists —
        # make sure it does so the retry returns a meaningful value.
        real_client = storage.client
        assert real_client is not None
        assert real_client.has_collection(collection_name=name)

        # Install the flaky client and record its identity so we can confirm
        # the reconnect actually swapped the attribute.
        flaky = _FlakyLiteClient()
        storage.client = flaky
        original_flaky_id = id(flaky)

        result = await storage._call_client("has_collection", collection_name=name)

        assert result is True, "retry against reconnected client must return True"
        assert flaky.call_count == 1, (
            "old flaky client should only have been invoked once; the retry "
            f"must run against the new client, not the dead one (got {flaky.call_count})"
        )
        assert storage.client is not flaky, "reconnect did not swap self.client"
        assert id(storage.client) != original_flaky_id, "client identity should change after reconnect"
    finally:
        # Drop the mock client before close so close() doesn't touch it.
        storage.client = None


class _AlwaysFailsClient:
    """Mock client that raises the closed-channel ValueError on every call."""

    def __init__(self):
        self.call_count = 0

    def insert(self, **kwargs):
        self.call_count += 1
        raise ValueError("Cannot invoke RPC on closed channel!")


@pytest.mark.asyncio
async def test_remote_does_not_reconnect_on_channel_error():
    """A remote-URI storage must NOT reconnect on the same error signature.
    The exception propagates and ``id(storage.client)`` stays stable.

    We build the storage directly without ``initialize()`` so we can use a
    fake mock client and a fake remote URI — the test is purely about
    verifying the ``self._is_lite`` branch.
    """
    storage = MilvusMemoryStorage(
        uri="http://fake-remote:19530",
        collection_name="irrelevant",
    )
    assert storage._is_lite is False, "http:// URI must not be treated as Lite"

    fake = _AlwaysFailsClient()
    storage.client = fake
    stable_id = id(fake)
    # Pretend initialize() ran so the guard check passes.
    storage._initialized = True

    with pytest.raises(ValueError, match="closed channel"):
        await storage._call_client("insert", data=[{"id": "x"}])

    assert fake.call_count == 1, "remote path must invoke the client exactly once (no retry)"
    assert storage.client is fake, "remote path must not swap self.client"
    assert id(storage.client) == stable_id


# -- Review-follow-up tests (PR #721 review fixes) --------------------------


@pytest.mark.asyncio
async def test_get_by_exact_content_server_side_filter(storage, monkeypatch):
    """get_by_exact_content must push the filter down to Milvus rather than
    materialize every row into Python (previously it fetched up to 16384 rows
    and did substring matching in a loop).
    """
    from src.mcp_memory_service.utils.hashing import generate_content_hash as _ch

    # 20 distinct memories, one of which we'll search for by unique substring.
    for i in range(20):
        content = f"noise memory number {i} about unrelated subject"
        m = Memory(content=content, content_hash=_ch(content), tags=["noise"])
        await storage.store(m, skip_semantic_dedup=True)
    target_text = "unique needle memory — zxqvbn"
    target = Memory(
        content=target_text,
        content_hash=_ch(target_text),
        tags=["needle"],
    )
    await storage.store(target, skip_semantic_dedup=True)

    # Spy on _query_memories: the call must pass a filter that mentions
    # content_lower and narrows the scan at the Milvus layer.
    captured: dict = {}
    original = storage._query_memories

    async def _spy(*args, **kwargs):
        filter_expr = kwargs.get("filter_expr") or (args[0] if args else "")
        captured["filter_expr"] = filter_expr
        return await original(*args, **kwargs)

    monkeypatch.setattr(storage, "_query_memories", _spy)

    hits = await storage.get_by_exact_content("zxqvbn")

    assert len(hits) == 1
    assert hits[0].content_hash == target.content_hash
    assert "content_lower" in captured["filter_expr"]
    assert "zxqvbn" in captured["filter_expr"]


@pytest.mark.asyncio
async def test_get_by_exact_content_case_insensitive(storage):
    """Matches sqlite_vec's LIKE … COLLATE NOCASE semantics."""
    from src.mcp_memory_service.utils.hashing import generate_content_hash as _ch

    content = "Hello World case-sensitivity probe"
    m = Memory(content=content, content_hash=_ch(content), tags=["case"])
    await storage.store(m, skip_semantic_dedup=True)

    hits = await storage.get_by_exact_content("hello world")
    assert len(hits) == 1
    assert hits[0].content_hash == m.content_hash


@pytest.mark.asyncio
async def test_query_memories_returns_most_recent_first(storage):
    """Sorted pagination must work on the FULL matching set, not on whichever
    window Milvus happens to return first. Regression for the 16384-cap bug.
    """
    from src.mcp_memory_service.utils.hashing import generate_content_hash as _ch

    stored = []
    for i in range(5):
        content = f"ordering-test memory {i} unique-{uuid.uuid4().hex[:8]}"
        m = Memory(content=content, content_hash=_ch(content), tags=["order"])
        ok, _ = await storage.store(m, skip_semantic_dedup=True)
        assert ok
        stored.append(m.content_hash)
        # Distinct timestamps — created_at is a float so 100ms is enough.
        await asyncio.sleep(0.1)

    results = await storage.search_by_tag(["order"])
    assert len(results) == 5
    # Most recent stored first (search_by_tag sorts by created_at DESC).
    returned = [m.content_hash for m in results]
    assert returned == list(reversed(stored))

    # Now apply a tighter limit via get_all_memories and verify we still get
    # the two most recent, in correct order.
    top2 = await storage.get_all_memories(limit=2)
    assert [m.content_hash for m in top2] == list(reversed(stored))[:2]


@pytest.mark.asyncio
async def test_query_memories_pagination(storage):
    from src.mcp_memory_service.utils.hashing import generate_content_hash as _ch

    hashes = []
    for i in range(10):
        content = f"pagination memory {i} {uuid.uuid4().hex[:6]}"
        m = Memory(content=content, content_hash=_ch(content), tags=["page"])
        await storage.store(m, skip_semantic_dedup=True)
        hashes.append(m.content_hash)
        await asyncio.sleep(0.05)

    page1 = await storage.get_all_memories(limit=3, offset=0)
    page2 = await storage.get_all_memories(limit=3, offset=3)

    assert len(page1) == 3
    assert len(page2) == 3
    # No overlap between pages.
    assert set(m.content_hash for m in page1).isdisjoint(
        m.content_hash for m in page2
    )
    # Ordering consistent across pages — most recent first.
    expected_desc = list(reversed(hashes))
    assert [m.content_hash for m in page1] == expected_desc[:3]
    assert [m.content_hash for m in page2] == expected_desc[3:6]


def test_memory_to_entity_timestamps_consistent(milvus_db_path):
    """When created_at / created_at_iso are unset, the entity's epoch and ISO
    timestamps must reference the same instant. Regression for the bug where
    they were derived from two separate clock reads.
    """
    from datetime import datetime as _dt, timezone as _tz
    from src.mcp_memory_service.utils.hashing import generate_content_hash as _ch

    storage = MilvusMemoryStorage(
        uri=str(milvus_db_path),
        collection_name=f"ts_{uuid.uuid4().hex[:8]}",
    )
    # We don't call initialize() — _memory_to_entity is pure and just needs an
    # embedding to be supplied.

    content = "timestamp-consistency probe"
    mem = Memory(content=content, content_hash=_ch(content), tags=["ts"])
    # Memory.__post_init__ auto-populates timestamps — null them out so we
    # exercise the "both fields missing" fallback path in _resolve_timestamps.
    mem.created_at = None
    mem.created_at_iso = None
    mem.updated_at = None
    mem.updated_at_iso = None

    entity = storage._memory_to_entity(mem, embedding=[0.0] * 4)

    created_epoch = float(entity["created_at"])
    created_iso = entity["created_at_iso"]
    parsed = _dt.fromisoformat(created_iso).astimezone(_tz.utc).timestamp()
    assert abs(parsed - created_epoch) < 1.0, (
        f"created_at and created_at_iso refer to different moments "
        f"(epoch={created_epoch}, iso->epoch={parsed})"
    )

    updated_epoch = float(entity["updated_at"])
    updated_iso = entity["updated_at_iso"]
    parsed_u = _dt.fromisoformat(updated_iso).astimezone(_tz.utc).timestamp()
    assert abs(parsed_u - updated_epoch) < 1.0


def test_embedding_cache_does_not_collide():
    """Cache must key on the full text, not ``hash(text)``. Different strings
    must produce (and return) different embeddings — covers the silent-wrong-
    embedding risk of 64-bit hash collisions.
    """
    from src.mcp_memory_service.storage import milvus as milvus_mod

    # Reset cache to isolate this test from others in the session.
    with milvus_mod._EMBEDDING_CACHE_LOCK:
        milvus_mod._EMBEDDING_CACHE.clear()

    key_a = "model::hello"
    key_b = "model::world"
    milvus_mod._embedding_cache_put(key_a, [1.0, 2.0, 3.0])
    milvus_mod._embedding_cache_put(key_b, [4.0, 5.0, 6.0])

    assert milvus_mod._embedding_cache_get(key_a) == [1.0, 2.0, 3.0]
    assert milvus_mod._embedding_cache_get(key_b) == [4.0, 5.0, 6.0]
    # Distinct keys → distinct embeddings, not shared cells.
    assert milvus_mod._embedding_cache_get(key_a) is not milvus_mod._embedding_cache_get(key_b)


def test_embedding_cache_bounded():
    """Inserting 2000 unique entries must not push the cache past its max
    size — the LRU evicts the oldest.
    """
    from src.mcp_memory_service.storage import milvus as milvus_mod

    with milvus_mod._EMBEDDING_CACHE_LOCK:
        milvus_mod._EMBEDDING_CACHE.clear()

    max_size = milvus_mod._EMBEDDING_CACHE_MAX
    for i in range(2000):
        milvus_mod._embedding_cache_put(f"model::entry-{i}", [float(i)])

    assert milvus_mod._embedding_cache_size() <= max_size
    # Oldest entries should have been evicted.
    assert milvus_mod._embedding_cache_get("model::entry-0") is None
    # Newest entries should still be resident.
    assert milvus_mod._embedding_cache_get("model::entry-1999") == [1999.0]
