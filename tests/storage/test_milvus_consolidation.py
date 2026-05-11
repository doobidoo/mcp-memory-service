"""End-to-end integration tests for Milvus consolidation.

These tests use Milvus Lite (embedded single-file database) to verify the
full read path: store memories, run the DreamInspiredConsolidator, and
confirm the clustering / association stages actually receive populated
embeddings now that PR #878 + #881 + the current PR are wired together.

The module skips cleanly if ``pymilvus`` / ``milvus-lite`` are not
installed, matching the pattern already used in ``test_milvus.py``.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import pytest_asyncio

# Skip the whole module if pymilvus / milvus-lite are not installed.
pymilvus = pytest.importorskip("pymilvus")
milvus_lite = pytest.importorskip("milvus_lite")
# Embedding model is required for actually populating the vector column.
# Skip the whole module if sentence-transformers isn't available in the
# test environment (e.g. minimal local dev setups without the model
# cache). CI images that pin the Milvus extra will have it installed.
sentence_transformers = pytest.importorskip("sentence_transformers")

from src.mcp_memory_service.consolidation.base import ConsolidationConfig  # noqa: E402
from src.mcp_memory_service.consolidation.consolidator import (  # noqa: E402
    DreamInspiredConsolidator,
)
from src.mcp_memory_service.models.memory import Memory  # noqa: E402
from src.mcp_memory_service.storage.milvus import MilvusMemoryStorage  # noqa: E402
from src.mcp_memory_service.utils.hashing import generate_content_hash  # noqa: E402


@pytest.fixture(autouse=True)
def _offline_model_env(monkeypatch):
    """Force Hugging Face offline mode (mirrors the pattern in test_milvus.py)."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


@pytest.fixture(scope="module")
def milvus_db_path(tmp_path_factory):
    return tmp_path_factory.mktemp("milvus_consolidation") / "milvus.db"


@pytest_asyncio.fixture
async def storage(milvus_db_path):
    """Per-test storage with a fresh collection inside a shared Lite DB."""
    collection_name = f"mcp_cons_{uuid.uuid4().hex[:12]}"
    instance = MilvusMemoryStorage(
        uri=str(milvus_db_path),
        collection_name=collection_name,
    )
    await instance.initialize()
    try:
        yield instance
    finally:
        try:
            if instance.client is not None and instance.client.has_collection(
                collection_name
            ):
                instance.client.drop_collection(collection_name)
        except Exception:
            pass
        await instance.close()


# -- get_all_memories / get_memories_by_time_range with embeddings ---------


@pytest.mark.asyncio
async def test_get_all_memories_default_does_not_hydrate(storage):
    """CRUD default: embeddings are NOT loaded, preserving current perf."""
    m = Memory(
        content="default CRUD read should not fetch vector",
        content_hash=generate_content_hash("default CRUD read should not fetch vector"),
        tags=["crud"],
    )
    ok, _ = await storage.store(m)
    assert ok

    results = await storage.get_all_memories()
    assert len(results) == 1
    assert results[0].embedding is None, (
        "get_all_memories() default must NOT return embeddings"
    )


@pytest.mark.asyncio
async def test_get_all_memories_opt_in_hydrates(storage):
    """Consolidation opt-in: embedding is populated with len == embedding_dimension."""
    m = Memory(
        content="consolidation opt-in should fetch vector",
        content_hash=generate_content_hash(
            "consolidation opt-in should fetch vector"
        ),
        tags=["consolidation"],
    )
    ok, _ = await storage.store(m)
    assert ok

    results = await storage.get_all_memories(include_embeddings=True)
    assert len(results) == 1
    got = results[0]
    assert got.embedding is not None, "expected embedding hydrated, got None"
    assert isinstance(got.embedding, list)
    assert len(got.embedding) == storage.embedding_dimension


@pytest.mark.asyncio
async def test_get_memories_by_time_range_opt_in_hydrates(storage):
    """Time-range read + opt-in also hydrates embeddings."""
    m = Memory(
        content="time-range opt-in should fetch vector",
        content_hash=generate_content_hash(
            "time-range opt-in should fetch vector"
        ),
        tags=["time-range"],
    )
    ok, _ = await storage.store(m)
    assert ok

    results = await storage.get_memories_by_time_range(
        0.0, 9_999_999_999.0, include_embeddings=True,
    )
    assert len(results) == 1
    assert results[0].embedding is not None
    assert len(results[0].embedding) == storage.embedding_dimension


@pytest.mark.asyncio
async def test_round_trip_embedding_stable_within_tolerance(storage):
    """The embedding stored via store() round-trips back within 1e-4 per
    component via the consolidation read path.

    Storage is float32 on the Milvus side, so a small-but-nonzero delta is
    expected; anything wider than 1e-4 indicates a conversion bug.
    """
    m = Memory(
        content="round-trip stability — do not drift",
        content_hash=generate_content_hash("round-trip stability — do not drift"),
        tags=["roundtrip"],
    )
    # Generate the embedding the same way storage.store() will, so we can
    # compare against the value that got written.
    written = storage._generate_embedding(m.content)
    ok, _ = await storage.store(m)
    assert ok

    results = await storage.get_all_memories(include_embeddings=True)
    assert len(results) == 1
    read_back = results[0].embedding
    assert read_back is not None
    assert len(read_back) == len(written)
    for i, (a, b) in enumerate(zip(written, read_back)):
        assert abs(a - b) < 1e-4, (
            f"round-trip drift too large at component {i}: "
            f"written={a!r} read={b!r} delta={abs(a - b)!r}"
        )


# -- End-to-end consolidator run: clusters_created >= 1 --------------------


@pytest.mark.asyncio
async def test_consolidator_produces_at_least_one_cluster(storage):
    """End-to-end proof that the Milvus consolidation read path now works.

    Seeds two clearly distinct topical groups (Python programming vs Italian
    cooking) and runs the ``monthly`` consolidation horizon. Before this
    feature landed, the clustering stage warned "Only 0 memories have
    embeddings" and returned an empty cluster list; after the fix, at
    least one cluster must form. This test directly disproves the
    production failure mode reported on the live Milvus deployment.
    """
    # Five programming memories + five cooking memories. Keep them short
    # and distinct so DBSCAN has an easy time separating the two groups
    # using the default min_cluster_size.
    programming = [
        "Python async functions use await to pause coroutines",
        "Python dataclasses reduce boilerplate for small record types",
        "Python list comprehensions are more idiomatic than map and filter",
        "Python type hints are checked by mypy but not at runtime",
        "Python generators yield values lazily without building full lists",
    ]
    cooking = [
        "Classic carbonara uses guanciale and pecorino, never cream",
        "Italian pasta water should be as salty as the Mediterranean sea",
        "Risotto alla Milanese gets its yellow color from saffron strands",
        "Authentic Neapolitan pizza dough rests at room temperature overnight",
        "Pesto alla Genovese is crushed in a mortar, not blended in a mixer",
    ]
    for text in programming + cooking:
        m = Memory(
            content=text,
            content_hash=generate_content_hash(text),
            tags=["seed"],
        )
        ok, _ = await storage.store(m, skip_semantic_dedup=True)
        assert ok, f"failed to seed memory: {text!r}"

    config = ConsolidationConfig(min_cluster_size=2)
    consolidator = DreamInspiredConsolidator(storage, config)
    report = await consolidator.consolidate("monthly")

    assert report.errors == [], (
        f"consolidation reported errors: {report.errors!r}"
    )
    assert report.memories_processed == 10, (
        f"expected 10 seeded memories, got {report.memories_processed}"
    )
    assert report.clusters_created >= 1, (
        "Milvus consolidation must form at least one cluster now that "
        "embeddings are hydrated on the read path. This assertion directly "
        "disproves the production log 'Only 0 memories have embeddings'."
    )
