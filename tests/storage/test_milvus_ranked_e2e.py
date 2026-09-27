"""End-to-end test for Milvus ranked mode against a REAL Milvus Lite backend.

Unlike test_milvus_search_methods.py (mock-based), this spins up an actual
Milvus Lite file + real sentence-transformers embeddings and exercises the
full search_memories path. This is the test that would have caught the
ranking_weights TypeError in production (commit b4d2723a, #1028), since the
mock tests stubbed out the very layer where the kwarg was dropped.

Run: pytest tests/storage/test_milvus_ranked_e2e.py -v
"""

import os
import uuid

import pytest
import pytest_asyncio

pymilvus = pytest.importorskip("pymilvus")
milvus_lite = pytest.importorskip("milvus_lite")
pytest.importorskip("sentence_transformers")

from mcp_memory_service.models.memory import Memory  # noqa: E402
from mcp_memory_service.storage.milvus import MilvusMemoryStorage  # noqa: E402
from mcp_memory_service.utils.hashing import generate_content_hash  # noqa: E402

# Local cached model — small, offline-friendly.
_E2E_MODEL = "all-MiniLM-L6-v2"


@pytest.fixture(autouse=True)
def _offline_model_env(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


@pytest.fixture(scope="module")
def milvus_db_path(tmp_path_factory):
    return tmp_path_factory.mktemp("milvus_ranked_e2e") / "milvus.db"


@pytest_asyncio.fixture
async def storage(milvus_db_path):
    collection_name = f"mcp_ranked_{uuid.uuid4().hex[:12]}"
    instance = MilvusMemoryStorage(
        uri=str(milvus_db_path),
        collection_name=collection_name,
        embedding_model=_E2E_MODEL,
    )
    await instance.initialize()
    # Seed a handful of memories with distinct topics.
    seeds = [
        ("Python async programming with asyncio and await", ["python", "async"]),
        ("FastAPI dependency injection and routers", ["python", "fastapi"]),
        ("Redis caching strategies for high throughput", ["redis", "cache"]),
        ("Kubernetes deployment rollout and rollback", ["k8s", "devops"]),
        ("Vector database similarity search with embeddings", ["vector", "search"]),
    ]
    for content, tags in seeds:
        await instance.store(
            Memory(
                content=content,
                content_hash=generate_content_hash(content),
                tags=tags,
                memory_type="note",
            )
        )
    try:
        yield instance
    finally:
        try:
            if instance.client is not None and instance.client.has_collection(collection_name):
                instance.client.drop_collection(collection_name)
        except Exception:
            pass
        await instance.close()


@pytest.mark.asyncio
async def test_semantic_with_ranking_weights_no_typeerror(storage):
    """Regression: handler always passes ranking_weights — must not crash."""
    result = await storage.search_memories(
        query="python web framework",
        mode="semantic",
        ranking_weights={"semantic": 0.6, "time_decay": 0.2},
    )
    assert "error" not in result
    assert result["mode"] == "semantic"
    assert result["total"] >= 1


@pytest.mark.asyncio
async def test_ranked_mode_returns_results(storage):
    """ranked mode against real Milvus Lite returns ranked memories."""
    result = await storage.search_memories(query="python async", mode="ranked", limit=5)
    assert "error" not in result
    assert result["mode"] == "ranked"
    assert result["total"] >= 1
    # Most relevant memory should mention python/async.
    top = result["memories"][0]["content"].lower()
    assert "python" in top or "async" in top or "fastapi" in top


@pytest.mark.asyncio
async def test_ranked_mode_with_custom_weights(storage):
    """ranked mode honors a custom ranking_weights dict end-to-end."""
    result = await storage.search_memories(
        query="caching",
        mode="ranked",
        ranking_weights={"semantic": 1.0, "time_decay": 0.0,
                         "access_frequency": 0.0, "quality": 0.0},
        limit=3,
    )
    assert "error" not in result
    assert result["total"] >= 1


@pytest.mark.asyncio
async def test_ranked_mode_with_tag_filter(storage):
    """ranked + tag filter narrows results to the tagged subset."""
    result = await storage.search_memories(
        query="programming",
        mode="ranked",
        tags=["python"],
        limit=10,
    )
    assert "error" not in result
    # Every returned memory must carry the python tag (server-side filter).
    for mem in result["memories"]:
        assert "python" in mem["tags"]


@pytest.mark.asyncio
async def test_ranked_debug_breakdown_present(storage):
    """ranked rerank attaches multi-signal breakdown to debug_info."""
    result = await storage.search_memories(
        query="kubernetes", mode="ranked", include_debug=True, limit=3
    )
    assert "error" not in result
    assert result["total"] >= 1
    # apply_ranked_rerank stamps each memory's debug_info with ranked signals.
    top = result["memories"][0]
    assert "debug_info" in top
    assert top["debug_info"].get("ranked") is True
    assert "ranked_score" in top["debug_info"]


@pytest.mark.asyncio
async def test_ranked_requires_query_e2e(storage):
    """ranked without query returns an error, not a crash."""
    result = await storage.search_memories(query=None, mode="ranked")
    assert "error" in result
    assert result["total"] == 0
