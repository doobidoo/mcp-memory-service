"""
Integration tests for graph traversal MCP tools.

Tests the three graph traversal tools:
- find_connected_memories: Multi-hop connection discovery
- find_shortest_path: Path finding between memories
- get_memory_subgraph: Subgraph extraction for visualization
"""
import json
import uuid
from functools import wraps

import pytest
import pytest_asyncio
from mcp_memory_service.server import MemoryServer
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.graph import GraphStorage
from mcp_memory_service.config import SQLITE_VEC_PATH, STORAGE_BACKEND


@pytest_asyncio.fixture
async def memory_server():
    """Create a test instance of the memory server."""
    server = MemoryServer()
    yield server


@pytest_asyncio.fixture
async def graph_storage():
    """Create a test instance of graph storage with table initialization."""
    if STORAGE_BACKEND not in ['sqlite_vec', 'hybrid']:
        pytest.skip(f"Graph operations not supported for backend: {STORAGE_BACKEND}")
    graph = GraphStorage(SQLITE_VEC_PATH)

    # Create memory_graph table if it doesn't exist
    conn = await graph._get_connection()
    async with graph._lock:
        cursor = conn.cursor()
        try:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_graph (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_hash TEXT NOT NULL,
                    target_hash TEXT NOT NULL,
                    similarity REAL DEFAULT 0.0,
                    connection_types TEXT,
                    metadata TEXT,
                    created_at REAL DEFAULT (unixepoch()),
                    UNIQUE(source_hash, target_hash)
                )
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_memory_graph_source ON memory_graph(source_hash)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_memory_graph_target ON memory_graph(target_hash)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_memory_graph_similarity ON memory_graph(similarity)
            ''')
            conn.commit()
        finally:
            cursor.close()

    yield graph
    await graph.close()


@pytest_asyncio.fixture
async def setup_graph_data(memory_server, graph_storage):
    """
    Set up test data with associations for graph traversal.

    Creates a small graph:
    A -> B -> C
    A -> D
    B -> D
    """
    test_id = str(uuid.uuid4())[:8]  # Short unique ID for this test run

    # Store test memories with unique content
    mem_a = await memory_server.store_memory(
        content=f"Graph test memory A {test_id}",
        metadata={"tags": ["test", f"graph-test-{test_id}"]}
    )
    mem_b = await memory_server.store_memory(
        content=f"Graph test memory B {test_id}",
        metadata={"tags": ["test", f"graph-test-{test_id}"]}
    )
    mem_c = await memory_server.store_memory(
        content=f"Graph test memory C {test_id}",
        metadata={"tags": ["test", f"graph-test-{test_id}"]}
    )
    mem_d = await memory_server.store_memory(
        content=f"Graph test memory D {test_id}",
        metadata={"tags": ["test", f"graph-test-{test_id}"]}
    )

    # Extract hashes with error handling
    assert mem_a.get("success"), f"Failed to store mem_a: {mem_a}"
    assert mem_b.get("success"), f"Failed to store mem_b: {mem_b}"
    assert mem_c.get("success"), f"Failed to store mem_c: {mem_c}"
    assert mem_d.get("success"), f"Failed to store mem_d: {mem_d}"

    hash_a = mem_a["hash"]
    hash_b = mem_b["hash"]
    hash_c = mem_c["hash"]
    hash_d = mem_d["hash"]

    # Create associations
    await graph_storage.store_association(
        hash_a, hash_b, 0.8, ["semantic"], {"test": True}
    )
    await graph_storage.store_association(
        hash_b, hash_c, 0.7, ["semantic"], {"test": True}
    )
    await graph_storage.store_association(
        hash_a, hash_d, 0.6, ["semantic"], {"test": True}
    )
    await graph_storage.store_association(
        hash_b, hash_d, 0.5, ["semantic"], {"test": True}
    )

    return {
        "hash_a": hash_a,
        "hash_b": hash_b,
        "hash_c": hash_c,
        "hash_d": hash_d
    }


@pytest.mark.asyncio
async def test_find_connected_memories_valid(memory_server, setup_graph_data):
    """Test find_connected_memories with valid input."""
    if STORAGE_BACKEND not in ['sqlite_vec', 'hybrid']:
        pytest.skip(f"Graph operations not supported for backend: {STORAGE_BACKEND}")

    hashes = setup_graph_data
    hash_a = hashes["hash_a"]

    # Call the MCP tool handler
    result = await memory_server.handle_find_connected_memories({
        "hash": hash_a,
        "max_hops": 2
    })

    # Parse JSON response
    assert len(result) == 1
    response = json.loads(result[0].text)

    # Verify response structure
    assert response["success"] is True
    assert "connected" in response
    assert "count" in response
    assert response["count"] > 0

    # Should find B, C, and D within 2 hops from A
    connected_hashes = {item["hash"] for item in response["connected"]}
    assert hashes["hash_b"] in connected_hashes
    assert hashes["hash_d"] in connected_hashes
    # C should be reachable in 2 hops
    assert hashes["hash_c"] in connected_hashes


@pytest.mark.asyncio
async def test_find_connected_memories_missing_hash(memory_server):
    """Test find_connected_memories with missing hash parameter."""
    if STORAGE_BACKEND not in ['sqlite_vec', 'hybrid']:
        pytest.skip(f"Graph operations not supported for backend: {STORAGE_BACKEND}")

    result = await memory_server.handle_find_connected_memories({})

    assert len(result) == 1
    response = json.loads(result[0].text)
    assert response["success"] is False
    assert "error" in response
    assert "Missing required parameter: hash" in response["error"]


@pytest.mark.asyncio
async def test_find_connected_memories_no_connections(memory_server):
    """Test find_connected_memories with isolated memory."""
    if STORAGE_BACKEND not in ['sqlite_vec', 'hybrid']:
        pytest.skip(f"Graph operations not supported for backend: {STORAGE_BACKEND}")

    # Store isolated memory
    mem = await memory_server.store_memory(content="Isolated memory")
    hash_isolated = mem["hash"]

    result = await memory_server.handle_find_connected_memories({
        "hash": hash_isolated,
        "max_hops": 2
    })

    assert len(result) == 1
    response = json.loads(result[0].text)
    assert response["success"] is True
    assert response["count"] == 0
    assert len(response["connected"]) == 0


@pytest.mark.asyncio
async def test_find_shortest_path_valid(memory_server, setup_graph_data):
    """Test find_shortest_path with valid input."""
    if STORAGE_BACKEND not in ['sqlite_vec', 'hybrid']:
        pytest.skip(f"Graph operations not supported for backend: {STORAGE_BACKEND}")

    hashes = setup_graph_data
    hash_a = hashes["hash_a"]
    hash_c = hashes["hash_c"]

    # Find path from A to C
    result = await memory_server.handle_find_shortest_path({
        "hash1": hash_a,
        "hash2": hash_c,
        "max_depth": 5
    })

    assert len(result) == 1
    response = json.loads(result[0].text)

    assert response["success"] is True
    assert response["path"] is not None
    assert response["length"] > 0

    # Path should be A -> B -> C (length 3)
    path = response["path"]
    assert path[0] == hash_a
    assert path[-1] == hash_c
    assert len(path) == 3


@pytest.mark.asyncio
async def test_find_shortest_path_no_path(memory_server, setup_graph_data):
    """Test find_shortest_path when no path exists."""
    if STORAGE_BACKEND not in ['sqlite_vec', 'hybrid']:
        pytest.skip(f"Graph operations not supported for backend: {STORAGE_BACKEND}")

    hashes = setup_graph_data
    hash_a = hashes["hash_a"]

    # Create isolated memory
    mem_isolated = await memory_server.store_memory(content="Isolated")
    hash_isolated = mem_isolated["hash"]

    result = await memory_server.handle_find_shortest_path({
        "hash1": hash_a,
        "hash2": hash_isolated,
        "max_depth": 5
    })

    assert len(result) == 1
    response = json.loads(result[0].text)

    assert response["success"] is True
    assert response["path"] is None
    assert response["length"] == 0


@pytest.mark.asyncio
async def test_find_shortest_path_missing_params(memory_server):
    """Test find_shortest_path with missing parameters."""
    if STORAGE_BACKEND not in ['sqlite_vec', 'hybrid']:
        pytest.skip(f"Graph operations not supported for backend: {STORAGE_BACKEND}")

    # Missing hash2
    result = await memory_server.handle_find_shortest_path({
        "hash1": "abc123"
    })

    assert len(result) == 1
    response = json.loads(result[0].text)
    assert response["success"] is False
    assert "error" in response


@pytest.mark.asyncio
async def test_get_memory_subgraph_valid(memory_server, setup_graph_data):
    """Test get_memory_subgraph with valid input."""
    if STORAGE_BACKEND not in ['sqlite_vec', 'hybrid']:
        pytest.skip(f"Graph operations not supported for backend: {STORAGE_BACKEND}")

    hashes = setup_graph_data
    hash_a = hashes["hash_a"]

    result = await memory_server.handle_get_memory_subgraph({
        "hash": hash_a,
        "radius": 2
    })

    assert len(result) == 1
    response = json.loads(result[0].text)

    assert response["success"] is True
    assert "nodes" in response
    assert "edges" in response
    assert "node_count" in response
    assert "edge_count" in response

    # Should include center node plus connected nodes
    assert hash_a in response["nodes"]
    assert response["node_count"] > 1
    assert response["edge_count"] > 0

    # Verify edges have required fields
    if response["edges"]:
        edge = response["edges"][0]
        assert "source" in edge
        assert "target" in edge
        assert "similarity" in edge
        assert "connection_types" in edge
        assert "metadata" in edge


@pytest.mark.asyncio
async def test_get_memory_subgraph_isolated(memory_server):
    """Test get_memory_subgraph with isolated memory."""
    if STORAGE_BACKEND not in ['sqlite_vec', 'hybrid']:
        pytest.skip(f"Graph operations not supported for backend: {STORAGE_BACKEND}")

    # Store isolated memory with unique content
    test_id = str(uuid.uuid4())[:8]
    mem = await memory_server.store_memory(
        content=f"Isolated graph test memory {test_id}"
    )
    assert mem.get("success"), f"Failed to store isolated memory: {mem}"
    hash_isolated = mem["hash"]

    result = await memory_server.handle_get_memory_subgraph({
        "hash": hash_isolated,
        "radius": 2
    })

    assert len(result) == 1
    response = json.loads(result[0].text)

    assert response["success"] is True
    assert response["node_count"] == 1  # Only center node
    assert response["edge_count"] == 0  # No connections
    assert hash_isolated in response["nodes"]


@pytest.mark.asyncio
async def test_get_memory_subgraph_missing_hash(memory_server):
    """Test get_memory_subgraph with missing hash parameter."""
    if STORAGE_BACKEND not in ['sqlite_vec', 'hybrid']:
        pytest.skip(f"Graph operations not supported for backend: {STORAGE_BACKEND}")

    result = await memory_server.handle_get_memory_subgraph({})

    assert len(result) == 1
    response = json.loads(result[0].text)
    assert response["success"] is False
    assert "error" in response


@pytest.mark.asyncio
async def test_graph_tools_cloudflare_graceful_fallback(memory_server):
    """Test that graph tools gracefully handle cloudflare backend."""
    if STORAGE_BACKEND != "cloudflare":
        pytest.skip("Test only applicable for cloudflare backend")

    # Test find_connected_memories
    result = await memory_server.handle_find_connected_memories({
        "hash": "test_hash"
    })
    response = json.loads(result[0].text)
    assert response["success"] is False
    assert "Graph operations not available" in response["error"]

    # Test find_shortest_path
    result = await memory_server.handle_find_shortest_path({
        "hash1": "test1",
        "hash2": "test2"
    })
    response = json.loads(result[0].text)
    assert response["success"] is False
    assert "Graph operations not available" in response["error"]

    # Test get_memory_subgraph
    result = await memory_server.handle_get_memory_subgraph({
        "hash": "test_hash"
    })
    response = json.loads(result[0].text)
    assert response["success"] is False
    assert "Graph operations not available" in response["error"]


# ---------------------------------------------------------------------------
# Two-phase query skeleton (#56/#61): normalize_entity_id + handler contracts
# ---------------------------------------------------------------------------

class TestNormalizeEntityId:
    """Canonical entity_id normalization — deterministic + consolidation-stable."""

    def test_ascii_slug(self):
        from mcp_memory_service.storage.graph import normalize_entity_id
        assert normalize_entity_id("Hello World") == "hello-world"
        assert normalize_entity_id("FastAPI") == "fastapi"
        assert normalize_entity_id("  spaced  out  ") == "spaced-out"
        assert normalize_entity_id("foo/bar_baz.qux") == "foo-bar-baz-qux"

    def test_accented_decomposes_to_ascii(self):
        from mcp_memory_service.storage.graph import normalize_entity_id
        assert normalize_entity_id("Café") == "cafe"
        assert normalize_entity_id("Müller") == "muller"
        assert normalize_entity_id("naïve résumé") == "naive-resume"

    def test_cjk_and_non_latin_fall_back_to_hash(self):
        from mcp_memory_service.storage.graph import normalize_entity_id
        for name in ("记忆服务", "Привет", "Ελληνικά", "مرحبا"):
            slug = normalize_entity_id(name)
            assert slug.startswith("e-"), f"{name!r} -> {slug!r}"
            assert len(slug) == 10  # 'e-' + 8 hex chars

    def test_deterministic_and_stable(self):
        from mcp_memory_service.storage.graph import normalize_entity_id
        assert normalize_entity_id("记忆服务") == normalize_entity_id("记忆服务")
        assert normalize_entity_id("Café") == normalize_entity_id("Café")

    def test_collision_same_surface_forms_share_suffix(self):
        """Distinct surface forms that slugify identically get the same id —
        callers append the same short hash suffix to disambiguate."""
        from mcp_memory_service.storage.graph import normalize_entity_id
        assert normalize_entity_id("Hello World") == normalize_entity_id("hello   world")

    def test_empty_and_none_safe(self):
        from mcp_memory_service.storage.graph import normalize_entity_id
        assert normalize_entity_id("").startswith("e-")
        assert normalize_entity_id(None).startswith("e-")


@pytest.mark.asyncio
async def test_memory_explore_missing_query(memory_server):
    """memory_explore validates required 'query' before touching storage."""
    from mcp_memory_service.server.handlers.graph import handle_memory_explore
    result = await handle_memory_explore(memory_server, {})
    assert len(result) == 1
    response = json.loads(result[0].text)
    assert response["success"] is False
    assert "Missing required parameter: query" in response["error"]
    assert response["entities"] == []


@pytest.mark.asyncio
async def test_memory_explore_returns_contract_shape(memory_server, setup_graph_data):
    """memory_explore returns the locked response shape (entities/count)."""
    if STORAGE_BACKEND not in ['sqlite_vec', 'hybrid']:
        pytest.skip(f"Graph operations not supported for backend: {STORAGE_BACKEND}")
    from mcp_memory_service.server.handlers.graph import handle_memory_explore
    result = await handle_memory_explore(memory_server, {"query": "graph test memory"})
    assert len(result) == 1
    response = json.loads(result[0].text)
    assert "entities" in response
    assert "count" in response
    assert isinstance(response["entities"], list)
    assert response["count"] == len(response["entities"])
    # Every entity (if any) must carry the full contract keys.
    for ent in response["entities"]:
        assert set(ent) >= {
            "entity_id", "name", "entity_type", "summary",
            "relation_count", "top_chunks",
        }
        assert isinstance(ent["top_chunks"], list)


@pytest.mark.asyncio
async def test_memory_explore_all_store_retrieves_global_candidates(
    memory_server, setup_graph_data, monkeypatch
):
    """The explicit all-store sentinel must use an unscoped retrieval."""
    if STORAGE_BACKEND not in ['sqlite_vec', 'hybrid']:
        pytest.skip(f"Graph operations not supported for backend: {STORAGE_BACKEND}")
    from mcp_memory_service.server.handlers.graph import handle_memory_explore

    memory = Memory(
        content=f"All-store explore candidate {uuid.uuid4().hex}",
        content_hash=f"all-store-{uuid.uuid4().hex}",
        tags=["all-store"],
    )
    stored, _ = await memory_server.storage.store(memory, store="docs")
    assert stored is True
    original_retrieve = memory_server.storage.retrieve
    observed = {}

    @wraps(original_retrieve)
    async def track_retrieve(*args, **kwargs):
        results = await original_retrieve(*args, **kwargs)
        observed["store_passed"] = "store" in kwargs
        observed["store"] = kwargs.get("store")
        observed["hashes"] = [result.memory.content_hash for result in results]
        return results

    monkeypatch.setattr(memory_server.storage, "retrieve", track_retrieve)
    await handle_memory_explore(
        memory_server, {"query": memory.content, "store": "all"}
    )

    assert observed["store_passed"] is True
    assert observed["store"] is None
    assert memory.content_hash in observed["hashes"]


@pytest.mark.asyncio
async def test_memory_detail_missing_entity_id(memory_server):
    """memory_detail validates required 'entity_id' before touching the graph."""
    from mcp_memory_service.server.handlers.graph import handle_memory_detail
    result = await handle_memory_detail(memory_server, {})
    assert len(result) == 1
    response = json.loads(result[0].text)
    assert response["success"] is False
    assert "Missing required parameter: entity_id" in response["error"]
    assert response["chunks"] == []
    assert response["related_entities"] == []


@pytest.mark.asyncio
async def test_retrieve_candidates_forwards_store_when_supported():
    """_retrieve_candidates forwards `store` when the backend retrieve() accepts
    it (composes with multi-store scoping, #62)."""
    from mcp_memory_service.server.handlers.graph import _retrieve_candidates

    class StoreAwareStorage:
        def __init__(self):
            self.kwargs = None

        async def retrieve(self, query, n_results=5, tags=None, store="default"):
            self.kwargs = {"query": query, "n_results": n_results, "tags": tags, "store": store}
            return []

    s = StoreAwareStorage()
    await _retrieve_candidates(s, "q", n_results=3, tags=["t"], store="docs")
    assert s.kwargs["store"] == "docs"
    assert s.kwargs["n_results"] == 3


@pytest.mark.asyncio
async def test_retrieve_candidates_omits_store_when_unsupported():
    """A backend whose retrieve() lacks `store` must not receive it (no crash)."""
    from mcp_memory_service.server.handlers.graph import _retrieve_candidates

    class LegacyStorage:
        def __init__(self):
            self.kwargs = None

        async def retrieve(self, query, n_results=5, tags=None):
            self.kwargs = {"query": query, "n_results": n_results, "tags": tags}
            return []

    s = LegacyStorage()
    await _retrieve_candidates(s, "q", n_results=3, tags=None, store="docs")
    assert "store" not in s.kwargs


@pytest.mark.asyncio
async def test_memory_detail_unknown_entity(memory_server, graph_storage):
    """Unknown entity_id resolves to a clean error shape, not a crash."""
    if STORAGE_BACKEND not in ['sqlite_vec', 'hybrid']:
        pytest.skip(f"Graph operations not supported for backend: {STORAGE_BACKEND}")
    from mcp_memory_service.server.handlers.graph import handle_memory_detail
    result = await handle_memory_detail(
        memory_server, {"entity_id": "definitely-not-a-real-entity-zzz"}
    )
    assert len(result) == 1
    response = json.loads(result[0].text)
    assert response["success"] is False
    assert "Unknown entity_id" in response["error"]
    assert response["chunks"] == []
