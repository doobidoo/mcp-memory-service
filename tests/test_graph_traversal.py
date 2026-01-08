"""
Integration tests for graph traversal MCP tools.

Tests the three graph traversal tools:
- find_connected_memories: Multi-hop connection discovery
- find_shortest_path: Path finding between memories
- get_memory_subgraph: Subgraph extraction for visualization
"""
import pytest
import pytest_asyncio
import json
from mcp_memory_service.server import MemoryServer
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
    import uuid
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
    import uuid
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
