# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for GraphService used by the streamable-http MCP server.

Covers:
- Normal operation paths (find_connected, find_shortest_path, get_subgraph)
- Unavailable graph storage (Milvus/Cloudflare backends)
- Parameter validation
"""

import sys
import os
import types as _types
import pytest
from unittest.mock import AsyncMock, MagicMock

# ---------------------------------------------------------------------------
# Import GraphService. Prefer the real module when project dependencies are
# available (e.g., CI with full install). Fall back to an isolated stub setup
# only when the heavy dependency chain (numpy, torch, sentence-transformers)
# can't be loaded — this keeps the test file usable in lightweight dev envs
# without polluting sys.modules for other test files that need the real
# GraphStorage (e.g., tests/test_graph_traversal.py).
# ---------------------------------------------------------------------------

try:
    from mcp_memory_service.services.graph_service import GraphService
except ImportError:
    # Dependencies missing — build an isolated stub environment for this file.
    for _pkg in [
        'mcp_memory_service',
        'mcp_memory_service.services',
        'mcp_memory_service.storage',
    ]:
        if _pkg not in sys.modules:
            _m = _types.ModuleType(_pkg)
            _m.__path__ = []  # make it a package
            sys.modules[_pkg] = _m

    _graph_stub = _types.ModuleType('mcp_memory_service.storage.graph')

    class _StubGraphStorage:
        """Stub for import resolution — tests use MagicMock, never this class."""
        pass

    _graph_stub.GraphStorage = _StubGraphStorage
    sys.modules['mcp_memory_service.storage.graph'] = _graph_stub

    _src = os.path.join(os.path.dirname(__file__), '..', 'src')
    sys.path.insert(0, _src)

    import importlib.util
    _gs_spec = importlib.util.spec_from_file_location(
        'mcp_memory_service.services.graph_service',
        os.path.join(_src, 'mcp_memory_service', 'services', 'graph_service.py'),
    )
    _gs_mod = importlib.util.module_from_spec(_gs_spec)
    sys.modules['mcp_memory_service.services.graph_service'] = _gs_mod
    _gs_spec.loader.exec_module(_gs_mod)

    GraphService = _gs_mod.GraphService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_graph_storage():
    """Create a mock GraphStorage with async methods."""
    storage = MagicMock()
    storage.find_connected = AsyncMock(return_value=[
        ("hash_a", 1),
        ("hash_b", 2),
    ])
    storage.shortest_path = AsyncMock(return_value=["hash_1", "hash_mid", "hash_2"])
    storage.get_subgraph = AsyncMock(return_value={
        "nodes": ["hash_1", "hash_2", "hash_3"],
        "edges": [
            {
                "source": "hash_1",
                "target": "hash_2",
                "similarity": 0.85,
                "connection_types": ["semantic"],
            }
        ],
    })
    return storage


@pytest.fixture
def graph_service(mock_graph_storage):
    """GraphService with a working mock storage."""
    return GraphService(mock_graph_storage)


@pytest.fixture
def unavailable_graph_service():
    """GraphService with no storage (simulates Milvus/Cloudflare backend)."""
    return GraphService(None)


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------

class TestIsAvailable:
    def test_available_with_storage(self, graph_service):
        assert graph_service.is_available() is True

    def test_unavailable_without_storage(self, unavailable_graph_service):
        assert unavailable_graph_service.is_available() is False


# ---------------------------------------------------------------------------
# find_connected — normal path
# ---------------------------------------------------------------------------

class TestFindConnected:
    @pytest.mark.asyncio
    async def test_returns_connected_memories(self, graph_service, mock_graph_storage):
        result = await graph_service.find_connected("hash_center", max_hops=3)

        assert result["success"] is True
        assert result["count"] == 2
        assert result["connected"][0] == {"hash": "hash_a", "distance": 1}
        assert result["connected"][1] == {"hash": "hash_b", "distance": 2}
        mock_graph_storage.find_connected.assert_awaited_once_with("hash_center", max_hops=3)

    @pytest.mark.asyncio
    async def test_empty_result(self, graph_service, mock_graph_storage):
        mock_graph_storage.find_connected.return_value = []
        result = await graph_service.find_connected("isolated_hash")

        assert result["success"] is True
        assert result["count"] == 0
        assert result["connected"] == []

    @pytest.mark.asyncio
    async def test_error_handling(self, graph_service, mock_graph_storage):
        mock_graph_storage.find_connected.side_effect = RuntimeError("DB error")
        result = await graph_service.find_connected("hash_x")

        assert result["success"] is False
        assert "DB error" in result["error"]
        assert result["count"] == 0


# ---------------------------------------------------------------------------
# find_connected — unavailable
# ---------------------------------------------------------------------------

class TestFindConnectedUnavailable:
    @pytest.mark.asyncio
    async def test_returns_error_when_unavailable(self, unavailable_graph_service):
        result = await unavailable_graph_service.find_connected("any_hash")

        assert result["success"] is False
        assert "not available" in result["error"]
        assert result["connected"] == []
        assert result["count"] == 0


# ---------------------------------------------------------------------------
# find_shortest_path — normal path
# ---------------------------------------------------------------------------

class TestFindShortestPath:
    @pytest.mark.asyncio
    async def test_returns_path(self, graph_service, mock_graph_storage):
        result = await graph_service.find_shortest_path("hash_1", "hash_2", max_depth=4)

        assert result["success"] is True
        assert result["path"] == ["hash_1", "hash_mid", "hash_2"]
        assert result["length"] == 3
        mock_graph_storage.shortest_path.assert_awaited_once_with("hash_1", "hash_2", max_depth=4)

    @pytest.mark.asyncio
    async def test_no_path_found(self, graph_service, mock_graph_storage):
        mock_graph_storage.shortest_path.return_value = None
        result = await graph_service.find_shortest_path("hash_a", "hash_z")

        assert result["success"] is True
        assert result["path"] is None
        assert result["length"] == 0
        assert "No path found" in result.get("message", "")

    @pytest.mark.asyncio
    async def test_error_handling(self, graph_service, mock_graph_storage):
        mock_graph_storage.shortest_path.side_effect = RuntimeError("timeout")
        result = await graph_service.find_shortest_path("h1", "h2")

        assert result["success"] is False
        assert "timeout" in result["error"]


# ---------------------------------------------------------------------------
# find_shortest_path — unavailable
# ---------------------------------------------------------------------------

class TestFindShortestPathUnavailable:
    @pytest.mark.asyncio
    async def test_returns_error_when_unavailable(self, unavailable_graph_service):
        result = await unavailable_graph_service.find_shortest_path("h1", "h2")

        assert result["success"] is False
        assert "not available" in result["error"]
        assert result["path"] is None


# ---------------------------------------------------------------------------
# get_subgraph — normal path
# ---------------------------------------------------------------------------

class TestGetSubgraph:
    @pytest.mark.asyncio
    async def test_returns_subgraph(self, graph_service, mock_graph_storage):
        result = await graph_service.get_subgraph("hash_center", radius=3)

        assert result["success"] is True
        assert result["node_count"] == 3
        assert result["edge_count"] == 1
        assert "hash_1" in result["nodes"]
        assert result["edges"][0]["source"] == "hash_1"
        mock_graph_storage.get_subgraph.assert_awaited_once_with("hash_center", radius=3)

    @pytest.mark.asyncio
    async def test_empty_subgraph(self, graph_service, mock_graph_storage):
        mock_graph_storage.get_subgraph.return_value = {"nodes": [], "edges": []}
        result = await graph_service.get_subgraph("isolated")

        assert result["success"] is True
        assert result["node_count"] == 0
        assert result["edge_count"] == 0

    @pytest.mark.asyncio
    async def test_error_handling(self, graph_service, mock_graph_storage):
        mock_graph_storage.get_subgraph.side_effect = RuntimeError("corrupt graph")
        result = await graph_service.get_subgraph("hash_x")

        assert result["success"] is False
        assert "corrupt graph" in result["error"]


# ---------------------------------------------------------------------------
# get_subgraph — unavailable
# ---------------------------------------------------------------------------

class TestGetSubgraphUnavailable:
    @pytest.mark.asyncio
    async def test_returns_error_when_unavailable(self, unavailable_graph_service):
        result = await unavailable_graph_service.get_subgraph("any")

        assert result["success"] is False
        assert "not available" in result["error"]
        assert result["nodes"] == []
        assert result["edges"] == []
