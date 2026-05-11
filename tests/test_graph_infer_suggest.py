"""Integration tests for infer/suggest actions in memory_graph handler."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_memory_service.server.handlers.graph import handle_memory_graph


@pytest.fixture
def mock_graph_storage():
    """Create a mock graph storage with required methods."""
    graph = MagicMock()
    graph.find_connected = AsyncMock(return_value=[])
    graph.shortest_path = AsyncMock(return_value=None)
    graph.transitive_closure = AsyncMock(return_value=[
        ("hash1", "hash3", 2)
    ])
    graph.common_neighbors = AsyncMock(return_value=[
        ("hash2", 3, 5)
    ])
    return graph


@pytest.mark.asyncio
async def test_infer_action_returns_list(mock_graph_storage):
    """memory_graph(action='infer') returns a list without exception."""
    with patch(
        "mcp_memory_service.server.handlers.graph.get_graph_storage",
        return_value=mock_graph_storage,
    ):
        result = await handle_memory_graph(None, {
            "action": "infer",
            "rel_type": "causes",
            "max_hops": 2,
        })

    assert len(result) == 1
    data = json.loads(result[0].text)
    assert data["success"] is True
    assert isinstance(data["inferred"], list)


@pytest.mark.asyncio
async def test_suggest_action_returns_list(mock_graph_storage):
    """memory_graph(action='suggest') returns a list without exception."""
    with patch(
        "mcp_memory_service.server.handlers.graph.get_graph_storage",
        return_value=mock_graph_storage,
    ):
        result = await handle_memory_graph(None, {
            "action": "suggest",
            "hash": "abc123",
        })

    assert len(result) == 1
    data = json.loads(result[0].text)
    assert data["success"] is True
    assert isinstance(data["suggestions"], list)


@pytest.mark.asyncio
async def test_invalid_action_returns_error():
    """memory_graph with invalid action returns a clear error."""
    result = await handle_memory_graph(None, {
        "action": "invalid_action",
    })

    assert len(result) == 1
    assert "Error" in result[0].text
    assert "invalid_action" in result[0].text
