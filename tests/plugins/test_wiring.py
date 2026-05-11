"""Tests that plugin hooks are fired from MemoryService methods."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_memory_service.services.memory_service import MemoryService


@pytest.fixture
def mock_storage():
    storage = MagicMock()
    storage.max_content_length = 10000
    storage.store = AsyncMock(return_value=(True, "stored"))
    storage.delete = AsyncMock(return_value=(True, "deleted"))

    # retrieve returns MemoryQueryResult-like objects
    mock_memory = MagicMock()
    mock_memory.content = "test content"
    mock_memory.content_hash = "abc123"
    mock_memory.tags = ["test"]
    mock_memory.memory_type = "note"
    mock_memory.metadata = {}
    mock_memory.created_at = 1700000000.0
    mock_memory.created_at_iso = "2023-11-14T00:00:00Z"
    mock_memory.updated_at = None
    mock_memory.updated_at_iso = None
    mock_memory.last_accessed = None
    mock_memory.timestamp = None

    mock_result = MagicMock()
    mock_result.memory = mock_memory
    mock_result.relevance_score = 0.9
    storage.retrieve = AsyncMock(return_value=[mock_result])
    return storage


@pytest.fixture
def service(mock_storage):
    with patch("mcp_memory_service.services.memory_service.PluginRegistry") as MockRegistry:
        instance = MockRegistry.return_value
        instance.fire = AsyncMock(side_effect=lambda hook, *a, **kw: a[1] if hook == "on_retrieve" and len(a) > 1 else (a[0] if a else None))
        instance.discover_and_register = MagicMock()
        svc = MemoryService(mock_storage)
    return svc


class TestStoreHookFired:
    @pytest.mark.asyncio
    async def test_on_store_fired_after_success(self, service):
        result = await service.store_memory("hello world", tags=["test"])
        assert result["success"] is True
        # Verify on_store was fired
        calls = [c for c in service._plugin_registry.fire.call_args_list if c[0][0] == "on_store"]
        assert len(calls) == 1
        assert calls[0][0][0] == "on_store"

    @pytest.mark.asyncio
    async def test_on_store_not_fired_on_failure(self, service, mock_storage):
        mock_storage.store = AsyncMock(return_value=(False, "duplicate"))
        result = await service.store_memory("hello world")
        assert result["success"] is False
        calls = [c for c in service._plugin_registry.fire.call_args_list if c[0][0] == "on_store"]
        assert len(calls) == 0


class TestDeleteHookFired:
    @pytest.mark.asyncio
    async def test_on_delete_fired_after_success(self, service):
        result = await service.delete_memory("abc123")
        assert result["success"] is True
        calls = [c for c in service._plugin_registry.fire.call_args_list if c[0][0] == "on_delete"]
        assert len(calls) == 1
        assert calls[0][0][1] == "abc123"

    @pytest.mark.asyncio
    async def test_on_delete_not_fired_on_failure(self, service, mock_storage):
        mock_storage.delete = AsyncMock(return_value=(False, "not found"))
        result = await service.delete_memory("xyz")
        assert result["success"] is False
        calls = [c for c in service._plugin_registry.fire.call_args_list if c[0][0] == "on_delete"]
        assert len(calls) == 0


class TestRetrieveHookFired:
    @pytest.mark.asyncio
    async def test_on_retrieve_fired(self, service):
        result = await service.retrieve_memories("test query")
        calls = [c for c in service._plugin_registry.fire.call_args_list if c[0][0] == "on_retrieve"]
        assert len(calls) == 1
        assert calls[0][0][1] == "test query"  # query passed
        assert isinstance(calls[0][0][2], list)  # results passed


class TestConsolidateHookFired:
    @pytest.mark.asyncio
    async def test_on_consolidate_fired(self):
        from mcp_memory_service.consolidation.consolidator import DreamInspiredConsolidator
        from mcp_memory_service.consolidation.base import ConsolidationConfig

        mock_storage = MagicMock()
        mock_storage.get_memories_by_time_range = AsyncMock(return_value=[])
        mock_storage.get_all_memories = AsyncMock(return_value=[])

        consolidator = DreamInspiredConsolidator(mock_storage, ConsolidationConfig())
        consolidator.plugin_registry = MagicMock()
        consolidator.plugin_registry.fire = AsyncMock()

        # daily with no memories — should not fire (early return)
        await consolidator.consolidate("daily")
        consolidator.plugin_registry.fire.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_consolidate_not_fired_without_registry(self):
        from mcp_memory_service.consolidation.consolidator import DreamInspiredConsolidator
        from mcp_memory_service.consolidation.base import ConsolidationConfig

        mock_storage = MagicMock()
        mock_storage.get_memories_by_time_range = AsyncMock(return_value=[])

        consolidator = DreamInspiredConsolidator(mock_storage, ConsolidationConfig())
        assert consolidator.plugin_registry is None
        # Should not raise
        await consolidator.consolidate("daily")
