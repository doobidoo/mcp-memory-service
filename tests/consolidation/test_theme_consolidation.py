"""Tests for INC 2: memory_consolidate(theme=...) - theme-restricted consolidation."""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from typing import List

from mcp_memory_service.consolidation.consolidator import DreamInspiredConsolidator
from mcp_memory_service.consolidation.base import ConsolidationReport
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.server.handlers import consolidation as consolidation_handlers


# Module-level fixtures for theme consolidation tests

@pytest.fixture
def theme_memories():
    """Create memories with distinct themes for testing."""
    base_time = datetime.now().timestamp()
    
    def _mem_ts(offset):
        """Helper: return created_at/updated_at kwargs for a memory `offset` seconds old."""
        ts = base_time - offset
        iso = datetime.fromtimestamp(ts).isoformat() + 'Z'
        return dict(created_at=ts, created_at_iso=iso, updated_at=ts, updated_at_iso=iso)
    
    return [
        # Python development theme
        Memory(
            content="Fixed Python import issue in the API module",
            content_hash="py001",
            tags=["python", "api", "bug"],
            memory_type="observation",
            embedding=[0.8, 0.1, 0.1, 0.0, 0.0] * 64,
            **_mem_ts(86400),  # 1 day ago
        ),
        Memory(
            content="Refactored Python async functions for better performance",
            content_hash="py002", 
            tags=["python", "async", "performance"],
            memory_type="learning",
            embedding=[0.7, 0.2, 0.1, 0.0, 0.0] * 64,
            **_mem_ts(172800),  # 2 days ago
        ),
        
        # Database theme
        Memory(
            content="Optimized PostgreSQL query for user lookup",
            content_hash="db001",
            tags=["database", "postgresql", "optimization"],
            memory_type="observation", 
            embedding=[0.1, 0.8, 0.1, 0.0, 0.0] * 64,
            **_mem_ts(259200),  # 3 days ago
        ),
        Memory(
            content="Database migration completed successfully",
            content_hash="db002",
            tags=["database", "migration", "success"],
            memory_type="decision",
            embedding=[0.0, 0.9, 0.1, 0.0, 0.0] * 64,
            **_mem_ts(345600),  # 4 days ago
        ),
        
        # Infrastructure theme  
        Memory(
            content="Kubernetes deployment configuration updated",
            content_hash="k8s001",
            tags=["kubernetes", "deployment", "config"],
            memory_type="observation",
            embedding=[0.0, 0.0, 0.8, 0.2, 0.0] * 64,
            **_mem_ts(432000),  # 5 days ago
        ),
    ]

@pytest.fixture
def mock_storage_with_themes(theme_memories):
    """Mock storage that returns themed memories."""
    storage = AsyncMock()
    storage.get_memories_by_time_range.return_value = theme_memories
    
    # Mock search_memories to return themed results based on query
    async def mock_search_memories(query=None, limit=10, **kwargs):
        if not query:
            return {"memories": [m.to_dict() for m in theme_memories], "total": len(theme_memories)}
        
        # Simulate semantic search - return memories matching theme
        results = []
        if "python" in query.lower():
            results = [m for m in theme_memories if "python" in m.content.lower() or "py" in m.content_hash]
        elif "database" in query.lower():
            results = [m for m in theme_memories if "database" in m.content.lower() or "db" in m.content_hash]
        elif "kubernetes" in query.lower():
            results = [m for m in theme_memories if "kubernetes" in m.content.lower() or "k8s" in m.content_hash]
        
        return {"memories": [m.to_dict() for m in results], "total": len(results)}
    
    storage.search_memories.side_effect = mock_search_memories
    return storage

@pytest.fixture
def theme_consolidator(mock_storage_with_themes, consolidation_config):
    """Consolidator with theme-aware mock storage."""
    return DreamInspiredConsolidator(mock_storage_with_themes, consolidation_config)


@pytest.mark.integration
class TestThemeConsolidation:
    """Test theme-restricted consolidation for INC 2."""
    
    @pytest.mark.asyncio
    async def test_consolidate_with_theme_filters_memories(self, theme_consolidator):
        """TEST: consolidate(theme='python') should process only Python-related memories.
        
        WHY IT FAILS: The current consolidate() method ignores the theme parameter
        and processes all memories returned by _get_memories_for_horizon().
        """
        report = await theme_consolidator.consolidate("weekly", theme="python")
        
        # Should only process Python memories (2 out of 5 total)
        assert report.memories_processed == 2, (
            f"Expected 2 Python memories to be processed, got {report.memories_processed}. "
            "Current code ignores theme and processes all memories."
        )
    
    @pytest.mark.asyncio  
    async def test_consolidate_with_database_theme(self, theme_consolidator):
        """TEST: consolidate(theme='database') should process only database memories.
        
        WHY IT FAILS: The consolidate method does not filter by theme,
        so it processes all 5 memories instead of just the 2 database ones.
        """
        report = await theme_consolidator.consolidate("weekly", theme="database")
        
        # Should only process database memories (2 out of 5 total)  
        assert report.memories_processed == 2, (
            f"Expected 2 database memories to be processed, got {report.memories_processed}. "
            "Current code processes all memories regardless of theme."
        )
    
    @pytest.mark.asyncio
    async def test_consolidate_kwargs_ignored_current_behavior(self, theme_consolidator):
        """TEST: consolidate(**kwargs) now properly handles theme parameter.
        
        This test was originally written to demonstrate that theme was ignored,
        but now that theme filtering is implemented, it verifies the correct behavior.
        """
        # Now that theme filtering is implemented, these should behave differently
        report1 = await theme_consolidator.consolidate("weekly")
        report2 = await theme_consolidator.consolidate("weekly", theme="python")  
        report3 = await theme_consolidator.consolidate("weekly", theme="nonexistent")
        
        # Should process different numbers of memories based on theme filtering
        assert report1.memories_processed == 5, f"Expected 5 memories without theme, got {report1.memories_processed}"
        assert report2.memories_processed == 2, f"Expected 2 Python memories, got {report2.memories_processed}" 
        assert report3.memories_processed == 0, f"Expected 0 memories for nonexistent theme, got {report3.memories_processed}"

    @pytest.mark.asyncio
    async def test_consolidate_without_theme_preserves_behavior(self, theme_consolidator):
        """TEST: consolidate() without theme should process all memories (backward compatibility).
        
        WHY IT FAILS: This test should actually pass with current code, but we include it
        to verify that adding theme support doesn't break existing behavior.
        """
        report = await theme_consolidator.consolidate("weekly")
        
        # Should process all 5 memories when no theme is specified
        assert report.memories_processed == 5, (
            f"Expected all 5 memories to be processed without theme, got {report.memories_processed}"
        )
    
    @pytest.mark.asyncio
    async def test_consolidate_with_nonexistent_theme(self, theme_consolidator):
        """TEST: consolidate(theme='nonexistent') should process zero memories.
        
        WHY IT FAILS: Current code ignores theme and processes all memories,
        so it will process 5 instead of 0.
        """
        report = await theme_consolidator.consolidate("weekly", theme="nonexistent")
        
        # Should process no memories when theme doesn't match anything
        assert report.memories_processed == 0, (
            f"Expected 0 memories for nonexistent theme, got {report.memories_processed}. "
            "Current code ignores theme filter."
        )


@pytest.mark.integration  
class TestThemeConsolidationHandler:
    """Test the MCP handler integration for theme consolidation."""
    
    @pytest.fixture
    def mock_server_with_theme_consolidator(self, theme_consolidator):
        """Mock server with theme-aware consolidator."""
        server = MagicMock()
        server.consolidator = theme_consolidator
        return server
    
    @pytest.mark.asyncio
    async def test_handler_accepts_theme_parameter(self, mock_server_with_theme_consolidator, monkeypatch):
        """TEST: handle_consolidate_memories should accept and pass through theme parameter."""
        # Mock CONSOLIDATION_ENABLED to be True
        monkeypatch.setattr('mcp_memory_service.server.handlers.consolidation.CONSOLIDATION_ENABLED', True)
        
        arguments = {
            "time_horizon": "weekly",
            "theme": "python"
        }
        
        # Mock the consolidate method to capture arguments
        mock_consolidate = AsyncMock(return_value=MagicMock(
            time_horizon="weekly",
            start_time=datetime.now(),
            end_time=datetime.now(),
            memories_processed=2,
            associations_discovered=0,
            clusters_created=0,
            memories_compressed=0,
            memories_archived=0,
            errors=[]
        ))
        mock_server_with_theme_consolidator.consolidator.consolidate = mock_consolidate
        
        await consolidation_handlers.handle_consolidate_memories(
            mock_server_with_theme_consolidator, arguments
        )
        
        # Should have called consolidate with theme parameter
        mock_consolidate.assert_called_once_with("weekly", theme="python")
    
    @pytest.mark.asyncio
    async def test_handler_theme_parameter_optional(self, mock_server_with_theme_consolidator, monkeypatch):
        """TEST: handle_consolidate_memories should work without theme (backward compatibility)."""
        # Mock CONSOLIDATION_ENABLED to be True
        monkeypatch.setattr('mcp_memory_service.server.handlers.consolidation.CONSOLIDATION_ENABLED', True)
        
        arguments = {"time_horizon": "weekly"}
        
        mock_consolidate = AsyncMock(return_value=MagicMock(
            time_horizon="weekly",
            start_time=datetime.now(),
            end_time=datetime.now(), 
            memories_processed=5,
            associations_discovered=0,
            clusters_created=0,
            memories_compressed=0,
            memories_archived=0,
            errors=[]
        ))
        mock_server_with_theme_consolidator.consolidator.consolidate = mock_consolidate
        
        await consolidation_handlers.handle_consolidate_memories(
            mock_server_with_theme_consolidator, arguments
        )
        
        # Should have called consolidate with only time_horizon (no theme)
        mock_consolidate.assert_called_once_with("weekly")