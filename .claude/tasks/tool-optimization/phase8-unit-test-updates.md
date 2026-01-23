# Phase 8: Unit Test Updates

## Ziel

Unit-Tests für MemoryService-Methoden aktualisieren und neue Tests für konsolidierte Methoden hinzufügen.

## Betroffene Dateien

| Datei | Änderungen | Priorität |
|-------|-----------|-----------|
| `tests/unit/test_memory_service.py` | Neue Methoden testen | 🔴 HIGH |
| `tests/unit/test_fastapi_dependencies.py` | Dependency-Namen prüfen | 🟡 MEDIUM |
| `tests/unit/test_cloudflare_storage.py` | Storage-Methoden | 🟢 LOW |

---

## Task 8.1: MemoryService Tests erweitern

### Neue zu testende Methoden

Nach der Konsolidierung gibt es neue unified Methoden:

```python
# Neue MemoryService Methoden
async def search_memories(
    query: Optional[str],
    mode: Literal["semantic", "exact", "hybrid"],
    time_expr: Optional[str],
    after: Optional[str],
    before: Optional[str],
    tags: Optional[List[str]],
    quality_boost: float,
    limit: int,
    include_debug: bool
) -> Dict[str, Any]

async def delete_memories(
    content_hash: Optional[str],
    tags: Optional[List[str]],
    tag_match: Literal["any", "all"],
    before: Optional[str],
    after: Optional[str],
    dry_run: bool
) -> Dict[str, Any]

async def manage_consolidation(
    action: Literal["run", "status", "recommend", "scheduler", "pause", "resume"],
    time_horizon: Optional[str],
    immediate: bool
) -> Dict[str, Any]
```

### Amp Prompt

```
Add unit tests for new unified methods in tests/unit/test_memory_service.py:

1. First read the existing file structure.

2. Add new test class for search_memories:

class TestSearchMemories:
    """Tests for unified search_memories method."""
    
    @pytest.mark.asyncio
    async def test_semantic_search_default(self, memory_service, mock_storage):
        """Test default semantic search mode."""
        mock_storage.retrieve.return_value = [
            MemoryQueryResult(
                memory=Memory(content="Test", content_hash="hash1", tags=[], memory_type="note"),
                relevance_score=0.9
            )
        ]
        
        result = await memory_service.search_memories(query="test query")
        
        assert result["mode"] == "semantic"
        assert "memories" in result
        mock_storage.retrieve.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_exact_search_mode(self, memory_service, mock_storage):
        """Test exact string match mode."""
        mock_storage.exact_match_search.return_value = [
            Memory(content="exact match", content_hash="hash1", tags=[], memory_type="note")
        ]
        
        result = await memory_service.search_memories(
            query="exact match",
            mode="exact"
        )
        
        assert result["mode"] == "exact"
    
    @pytest.mark.asyncio
    async def test_time_expression_parsing(self, memory_service, mock_storage):
        """Test time_expr parameter is parsed correctly."""
        mock_storage.recall.return_value = []
        
        result = await memory_service.search_memories(time_expr="last week")
        
        # Should have called recall with parsed timestamps
        mock_storage.recall.assert_called_once()
        call_args = mock_storage.recall.call_args
        assert call_args.kwargs.get("start_timestamp") is not None
    
    @pytest.mark.asyncio
    async def test_quality_boost_reranking(self, memory_service, mock_storage):
        """Test quality_boost triggers reranking."""
        # Setup memories with different quality scores
        mock_storage.retrieve.return_value = [
            MemoryQueryResult(
                memory=Memory(
                    content="Low quality", 
                    content_hash="hash1", 
                    tags=[], 
                    memory_type="note",
                    metadata={"quality_score": 0.3}
                ),
                relevance_score=0.9
            ),
            MemoryQueryResult(
                memory=Memory(
                    content="High quality", 
                    content_hash="hash2", 
                    tags=[], 
                    memory_type="note",
                    metadata={"quality_score": 0.9}
                ),
                relevance_score=0.7
            ),
        ]
        
        result = await memory_service.search_memories(
            query="test",
            quality_boost=0.5,
            limit=2
        )
        
        # With 50% quality weight, high quality memory should rank higher
        assert len(result["memories"]) <= 2
    
    @pytest.mark.asyncio
    async def test_combined_filters(self, memory_service, mock_storage):
        """Test combining multiple filter parameters."""
        mock_storage.retrieve.return_value = []
        
        result = await memory_service.search_memories(
            query="test",
            time_expr="yesterday",
            tags=["important"],
            quality_boost=0.3,
            limit=5
        )
        
        assert "memories" in result
    
    @pytest.mark.asyncio
    async def test_include_debug_info(self, memory_service, mock_storage):
        """Test debug info is included when requested."""
        mock_storage.retrieve.return_value = []
        
        result = await memory_service.search_memories(
            query="test",
            include_debug=True
        )
        
        assert "debug" in result
    
    @pytest.mark.asyncio
    async def test_semantic_mode_requires_query(self, memory_service, mock_storage):
        """Test that semantic mode requires query parameter."""
        result = await memory_service.search_memories(mode="semantic")
        
        assert "error" in result


3. Add new test class for delete_memories:

class TestDeleteMemories:
    """Tests for unified delete_memories method."""
    
    @pytest.mark.asyncio
    async def test_delete_by_hash(self, memory_service, mock_storage):
        """Test deletion by content hash."""
        mock_storage.delete.return_value = (True, "Deleted")
        
        result = await memory_service.delete_memories(content_hash="test_hash")
        
        assert result["success"] is True
        assert result["deleted_count"] == 1
        mock_storage.delete.assert_called_once_with("test_hash")
    
    @pytest.mark.asyncio
    async def test_delete_by_tags_any_match(self, memory_service, mock_storage):
        """Test deletion by tags with ANY match."""
        mock_storage.delete_by_tags.return_value = {
            "deleted_count": 3,
            "deleted_hashes": ["h1", "h2", "h3"]
        }
        
        result = await memory_service.delete_memories(
            tags=["temp", "draft"],
            tag_match="any"
        )
        
        assert result["deleted_count"] == 3
    
    @pytest.mark.asyncio
    async def test_delete_by_tags_all_match(self, memory_service, mock_storage):
        """Test deletion by tags with ALL match."""
        mock_storage.delete_by_all_tags.return_value = {
            "deleted_count": 1,
            "deleted_hashes": ["h1"]
        }
        
        result = await memory_service.delete_memories(
            tags=["archived", "old"],
            tag_match="all"
        )
        
        assert result["deleted_count"] == 1
    
    @pytest.mark.asyncio
    async def test_delete_by_time_range(self, memory_service, mock_storage):
        """Test deletion by time range."""
        mock_storage.delete_by_timeframe.return_value = {
            "deleted_count": 5
        }
        
        result = await memory_service.delete_memories(
            after="2024-01-01",
            before="2024-06-30"
        )
        
        assert result["deleted_count"] == 5
    
    @pytest.mark.asyncio
    async def test_delete_dry_run(self, memory_service, mock_storage):
        """Test dry_run mode doesn't actually delete."""
        mock_storage.get_memories_matching_filters.return_value = [
            Memory(content="1", content_hash="h1", tags=["temp"], memory_type="note"),
            Memory(content="2", content_hash="h2", tags=["temp"], memory_type="note"),
        ]
        
        result = await memory_service.delete_memories(
            tags=["temp"],
            dry_run=True
        )
        
        assert result["dry_run"] is True
        assert result["deleted_count"] == 2
        # Actual delete should NOT be called
        mock_storage.delete.assert_not_called()
        mock_storage.delete_by_tags.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_delete_no_filters_error(self, memory_service, mock_storage):
        """Test that delete without any filters returns error."""
        result = await memory_service.delete_memories()
        
        assert "error" in result
        assert "filter" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_delete_combined_filters(self, memory_service, mock_storage):
        """Test combined tag + time filters."""
        mock_storage.delete_with_filters.return_value = {
            "deleted_count": 2
        }
        
        result = await memory_service.delete_memories(
            tags=["cleanup"],
            before="2024-01-01"
        )
        
        assert "deleted_count" in result


4. Add test class for manage_consolidation:

class TestManageConsolidation:
    """Tests for unified manage_consolidation method."""
    
    @pytest.mark.asyncio
    async def test_status_action(self, memory_service, mock_storage):
        """Test status action returns consolidation status."""
        # Mock consolidator if needed
        result = await memory_service.manage_consolidation(action="status")
        
        assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_run_requires_time_horizon(self, memory_service, mock_storage):
        """Test run action requires time_horizon."""
        result = await memory_service.manage_consolidation(action="run")
        
        assert "error" in result
        assert "time_horizon" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_run_with_valid_horizon(self, memory_service, mock_storage):
        """Test run with valid time horizon."""
        result = await memory_service.manage_consolidation(
            action="run",
            time_horizon="weekly"
        )
        
        # Should not error (actual consolidation may or may not run)
        assert "error" not in result or result.get("success")
    
    @pytest.mark.asyncio
    async def test_scheduler_action(self, memory_service, mock_storage):
        """Test scheduler status retrieval."""
        result = await memory_service.manage_consolidation(action="scheduler")
        
        assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_unknown_action_error(self, memory_service, mock_storage):
        """Test unknown action returns error."""
        result = await memory_service.manage_consolidation(action="invalid")
        
        assert "error" in result


5. Keep existing tests for deprecated methods to verify backwards compatibility.
   Add @pytest.mark.deprecated decorator or comment to mark them as testing legacy code.
```

---

## Task 8.2: FastAPI Dependencies Tests prüfen

### Amp Prompt

```
Check and update tests/unit/test_fastapi_dependencies.py:

1. Read the file to see what's being tested.

2. If it references old method names like 'store_memory', 'delete_memory':
   - These are testing the FastAPI dependency injection
   - The HTTP API layer doesn't change method names
   - Tests should remain as-is

3. If the file tests MCP tool names:
   - Update to new tool names: memory_store, memory_search, etc.

4. Add a test to verify the MemoryService has expected methods:

def test_memory_service_has_unified_methods():
    """Verify MemoryService exposes unified methods."""
    from mcp_memory_service.services.memory_service import MemoryService
    
    service = MemoryService.__new__(MemoryService)  # Don't initialize
    
    # New unified methods
    assert hasattr(service, 'search_memories')
    assert hasattr(service, 'delete_memories')
    assert hasattr(service, 'manage_consolidation')
    
    # Old methods should still exist (deprecated)
    assert hasattr(service, 'store_memory')
    assert hasattr(service, 'retrieve_memories')
    assert hasattr(service, 'delete_memory')
```

---

## Task 8.3: Storage-Layer Tests

**Hinweis:** Die Storage-Schicht (`MemoryStorage`) ändert sich NICHT.
Nur die `MemoryService`-Schicht wird konsolidiert.

Die Storage-Tests in `tests/unit/test_cloudflare_storage.py` sollten unverändert bleiben.

Jedoch sollte ein Smoke-Test hinzugefügt werden:

```python
# In tests/unit/test_storage_interface.py (neu)

"""
Tests to verify storage interface compatibility after service consolidation.
"""

import pytest
from mcp_memory_service.storage.base import MemoryStorage


class TestStorageInterfaceUnchanged:
    """Verify storage interface methods are unchanged."""
    
    def test_storage_has_required_methods(self):
        """Storage should have all methods the service layer expects."""
        required_methods = [
            # Core CRUD
            'store', 'retrieve', 'delete', 'get_by_hash',
            # Search
            'search_by_tag', 'recall',
            # Bulk operations
            'delete_by_tags', 'get_all_memories',
            # Stats
            'get_stats', 'count_all_memories',
        ]
        
        for method in required_methods:
            assert hasattr(MemoryStorage, method), f"Missing method: {method}"
```

---

## Checkliste

- [ ] `test_memory_service.py` erweitert
  - [ ] TestSearchMemories Klasse
  - [ ] TestDeleteMemories Klasse
  - [ ] TestManageConsolidation Klasse
- [ ] `test_fastapi_dependencies.py` geprüft
- [ ] `test_storage_interface.py` erstellt (optional)
- [ ] Deprecated-Markierungen für alte Tests
- [ ] Alle Unit-Tests grün: `uv run pytest tests/unit/ -v`
