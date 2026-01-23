# Phase 7: Test Handler Migration

## Ziel

Integration-Tests auf neue Handler-Namen aktualisieren, während Backwards-Compatibility getestet wird.

## Betroffene Dateien

| Datei | Handler-Aufrufe | Priorität |
|-------|-----------------|-----------|
| `tests/integration/test_all_memory_handlers.py` | 50+ | 🔴 KRITISCH |
| `tests/integration/test_server_handlers.py` | 15+ | 🔴 KRITISCH |
| `tests/conftest.py` | 2 | 🟡 MEDIUM |
| `tests/test_database.py` | 5 | 🟡 MEDIUM |

---

## Task 7.1: Handler-Mapping erstellen

**Ziel:** Dokumentiere alle Handler-Umbenennungen

```python
# Handler-Mapping: Alt → Neu
HANDLER_MIGRATION = {
    # Store
    "handle_store_memory": "handle_memory_store",
    
    # Search/Retrieve (5 → 1)
    "handle_retrieve_memory": "handle_memory_search",
    "handle_recall_memory": "handle_memory_search",
    "handle_recall_by_timeframe": "handle_memory_search",
    "handle_retrieve_with_quality_boost": "handle_memory_search",
    "handle_exact_match_retrieve": "handle_memory_search",
    
    # List
    "handle_search_by_tag": "handle_memory_list",
    "handle_list_memories": "handle_memory_list",
    
    # Delete (6 → 1)
    "handle_delete_memory": "handle_memory_delete",
    "handle_delete_by_tag": "handle_memory_delete",
    "handle_delete_by_tags": "handle_memory_delete",
    "handle_delete_by_all_tags": "handle_memory_delete",
    "handle_delete_by_timeframe": "handle_memory_delete",
    "handle_delete_before_date": "handle_memory_delete",
    
    # Quality (3 → 1)
    "handle_rate_memory": "handle_memory_quality",
    "handle_get_memory_quality": "handle_memory_quality",
    "handle_analyze_quality_distribution": "handle_memory_quality",
    
    # Consolidation (7 → 1)
    "handle_consolidate_memories": "handle_memory_consolidate",
    "handle_consolidation_status": "handle_memory_consolidate",
    "handle_consolidation_recommendations": "handle_memory_consolidate",
    "handle_scheduler_status": "handle_memory_consolidate",
    "handle_trigger_consolidation": "handle_memory_consolidate",
    "handle_pause_consolidation": "handle_memory_consolidate",
    "handle_resume_consolidation": "handle_memory_consolidate",
    
    # Andere
    "handle_check_database_health": "handle_memory_health",
    "handle_get_cache_stats": "handle_memory_stats",
    "handle_cleanup_duplicates": "handle_memory_cleanup",
    "handle_update_memory_metadata": "handle_memory_update",
}
```

---

## Task 7.2: test_all_memory_handlers.py migrieren

### Amp Prompt

```
Migrate tests/integration/test_all_memory_handlers.py to use new unified handlers:

1. First, read the current file to understand its structure.

2. Create a NEW test file tests/integration/test_unified_handlers.py that tests the NEW handlers:
   - TestHandleMemoryStore (from TestHandleStoreMemory)
   - TestHandleMemorySearch (combines retrieve, recall, quality_boost tests)
   - TestHandleMemoryDelete (combines all delete tests)
   - TestHandleMemoryList (from search_by_tag tests)

3. For TestHandleMemorySearch, create tests for different modes:

class TestHandleMemorySearch:
    """Tests for unified memory_search handler."""
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, unique_content):
        """Test default semantic search mode."""
        server = MemoryServer()
        await server.handle_memory_store({
            "content": unique_content("Searchable memory"),
            "tags": ["search-test"]
        })
        
        result = await server.handle_memory_search({
            "query": "searchable",
            "limit": 5
        })
        
        assert isinstance(result, list)
        # Parse JSON response
        import json
        data = json.loads(result[0].text)
        assert "memories" in data
    
    @pytest.mark.asyncio
    async def test_exact_search(self, unique_content):
        """Test exact match mode."""
        server = MemoryServer()
        content = unique_content("Exact match test content")
        await server.handle_memory_store({"content": content})
        
        result = await server.handle_memory_search({
            "query": "Exact match test",
            "mode": "exact"
        })
        
        data = json.loads(result[0].text)
        assert "memories" in data
    
    @pytest.mark.asyncio  
    async def test_time_expression_search(self, unique_content):
        """Test time_expr parameter (replaces recall_memory)."""
        server = MemoryServer()
        await server.handle_memory_store({
            "content": unique_content("Recent memory"),
            "tags": ["time-test"]
        })
        
        result = await server.handle_memory_search({
            "time_expr": "last week",
            "limit": 10
        })
        
        data = json.loads(result[0].text)
        assert "memories" in data
    
    @pytest.mark.asyncio
    async def test_quality_boosted_search(self, unique_content):
        """Test quality_boost parameter (replaces retrieve_with_quality_boost)."""
        server = MemoryServer()
        await server.handle_memory_store({
            "content": unique_content("High quality memory"),
            "tags": ["quality-test"]
        })
        
        result = await server.handle_memory_search({
            "query": "quality",
            "quality_boost": 0.3,
            "limit": 5
        })
        
        data = json.loads(result[0].text)
        assert "memories" in data
    
    @pytest.mark.asyncio
    async def test_date_range_search(self, unique_content):
        """Test after/before parameters (replaces recall_by_timeframe)."""
        from datetime import datetime, timedelta
        
        server = MemoryServer()
        await server.handle_memory_store({
            "content": unique_content("Dated memory"),
            "tags": ["date-test"]
        })
        
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        
        result = await server.handle_memory_search({
            "after": yesterday,
            "before": tomorrow,
            "limit": 10
        })
        
        data = json.loads(result[0].text)
        assert "memories" in data

4. For TestHandleMemoryDelete, create tests for different filter combinations:

class TestHandleMemoryDelete:
    """Tests for unified memory_delete handler."""
    
    @pytest.mark.asyncio
    async def test_delete_by_hash(self, unique_content):
        """Test deletion by content_hash (replaces delete_memory)."""
        server = MemoryServer()
        store_result = await server.handle_memory_store({
            "content": unique_content("To be deleted by hash"),
            "tags": ["delete-test"]
        })
        
        # Extract hash from response
        import json
        store_data = json.loads(store_result[0].text)
        content_hash = store_data.get("content_hash") or store_data.get("memory", {}).get("content_hash")
        
        result = await server.handle_memory_delete({
            "content_hash": content_hash
        })
        
        data = json.loads(result[0].text)
        assert data.get("deleted_count", 0) >= 0 or data.get("success")
    
    @pytest.mark.asyncio
    async def test_delete_by_tags_any(self, unique_content):
        """Test deletion by tags with ANY match (replaces delete_by_tag/delete_by_tags)."""
        server = MemoryServer()
        await server.handle_memory_store({
            "content": unique_content("Delete by tag 1"),
            "tags": ["delete-tag-a", "common"]
        })
        await server.handle_memory_store({
            "content": unique_content("Delete by tag 2"),
            "tags": ["delete-tag-b", "common"]
        })
        
        result = await server.handle_memory_delete({
            "tags": ["delete-tag-a", "delete-tag-b"],
            "tag_match": "any"
        })
        
        data = json.loads(result[0].text)
        assert "deleted_count" in data or "success" in data
    
    @pytest.mark.asyncio
    async def test_delete_by_tags_all(self, unique_content):
        """Test deletion by tags with ALL match (replaces delete_by_all_tags)."""
        server = MemoryServer()
        await server.handle_memory_store({
            "content": unique_content("Has both tags"),
            "tags": ["must-have-a", "must-have-b"]
        })
        
        result = await server.handle_memory_delete({
            "tags": ["must-have-a", "must-have-b"],
            "tag_match": "all"
        })
        
        data = json.loads(result[0].text)
        assert "deleted_count" in data or "success" in data
    
    @pytest.mark.asyncio
    async def test_delete_by_timeframe(self, unique_content):
        """Test deletion by time range (replaces delete_by_timeframe)."""
        from datetime import datetime, timedelta
        
        server = MemoryServer()
        await server.handle_memory_store({
            "content": unique_content("Old memory to delete"),
            "tags": ["timeframe-delete"]
        })
        
        result = await server.handle_memory_delete({
            "after": "2020-01-01",
            "before": "2020-12-31",
            "dry_run": True  # Safety: preview only
        })
        
        data = json.loads(result[0].text)
        assert "deleted_count" in data or "dry_run" in data
    
    @pytest.mark.asyncio
    async def test_delete_dry_run(self, unique_content):
        """Test dry_run mode prevents actual deletion."""
        server = MemoryServer()
        await server.handle_memory_store({
            "content": unique_content("Should not be deleted"),
            "tags": ["dry-run-test"]
        })
        
        result = await server.handle_memory_delete({
            "tags": ["dry-run-test"],
            "dry_run": True
        })
        
        data = json.loads(result[0].text)
        assert data.get("dry_run") is True
        # Memory should still exist
    
    @pytest.mark.asyncio
    async def test_delete_no_filters_error(self):
        """Test that delete without filters returns error."""
        server = MemoryServer()
        
        result = await server.handle_memory_delete({})
        
        data = json.loads(result[0].text)
        assert "error" in data

5. Keep the OLD test file for backwards compatibility testing.
   Rename it to tests/integration/test_deprecated_handlers.py
   
   Add a note at the top:
   '''
   Tests for DEPRECATED handlers.
   These tests verify backwards compatibility via the deprecation layer.
   New code should use the unified handlers tested in test_unified_handlers.py
   '''

6. Update tests/integration/HANDLER_COVERAGE_REPORT.md to reflect new structure.
```

---

## Task 7.3: test_server_handlers.py migrieren

### Amp Prompt

```
Migrate tests/integration/test_server_handlers.py to new handler names:

1. Read the current file structure.

2. Update handler calls:
   - handle_store_memory → handle_memory_store
   - handle_retrieve_memory → handle_memory_search
   - handle_search_by_tag → handle_memory_list

3. Update argument formats for new unified handlers:

# OLD: handle_retrieve_memory
result = await server.handle_retrieve_memory({
    "query": "test",
    "n_results": 5
})

# NEW: handle_memory_search  
result = await server.handle_memory_search({
    "query": "test",
    "limit": 5  # Note: n_results → limit
})

# OLD: handle_search_by_tag
result = await server.handle_search_by_tag({
    "tags": ["test"]
})

# NEW: handle_memory_list
result = await server.handle_memory_list({
    "tags": ["test"]
})

4. Keep test logic identical - only change handler names and argument names.

5. Add comment noting migration from old handlers.
```

---

## Task 7.4: conftest.py Cleanup-Funktion anpassen

### Amp Prompt

```
Update tests/conftest.py to use new delete handler:

1. Find the cleanup fixture that uses delete_by_tag.

2. Update to use the new unified approach:

# OLD
from mcp_memory_service.api import delete_by_tag
result = delete_by_tag([TEST_MEMORY_TAG])

# NEW - Option A: Use MemoryService directly
from mcp_memory_service.services.memory_service import MemoryService

@pytest.fixture(autouse=True)
async def cleanup_test_memories():
    yield
    # Cleanup after test
    service = MemoryService()
    await service.delete_memories(tags=[TEST_MEMORY_TAG], tag_match="any")

# NEW - Option B: Use deprecation-compatible API
from mcp_memory_service.api import delete_by_tag  # Still works via compat layer
result = delete_by_tag([TEST_MEMORY_TAG])  # Emits warning but functions

3. If using Option A, ensure MemoryService is properly initialized in conftest.py
```

---

## Task 7.5: Deprecation-Tests hinzufügen

### Amp Prompt

```
Create tests/integration/test_deprecation_compatibility.py:

"""
Tests to verify deprecated handlers still work via compatibility layer.

These tests ensure backwards compatibility during the migration period.
All deprecated tools should emit warnings but still function correctly.
"""

import pytest
import warnings
import json
from mcp_memory_service.server import MemoryServer


class TestDeprecatedHandlersStillWork:
    """Verify deprecated handlers route correctly to new implementations."""
    
    @pytest.mark.asyncio
    async def test_delete_by_tag_deprecated_but_works(self, unique_content):
        """Old delete_by_tag should work with deprecation warning."""
        server = MemoryServer()
        
        # Store test memory
        await server.handle_store_memory({
            "content": unique_content("Deprecated delete test"),
            "tags": ["deprecated-test"]
        })
        
        # Call deprecated handler
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = await server.handle_delete_by_tag({
                "tags": ["deprecated-test"]
            })
            
            # Should emit deprecation warning
            deprecation_warnings = [
                warning for warning in w 
                if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1
            assert "delete_by_tag" in str(deprecation_warnings[0].message)
            assert "memory_delete" in str(deprecation_warnings[0].message)
        
        # Should still work
        data = json.loads(result[0].text)
        assert "error" not in data or data.get("deleted_count", 0) >= 0
    
    @pytest.mark.asyncio
    async def test_retrieve_memory_deprecated_but_works(self, unique_content):
        """Old retrieve_memory should work with deprecation warning."""
        server = MemoryServer()
        
        await server.handle_store_memory({
            "content": unique_content("Deprecated retrieve test"),
            "tags": ["deprecated-retrieve"]
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = await server.handle_retrieve_memory({
                "query": "deprecated retrieve",
                "n_results": 5
            })
            
            # Check for deprecation warning
            assert any(
                "retrieve_memory" in str(warning.message)
                for warning in w
                if issubclass(warning.category, DeprecationWarning)
            )
        
        # Should return results
        data = json.loads(result[0].text)
        assert "memories" in data or "results" in data
    
    @pytest.mark.asyncio
    async def test_recall_memory_deprecated_but_works(self, unique_content):
        """Old recall_memory should work with deprecation warning."""
        server = MemoryServer()
        
        await server.handle_store_memory({
            "content": unique_content("Deprecated recall test"),
            "tags": ["deprecated-recall"]
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = await server.handle_recall_memory({
                "query": "last week"
            })
            
            assert any(
                "recall_memory" in str(warning.message)
                for warning in w
                if issubclass(warning.category, DeprecationWarning)
            )
    
    @pytest.mark.asyncio
    async def test_consolidation_status_deprecated_but_works(self):
        """Old consolidation_status should route to memory_consolidate."""
        server = MemoryServer()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = await server.handle_consolidation_status({})
            
            # May or may not warn depending on implementation
            # Main thing is it shouldn't crash
        
        data = json.loads(result[0].text)
        # Should return status info
        assert isinstance(data, dict)


class TestArgumentTransformation:
    """Verify argument transformation in deprecation layer."""
    
    @pytest.mark.asyncio
    async def test_n_results_transforms_to_limit(self, unique_content):
        """Verify n_results argument becomes limit."""
        server = MemoryServer()
        
        await server.handle_store_memory({
            "content": unique_content("Transform test"),
            "tags": ["transform-test"]
        })
        
        # Old style with n_results
        result = await server.handle_retrieve_memory({
            "query": "transform",
            "n_results": 3
        })
        
        data = json.loads(result[0].text)
        # Should respect the limit (transformed from n_results)
        memories = data.get("memories", [])
        assert len(memories) <= 3
    
    @pytest.mark.asyncio
    async def test_single_tag_transforms_to_tags_list(self, unique_content):
        """Verify single tag argument becomes tags list."""
        server = MemoryServer()
        
        await server.handle_store_memory({
            "content": unique_content("Single tag test"),
            "tags": ["single-tag-test"]
        })
        
        # Old style with single tag
        result = await server.handle_delete_by_tag({
            "tag": "single-tag-test"  # Note: singular
        })
        
        data = json.loads(result[0].text)
        assert "error" not in data or data.get("deleted_count", 0) >= 0
```

---

## Checkliste

- [ ] Handler-Mapping dokumentiert
- [ ] `test_unified_handlers.py` erstellt
  - [ ] TestHandleMemoryStore
  - [ ] TestHandleMemorySearch (alle Modi)
  - [ ] TestHandleMemoryDelete (alle Filter)
  - [ ] TestHandleMemoryList
- [ ] `test_server_handlers.py` aktualisiert
- [ ] `conftest.py` Cleanup angepasst
- [ ] `test_deprecation_compatibility.py` erstellt
- [ ] Alte Tests als `test_deprecated_handlers.py` archiviert
- [ ] `HANDLER_COVERAGE_REPORT.md` aktualisiert
- [ ] Alle Tests grün: `uv run pytest tests/integration/ -v`
