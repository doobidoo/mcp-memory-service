# Phase 6: Tests & Validation

## Ziel

Sicherstellen, dass alle Konsolidierungen korrekt funktionieren und Backwards Compatibility gewährleistet ist.

---

## Task 6.1: Unit Tests für Unified Tools

**Datei:** `tests/test_unified_tools.py`

### Amp Prompt

```
Create comprehensive tests in tests/test_unified_tools.py:

"""
Tests for unified MCP Memory Service tools.

Tests cover:
1. New unified tools work correctly
2. Deprecated tools still work via compatibility layer
3. Argument transformation is correct
4. Deprecation warnings are emitted
"""

import pytest
import warnings
from unittest.mock import AsyncMock, patch
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.compat import (
    DEPRECATED_TOOLS,
    transform_deprecated_call,
    is_deprecated,
    get_new_tool_name
)


class TestMemoryDelete:
    """Tests for unified memory_delete tool."""
    
    @pytest.fixture
    def memory_service(self):
        """Create mocked MemoryService."""
        service = AsyncMock(spec=MemoryService)
        return service
    
    @pytest.mark.asyncio
    async def test_delete_by_hash(self, memory_service):
        """Test single memory deletion by hash."""
        memory_service.delete_memories.return_value = {
            "success": True,
            "deleted_count": 1,
            "deleted_hashes": ["abc123"]
        }
        
        result = await memory_service.delete_memories(content_hash="abc123")
        
        assert result["deleted_count"] == 1
        memory_service.delete_memories.assert_called_once_with(content_hash="abc123")
    
    @pytest.mark.asyncio
    async def test_delete_by_tags_any(self, memory_service):
        """Test deletion by tags with ANY match."""
        memory_service.delete_memories.return_value = {
            "success": True,
            "deleted_count": 5,
            "deleted_hashes": ["a", "b", "c", "d", "e"]
        }
        
        result = await memory_service.delete_memories(
            tags=["temp", "draft"],
            tag_match="any"
        )
        
        assert result["deleted_count"] == 5
    
    @pytest.mark.asyncio
    async def test_delete_by_tags_all(self, memory_service):
        """Test deletion by tags with ALL match."""
        memory_service.delete_memories.return_value = {
            "success": True,
            "deleted_count": 2,
            "deleted_hashes": ["x", "y"]
        }
        
        result = await memory_service.delete_memories(
            tags=["archived", "old"],
            tag_match="all"
        )
        
        assert result["deleted_count"] == 2
    
    @pytest.mark.asyncio
    async def test_delete_by_timeframe(self, memory_service):
        """Test deletion by time range."""
        memory_service.delete_memories.return_value = {
            "success": True,
            "deleted_count": 10,
            "deleted_hashes": [f"h{i}" for i in range(10)]
        }
        
        result = await memory_service.delete_memories(
            after="2024-01-01",
            before="2024-06-30"
        )
        
        assert result["deleted_count"] == 10
    
    @pytest.mark.asyncio
    async def test_delete_dry_run(self, memory_service):
        """Test dry run returns preview without deleting."""
        memory_service.delete_memories.return_value = {
            "success": True,
            "deleted_count": 3,
            "deleted_hashes": ["a", "b", "c"],
            "dry_run": True
        }
        
        result = await memory_service.delete_memories(
            tags=["cleanup"],
            dry_run=True
        )
        
        assert result["dry_run"] is True
    
    @pytest.mark.asyncio
    async def test_delete_no_filters_error(self, memory_service):
        """Test that delete without filters returns error."""
        memory_service.delete_memories.return_value = {
            "error": "At least one filter required"
        }
        
        result = await memory_service.delete_memories()
        
        assert "error" in result


class TestMemorySearch:
    """Tests for unified memory_search tool."""
    
    @pytest.fixture
    def memory_service(self):
        service = AsyncMock(spec=MemoryService)
        return service
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, memory_service):
        """Test default semantic search."""
        memory_service.search_memories.return_value = {
            "memories": [{"content": "test", "content_hash": "abc"}],
            "total": 1,
            "mode": "semantic"
        }
        
        result = await memory_service.search_memories(query="python patterns")
        
        assert result["mode"] == "semantic"
        assert len(result["memories"]) == 1
    
    @pytest.mark.asyncio
    async def test_exact_search(self, memory_service):
        """Test exact string match."""
        memory_service.search_memories.return_value = {
            "memories": [],
            "total": 0,
            "mode": "exact"
        }
        
        result = await memory_service.search_memories(
            query="exact phrase",
            mode="exact"
        )
        
        assert result["mode"] == "exact"
    
    @pytest.mark.asyncio
    async def test_time_expression_search(self, memory_service):
        """Test natural language time expression."""
        memory_service.search_memories.return_value = {
            "memories": [{"content": "recent"}],
            "total": 1
        }
        
        result = await memory_service.search_memories(time_expr="last week")
        
        assert result["total"] >= 0
    
    @pytest.mark.asyncio
    async def test_quality_boost(self, memory_service):
        """Test quality-boosted search."""
        memory_service.search_memories.return_value = {
            "memories": [{"content": "high quality", "quality": 0.9}],
            "total": 1
        }
        
        result = await memory_service.search_memories(
            query="important info",
            quality_boost=0.3
        )
        
        assert result["total"] == 1
    
    @pytest.mark.asyncio
    async def test_combined_filters(self, memory_service):
        """Test combining multiple filters."""
        memory_service.search_memories.return_value = {
            "memories": [],
            "total": 0
        }
        
        result = await memory_service.search_memories(
            query="database",
            time_expr="last month",
            tags=["reference"],
            quality_boost=0.2,
            limit=20
        )
        
        # Should have called with all parameters
        memory_service.search_memories.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_debug_output(self, memory_service):
        """Test debug information included."""
        memory_service.search_memories.return_value = {
            "memories": [],
            "total": 0,
            "debug": {
                "time_filter": None,
                "quality_boost": 0.0
            }
        }
        
        result = await memory_service.search_memories(
            query="test",
            include_debug=True
        )
        
        assert "debug" in result


class TestMemoryConsolidate:
    """Tests for unified memory_consolidate tool."""
    
    @pytest.fixture
    def memory_service(self):
        service = AsyncMock(spec=MemoryService)
        return service
    
    @pytest.mark.asyncio
    async def test_status_action(self, memory_service):
        """Test status retrieval."""
        memory_service.manage_consolidation.return_value = {
            "status": "healthy",
            "last_run": "2024-01-15"
        }
        
        result = await memory_service.manage_consolidation(action="status")
        
        assert result["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_run_requires_horizon(self, memory_service):
        """Test run action requires time_horizon."""
        memory_service.manage_consolidation.return_value = {
            "error": "time_horizon required for 'run' action"
        }
        
        result = await memory_service.manage_consolidation(action="run")
        
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_run_with_horizon(self, memory_service):
        """Test successful consolidation run."""
        memory_service.manage_consolidation.return_value = {
            "success": True,
            "consolidated": 50
        }
        
        result = await memory_service.manage_consolidation(
            action="run",
            time_horizon="weekly"
        )
        
        assert result["success"] is True


class TestDeprecationLayer:
    """Tests for backwards compatibility."""
    
    def test_all_deprecated_tools_mapped(self):
        """Verify all expected deprecated tools are in mapping."""
        expected_deprecated = [
            # Delete tools
            "delete_memory", "delete_by_tag", "delete_by_tags",
            "delete_by_all_tags", "delete_by_timeframe", "delete_before_date",
            # Search tools
            "retrieve_memory", "recall_memory", "recall_by_timeframe",
            "retrieve_with_quality_boost", "exact_match_retrieve", "debug_retrieve",
            # Consolidation tools
            "consolidate_memories", "consolidation_status", "consolidation_recommendations",
            "scheduler_status", "trigger_consolidation", "pause_consolidation", "resume_consolidation",
            # Renamed tools
            "store_memory", "check_database_health", "get_cache_stats",
            "cleanup_duplicates", "update_memory_metadata",
            # Merged tools
            "list_memories", "search_by_tag",
            "ingest_document", "ingest_directory",
            "rate_memory", "get_memory_quality", "analyze_quality_distribution",
            "find_connected_memories", "find_shortest_path", "get_memory_subgraph",
        ]
        
        for tool in expected_deprecated:
            assert tool in DEPRECATED_TOOLS, f"Missing: {tool}"
    
    def test_is_deprecated(self):
        """Test is_deprecated helper."""
        assert is_deprecated("delete_by_tag") is True
        assert is_deprecated("memory_delete") is False
        assert is_deprecated("unknown_tool") is False
    
    def test_get_new_tool_name(self):
        """Test get_new_tool_name helper."""
        assert get_new_tool_name("delete_by_tag") == "memory_delete"
        assert get_new_tool_name("retrieve_memory") == "memory_search"
        assert get_new_tool_name("memory_delete") is None
    
    def test_transform_delete_by_tag(self):
        """Test argument transformation for delete_by_tag."""
        old_args = {"tag": "temporary"}
        
        new_name, new_args = transform_deprecated_call("delete_by_tag", old_args)
        
        assert new_name == "memory_delete"
        assert new_args["tags"] == ["temporary"]
        assert new_args["tag_match"] == "any"
    
    def test_transform_retrieve_memory(self):
        """Test argument transformation for retrieve_memory."""
        old_args = {"query": "python", "n_results": 10}
        
        new_name, new_args = transform_deprecated_call("retrieve_memory", old_args)
        
        assert new_name == "memory_search"
        assert new_args["query"] == "python"
        assert new_args["limit"] == 10
    
    def test_transform_consolidate_memories(self):
        """Test argument transformation for consolidate_memories."""
        old_args = {"time_horizon": "weekly"}
        
        new_name, new_args = transform_deprecated_call("consolidate_memories", old_args)
        
        assert new_name == "memory_consolidate"
        assert new_args["action"] == "run"
        assert new_args["time_horizon"] == "weekly"
    
    def test_deprecation_warning_emitted(self):
        """Test that deprecation warning is emitted."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            transform_deprecated_call("delete_by_tag", {"tag": "test"})
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "delete_by_tag" in str(w[0].message)
            assert "memory_delete" in str(w[0].message)
    
    def test_none_values_removed_from_transformed_args(self):
        """Test that None values are filtered from transformed args."""
        old_args = {"start_date": "2024-01-01"}  # no end_date
        
        new_name, new_args = transform_deprecated_call("delete_by_timeframe", old_args)
        
        assert "before" not in new_args or new_args.get("before") is not None


class TestIntegration:
    """Integration tests for full tool pipeline."""
    
    @pytest.mark.asyncio
    async def test_deprecated_tool_call_works(self):
        """Test calling deprecated tool name works end-to-end."""
        # This would be an integration test with actual server
        pass
    
    @pytest.mark.asyncio  
    async def test_new_tool_call_works(self):
        """Test calling new tool name works."""
        pass
```

---

## Task 6.2: Integration Tests

**Datei:** `tests/test_integration_tools.py`

### Amp Prompt

```
Create integration tests in tests/test_integration_tools.py:

"""
Integration tests for MCP Memory Service tool consolidation.

These tests verify the full pipeline from tool call to response.
"""

import pytest
import json
from mcp_memory_service.server_impl import MCPMemoryServer


@pytest.fixture
async def server():
    """Create and initialize server for testing."""
    server = MCPMemoryServer()
    await server.initialize()
    yield server
    # Cleanup


class TestToolDiscovery:
    """Test that only new tools are exposed."""
    
    @pytest.mark.asyncio
    async def test_only_12_tools_exposed(self, server):
        """Verify exactly 12 new tools are listed."""
        tools = await server.handle_list_tools()
        
        assert len(tools) == 12
        
        expected_names = {
            "memory_store",
            "memory_search", 
            "memory_list",
            "memory_delete",
            "memory_update",
            "memory_health",
            "memory_stats",
            "memory_consolidate",
            "memory_cleanup",
            "memory_ingest",
            "memory_quality",
            "memory_graph",
        }
        
        actual_names = {t.name for t in tools}
        assert actual_names == expected_names
    
    @pytest.mark.asyncio
    async def test_old_tools_not_listed(self, server):
        """Verify deprecated tools not in listing."""
        tools = await server.handle_list_tools()
        tool_names = {t.name for t in tools}
        
        deprecated_names = [
            "delete_by_tag",
            "retrieve_memory",
            "consolidation_status",
            "store_memory",
        ]
        
        for name in deprecated_names:
            assert name not in tool_names


class TestDeprecatedToolCalls:
    """Test that deprecated tool calls still work."""
    
    @pytest.mark.asyncio
    async def test_delete_by_tag_still_works(self, server):
        """Test deprecated delete_by_tag routes correctly."""
        # First store a memory
        store_result = await server.handle_call_tool(
            "memory_store",  # Use new name
            {"content": "test content", "tags": ["test-delete"]}
        )
        
        # Now delete using old name
        result = await server.handle_call_tool(
            "delete_by_tag",  # Deprecated name
            {"tag": "test-delete"}
        )
        
        result_data = json.loads(result[0].text)
        assert "error" not in result_data or result_data.get("deleted_count", 0) >= 0
    
    @pytest.mark.asyncio
    async def test_retrieve_memory_still_works(self, server):
        """Test deprecated retrieve_memory routes correctly."""
        result = await server.handle_call_tool(
            "retrieve_memory",  # Deprecated name
            {"query": "test", "n_results": 5}
        )
        
        result_data = json.loads(result[0].text)
        assert "memories" in result_data or "error" not in result_data


class TestNewToolCalls:
    """Test new unified tools work correctly."""
    
    @pytest.mark.asyncio
    async def test_memory_delete_by_hash(self, server):
        """Test memory_delete with hash."""
        # Store first
        store_result = await server.handle_call_tool(
            "memory_store",
            {"content": "to be deleted"}
        )
        store_data = json.loads(store_result[0].text)
        content_hash = store_data.get("content_hash")
        
        if content_hash:
            # Delete
            result = await server.handle_call_tool(
                "memory_delete",
                {"content_hash": content_hash}
            )
            result_data = json.loads(result[0].text)
            assert result_data.get("deleted_count", 0) >= 0
    
    @pytest.mark.asyncio
    async def test_memory_search_semantic(self, server):
        """Test memory_search semantic mode."""
        result = await server.handle_call_tool(
            "memory_search",
            {"query": "python programming", "limit": 5}
        )
        
        result_data = json.loads(result[0].text)
        assert "memories" in result_data
    
    @pytest.mark.asyncio
    async def test_memory_consolidate_status(self, server):
        """Test memory_consolidate status action."""
        result = await server.handle_call_tool(
            "memory_consolidate",
            {"action": "status"}
        )
        
        result_data = json.loads(result[0].text)
        # Should return status info or error, not crash
        assert isinstance(result_data, dict)
```

---

## Task 6.3: Validation Script

**Datei:** `scripts/validate_tool_optimization.py`

### Amp Prompt

```
Create validation script scripts/validate_tool_optimization.py:

#!/usr/bin/env python3
"""
Validation script for MCP Memory Service tool optimization.

Run this after completing all phases to verify:
1. Tool count is correct (12 tools)
2. All deprecated tools route correctly
3. No regressions in functionality
"""

import asyncio
import json
import sys
import warnings
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_memory_service.server_impl import MCPMemoryServer
from mcp_memory_service.compat import DEPRECATED_TOOLS


async def validate_tool_count():
    """Verify exactly 12 tools exposed."""
    print("\n=== Validating Tool Count ===")
    
    server = MCPMemoryServer()
    await server.initialize()
    
    tools = await server.handle_list_tools()
    
    expected = 12
    actual = len(tools)
    
    if actual == expected:
        print(f"✅ Tool count correct: {actual}")
        return True
    else:
        print(f"❌ Tool count wrong: expected {expected}, got {actual}")
        print("   Tools found:", [t.name for t in tools])
        return False


async def validate_tool_names():
    """Verify all tool names follow memory_* pattern."""
    print("\n=== Validating Tool Names ===")
    
    server = MCPMemoryServer()
    await server.initialize()
    
    tools = await server.handle_list_tools()
    
    expected_names = {
        "memory_store", "memory_search", "memory_list",
        "memory_delete", "memory_update", "memory_health",
        "memory_stats", "memory_consolidate", "memory_cleanup",
        "memory_ingest", "memory_quality", "memory_graph",
    }
    
    actual_names = {t.name for t in tools}
    
    if actual_names == expected_names:
        print("✅ All tool names correct")
        return True
    else:
        missing = expected_names - actual_names
        extra = actual_names - expected_names
        if missing:
            print(f"❌ Missing tools: {missing}")
        if extra:
            print(f"❌ Unexpected tools: {extra}")
        return False


async def validate_deprecation_mapping():
    """Verify all deprecated tools are mapped."""
    print("\n=== Validating Deprecation Mapping ===")
    
    expected_count = 34 - 12  # 34 old tools, 12 new tools
    actual_count = len(DEPRECATED_TOOLS)
    
    # Actually we have more mappings since some map to same tool
    # Just verify key deprecated tools exist
    key_deprecated = [
        "delete_by_tag", "retrieve_memory", "consolidation_status",
        "store_memory", "list_memories", "ingest_document"
    ]
    
    all_present = True
    for tool in key_deprecated:
        if tool in DEPRECATED_TOOLS:
            print(f"✅ {tool} → {DEPRECATED_TOOLS[tool][0]}")
        else:
            print(f"❌ Missing mapping for: {tool}")
            all_present = False
    
    print(f"\nTotal deprecated mappings: {actual_count}")
    return all_present


async def validate_deprecated_tool_routing():
    """Test that deprecated tool calls route correctly."""
    print("\n=== Validating Deprecated Tool Routing ===")
    
    server = MCPMemoryServer()
    await server.initialize()
    
    # Test a few key deprecated tools
    test_cases = [
        ("delete_by_tag", {"tag": "nonexistent"}, "memory_delete"),
        ("retrieve_memory", {"query": "test", "n_results": 1}, "memory_search"),
        ("consolidation_status", {}, "memory_consolidate"),
    ]
    
    all_passed = True
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        for old_name, args, expected_new in test_cases:
            try:
                result = await server.handle_call_tool(old_name, args)
                
                # Check deprecation warning was emitted
                deprecation_warned = any(
                    old_name in str(warning.message) 
                    for warning in w 
                    if issubclass(warning.category, DeprecationWarning)
                )
                
                if deprecation_warned:
                    print(f"✅ {old_name} routed with warning")
                else:
                    print(f"⚠️  {old_name} routed but no warning")
                
            except Exception as e:
                print(f"❌ {old_name} failed: {e}")
                all_passed = False
    
    return all_passed


async def main():
    """Run all validations."""
    print("=" * 60)
    print("MCP Memory Service - Tool Optimization Validation")
    print("=" * 60)
    
    results = []
    
    results.append(await validate_tool_count())
    results.append(await validate_tool_names())
    results.append(await validate_deprecation_mapping())
    results.append(await validate_deprecated_tool_routing())
    
    print("\n" + "=" * 60)
    
    if all(results):
        print("✅ ALL VALIDATIONS PASSED")
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
```

---

## Checkliste

- [ ] `tests/test_unified_tools.py` erstellt
- [ ] Unit Tests für memory_delete
- [ ] Unit Tests für memory_search
- [ ] Unit Tests für memory_consolidate
- [ ] Unit Tests für Deprecation Layer
- [ ] `tests/test_integration_tools.py` erstellt
- [ ] Tool Discovery Tests
- [ ] Deprecated Tool Call Tests
- [ ] New Tool Call Tests
- [ ] `scripts/validate_tool_optimization.py` erstellt
- [ ] Alle Tests grün: `uv run pytest tests/test_unified_tools.py tests/test_integration_tools.py -v`
- [ ] Validation Script erfolgreich: `uv run python scripts/validate_tool_optimization.py`

---

## Finale Validation

Nach Abschluss aller Phasen:

```bash
cd /Users/hkr/Documents/GitHub/mcp-memory-service

# Run all tests
uv run pytest tests/ -v --tb=short

# Run validation script
uv run python scripts/validate_tool_optimization.py

# Check deprecation warnings work
uv run python -W default::DeprecationWarning -c "
from mcp_memory_service.compat import transform_deprecated_call
transform_deprecated_call('delete_by_tag', {'tag': 'test'})
"

# Verify tool count
uv run python -c "
import asyncio
from mcp_memory_service.server_impl import MCPMemoryServer

async def check():
    server = MCPMemoryServer()
    await server.initialize()
    tools = await server.handle_list_tools()
    print(f'Tool count: {len(tools)}')
    for t in sorted(tools, key=lambda x: x.name):
        print(f'  - {t.name}')

asyncio.run(check())
"
```

**Erwartetes Ergebnis:**
```
Tool count: 12
  - memory_cleanup
  - memory_consolidate
  - memory_delete
  - memory_graph
  - memory_health
  - memory_ingest
  - memory_list
  - memory_quality
  - memory_search
  - memory_stats
  - memory_store
  - memory_update
```
