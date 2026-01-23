# Tool Migration Guide

## Overview

MCP Memory Service v2.0 consolidates 34 tools into 12 unified tools for better usability and MCP best practices compliance.

**Timeline:**
- **v2.0+**: All old tool names continue to work with deprecation warnings
- **v3.0**: Old tool names will be removed (breaking change)

## Deprecated Tools

All old tool names continue to work but emit deprecation warnings. They will be removed in v3.0.

### Delete Tools (6 → 1)

**New unified tool:** `memory_delete`

| Old Tool | Migration Example |
|----------|------------------|
| `delete_memory` | `{"content_hash": "abc123"}` → `{"content_hash": "abc123"}` |
| `delete_by_tag` | `{"tag": "temp"}` → `{"tags": ["temp"], "tag_match": "any"}` |
| `delete_by_tags` | `{"tags": ["a", "b"]}` → `{"tags": ["a", "b"], "tag_match": "any"}` |
| `delete_by_all_tags` | `{"tags": ["a", "b"]}` → `{"tags": ["a", "b"], "tag_match": "all"}` |
| `delete_by_timeframe` | `{"start_date": "2024-01-01", "end_date": "2024-12-31"}` → `{"after": "2024-01-01", "before": "2024-12-31"}` |
| `delete_before_date` | `{"before_date": "2024-01-01"}` → `{"before": "2024-01-01"}` |

**New capabilities:**
- Combine filters (tags + time range)
- `dry_run` mode for previewing deletions
- `tag_match` mode: "any" (OR) or "all" (AND)

### Search Tools (6 → 1)

**New unified tool:** `memory_search`

| Old Tool | Migration Example |
|----------|------------------|
| `retrieve_memory` | `{"query": "python", "n_results": 5}` → `{"query": "python", "limit": 5}` |
| `recall_memory` | `{"query": "last week"}` → `{"query": "...", "time_expr": "last week"}` |
| `recall_by_timeframe` | `{"start_date": "2024-01-01", "end_date": "2024-06-30"}` → `{"after": "2024-01-01", "before": "2024-06-30"}` |
| `retrieve_with_quality_boost` | `{"query": "python", "quality_weight": 0.3}` → `{"query": "python", "mode": "hybrid", "quality_boost": 0.3}` |
| `exact_match_retrieve` | `{"content": "exact text"}` → `{"query": "exact text", "mode": "exact"}` |
| `debug_retrieve` | `{"query": "debug", "n_results": 5}` → `{"query": "debug", "include_debug": true}` |

**New capabilities:**
- Unified search modes: "semantic", "exact", "hybrid"
- Combine time + semantic search
- Quality-based reranking with `quality_boost`

### Consolidation Tools (7 → 1)

**New unified tool:** `memory_consolidate`

| Old Tool | Migration Example |
|----------|------------------|
| `consolidate_memories` | `{"time_horizon": "weekly"}` → `{"action": "run", "time_horizon": "weekly"}` |
| `consolidation_status` | `{}` → `{"action": "status"}` |
| `consolidation_recommendations` | `{"time_horizon": "monthly"}` → `{"action": "recommend", "time_horizon": "monthly"}` |
| `scheduler_status` | `{}` → `{"action": "scheduler"}` |
| `trigger_consolidation` | `{"time_horizon": "daily", "immediate": true}` → `{"action": "run", "time_horizon": "daily", "immediate": true}` |
| `pause_consolidation` | `{"time_horizon": "weekly"}` → `{"action": "pause", "time_horizon": "weekly"}` |
| `resume_consolidation` | `{"time_horizon": "weekly"}` → `{"action": "resume", "time_horizon": "weekly"}` |

**Actions:**
- `run`: Execute consolidation
- `status`: Get system health
- `recommend`: Get recommendations
- `scheduler`: Get scheduler status
- `pause`: Pause jobs
- `resume`: Resume jobs

### Simple Renames

| Old Tool | New Tool | Notes |
|----------|----------|-------|
| `store_memory` | `memory_store` | Arguments unchanged |
| `check_database_health` | `memory_health` | Arguments unchanged |
| `get_cache_stats` | `memory_stats` | Arguments unchanged |
| `cleanup_duplicates` | `memory_cleanup` | Arguments unchanged |
| `update_memory_metadata` | `memory_update` | Arguments unchanged |

### List Tools

| Old Tool | New Tool | Migration Example |
|----------|----------|------------------|
| `search_by_tag` | `memory_list` | `{"tags": ["python"]}` → `{"tags": ["python"]}` |
| *(none)* | `memory_list` | New: browse with pagination |

**New capabilities:**
- Pagination: `page`, `page_size`
- Filter by `tags` or `memory_type`

### Ingest Tools (2 → 1)

**New unified tool:** `memory_ingest`

| Old Tool | Migration Example |
|----------|------------------|
| `ingest_document` | `{"file_path": "/doc.pdf", "tags": ["docs"]}` → `{"file_path": "/doc.pdf", "tags": ["docs"]}` |
| `ingest_directory` | `{"directory_path": "/docs", "recursive": true}` → `{"directory_path": "/docs", "recursive": true}` |

**Mode detection:** Automatically uses file or directory mode based on which path parameter is provided.

### Quality Tools (3 → 1)

**New unified tool:** `memory_quality`

| Old Tool | Migration Example |
|----------|------------------|
| `rate_memory` | `{"content_hash": "abc", "rating": 1}` → `{"action": "rate", "content_hash": "abc", "rating": 1}` |
| `get_memory_quality` | `{"content_hash": "abc"}` → `{"action": "get", "content_hash": "abc"}` |
| `analyze_quality_distribution` | `{"min_quality": 0.5}` → `{"action": "analyze", "min_quality": 0.5}` |

**Actions:**
- `rate`: Manually rate a memory (-1, 0, 1)
- `get`: Get quality metrics for a specific memory
- `analyze`: Analyze quality distribution across all memories

### Graph Tools (3 → 1)

**New unified tool:** `memory_graph`

| Old Tool | Migration Example |
|----------|------------------|
| `find_connected_memories` | `{"hash": "abc", "max_hops": 2}` → `{"action": "connected", "hash": "abc", "max_hops": 2}` |
| `find_shortest_path` | `{"hash1": "abc", "hash2": "def"}` → `{"action": "path", "hash1": "abc", "hash2": "def"}` |
| `get_memory_subgraph` | `{"hash": "abc", "radius": 2}` → `{"action": "subgraph", "hash": "abc", "radius": 2}` |

**Actions:**
- `connected`: BFS traversal to find connected memories
- `path`: Find shortest path between two memories
- `subgraph`: Extract graph structure for visualization

## New Unified Tools (12 Total)

1. **memory_store** - Store memories with tags and metadata
2. **memory_search** - Semantic, exact, and time-based search with quality boost
3. **memory_list** - Browse and filter memories with pagination
4. **memory_delete** - Flexible deletion with tag/time/hash filters
5. **memory_update** - Update metadata without recreating memory
6. **memory_health** - System health check and statistics
7. **memory_stats** - Cache and performance stats
8. **memory_consolidate** - Dream-inspired consolidation management
9. **memory_cleanup** - Duplicate detection and removal
10. **memory_ingest** - Document/directory import with chunking
11. **memory_quality** - Quality rating, metrics, and analysis
12. **memory_graph** - Association graph operations and traversal

## Migration Strategy

### Immediate (No Code Changes Required)

All old tool names continue working automatically. The deprecation layer handles parameter transformation transparently.

**Action:** None required - old tools work as before with deprecation warnings

### Recommended (Before v3.0)

Update your code to use new tool names and take advantage of new capabilities:

```python
# Old (deprecated)
result = await mcp.call_tool("retrieve_memory", {
    "query": "python",
    "n_results": 10
})

# New (recommended)
result = await mcp.call_tool("memory_search", {
    "query": "python",
    "limit": 10,
    "mode": "semantic"
})
```

### Required (v3.0+)

All deprecated tool names will be removed. You must migrate to the new unified tools.

## Detecting Deprecation Warnings

Deprecation warnings are logged to stderr:

```
WARNING: Tool 'retrieve_memory' is deprecated and will be removed in a future version. Use 'memory_search' instead.
```

Python warnings are also emitted for programmatic detection:

```python
import warnings
warnings.filterwarnings('error', category=DeprecationWarning)
```

## Benefits of New Tools

1. **Fewer Tools**: 12 vs 34 - easier to learn and use
2. **Unified Interface**: Consistent action-based patterns
3. **More Flexible**: Combine filters, modes, and options
4. **Better Discovery**: Clearer naming with `memory_*` prefix
5. **MCP Compliance**: Follows best practice of 5-15 tools per server

## Support

- **Documentation**: See README.md and CLAUDE.md
- **Issues**: https://github.com/doobidoo/mcp-memory-service/issues
- **Discussions**: https://github.com/doobidoo/mcp-memory-service/discussions
