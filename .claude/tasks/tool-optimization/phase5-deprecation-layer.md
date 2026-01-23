# Phase 5: Deprecation Layer

## Ziel

Saubere Backwards Compatibility für alle umbenannten/konsolidierten Tools.

## Architektur

```
┌─────────────────────────────────────────────────────────────┐
│                    handle_call_tool()                        │
├─────────────────────────────────────────────────────────────┤
│  1. Check DEPRECATED_TOOLS mapping                          │
│     → If found: log warning, transform args, use new name   │
│                                                             │
│  2. Route to handler                                        │
│     → memory_store, memory_search, etc.                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Task 5.1: Create Deprecation Module

**Datei:** `src/mcp_memory_service/compat.py`

### Amp Prompt

```
Create deprecation compatibility layer in src/mcp_memory_service/compat.py:

"""
Backwards compatibility for deprecated MCP tools.

Strategy:
- Keep old tool names working for 2 major versions
- Log deprecation warnings to stderr
- Transform arguments to new format
- Route to new unified handlers

Usage in server_impl.py:
    from .compat import DEPRECATED_TOOLS, transform_deprecated_call
    
    if name in DEPRECATED_TOOLS:
        name, arguments = transform_deprecated_call(name, arguments)
"""

import logging
import warnings
from typing import Dict, Any, Tuple, Callable, Optional

logger = logging.getLogger(__name__)

# Type alias for argument transformer functions
ArgTransformer = Callable[[Dict[str, Any]], Dict[str, Any]]

# Mapping: old_name -> (new_name, argument_transformer)
DEPRECATED_TOOLS: Dict[str, Tuple[str, ArgTransformer]] = {
    # Phase 1: Delete consolidation
    "delete_memory": ("memory_delete", lambda a: {"content_hash": a["content_hash"]}),
    "delete_by_tag": ("memory_delete", lambda a: {"tags": [a["tag"]], "tag_match": "any"}),
    "delete_by_tags": ("memory_delete", lambda a: {"tags": a["tags"], "tag_match": "any"}),
    "delete_by_all_tags": ("memory_delete", lambda a: {"tags": a["tags"], "tag_match": "all"}),
    "delete_by_timeframe": ("memory_delete", lambda a: {
        "after": a.get("start_date"),
        "before": a.get("end_date"),
        "tags": [a["tag"]] if a.get("tag") else None
    }),
    "delete_before_date": ("memory_delete", lambda a: {
        "before": a["before_date"],
        "tags": [a["tag"]] if a.get("tag") else None
    }),
    
    # Phase 2: Search consolidation
    "retrieve_memory": ("memory_search", lambda a: {
        "query": a["query"],
        "limit": a.get("n_results", 5)
    }),
    "recall_memory": ("memory_search", lambda a: {
        "query": a.get("query"),
        "time_expr": a.get("time_expr") or a.get("query"),  # query might be time expression
        "limit": a.get("n_results", 5)
    }),
    "recall_by_timeframe": ("memory_search", lambda a: {
        "after": a.get("start_date"),
        "before": a.get("end_date"),
        "limit": a.get("n_results", 5)
    }),
    "retrieve_with_quality_boost": ("memory_search", lambda a: {
        "query": a["query"],
        "quality_boost": a.get("quality_weight", 0.3),
        "limit": a.get("n_results", 10),
        "mode": "hybrid"
    }),
    "exact_match_retrieve": ("memory_search", lambda a: {
        "query": a["content"],
        "mode": "exact"
    }),
    "debug_retrieve": ("memory_search", lambda a: {
        "query": a["query"],
        "limit": a.get("n_results", 5),
        "include_debug": True
    }),
    
    # Phase 3: Consolidation consolidation
    "consolidate_memories": ("memory_consolidate", lambda a: {
        "action": "run",
        "time_horizon": a["time_horizon"]
    }),
    "consolidation_status": ("memory_consolidate", lambda a: {"action": "status"}),
    "consolidation_recommendations": ("memory_consolidate", lambda a: {
        "action": "recommend",
        "time_horizon": a["time_horizon"]
    }),
    "scheduler_status": ("memory_consolidate", lambda a: {"action": "scheduler"}),
    "trigger_consolidation": ("memory_consolidate", lambda a: {
        "action": "run",
        "time_horizon": a["time_horizon"],
        "immediate": a.get("immediate", True)
    }),
    "pause_consolidation": ("memory_consolidate", lambda a: {
        "action": "pause",
        "time_horizon": a.get("time_horizon")
    }),
    "resume_consolidation": ("memory_consolidate", lambda a: {
        "action": "resume",
        "time_horizon": a.get("time_horizon")
    }),
    
    # Phase 4: Simple renames
    "store_memory": ("memory_store", lambda a: a),
    "check_database_health": ("memory_health", lambda a: a),
    "get_cache_stats": ("memory_stats", lambda a: a),
    "cleanup_duplicates": ("memory_cleanup", lambda a: a),
    "update_memory_metadata": ("memory_update", lambda a: a),
    
    # Phase 4: List merge
    "list_memories": ("memory_list", lambda a: a),
    "search_by_tag": ("memory_list", lambda a: {
        "tags": a.get("tags") or ([a["tag"]] if a.get("tag") else [])
    }),
    
    # Phase 4: Ingest merge
    "ingest_document": ("memory_ingest", lambda a: {
        "file_path": a["file_path"],
        "tags": a.get("tags", []),
        "chunk_size": a.get("chunk_size", 1000),
        "chunk_overlap": a.get("chunk_overlap", 200),
        "memory_type": a.get("memory_type", "document")
    }),
    "ingest_directory": ("memory_ingest", lambda a: {
        "directory_path": a["directory_path"],
        "tags": a.get("tags", []),
        "recursive": a.get("recursive", True),
        "file_extensions": a.get("file_extensions", ["pdf", "txt", "md", "json"]),
        "chunk_size": a.get("chunk_size", 1000),
        "max_files": a.get("max_files", 100)
    }),
    
    # Phase 4: Quality merge
    "rate_memory": ("memory_quality", lambda a: {
        "action": "rate",
        "content_hash": a["content_hash"],
        "rating": a["rating"],
        "feedback": a.get("feedback", "")
    }),
    "get_memory_quality": ("memory_quality", lambda a: {
        "action": "get",
        "content_hash": a["content_hash"]
    }),
    "analyze_quality_distribution": ("memory_quality", lambda a: {
        "action": "analyze",
        "min_quality": a.get("min_quality", 0.0),
        "max_quality": a.get("max_quality", 1.0)
    }),
    
    # Phase 4: Graph merge
    "find_connected_memories": ("memory_graph", lambda a: {
        "action": "connected",
        "hash": a["hash"],
        "max_hops": a.get("max_hops", 2)
    }),
    "find_shortest_path": ("memory_graph", lambda a: {
        "action": "path",
        "hash1": a["hash1"],
        "hash2": a["hash2"],
        "max_depth": a.get("max_depth", 5)
    }),
    "get_memory_subgraph": ("memory_graph", lambda a: {
        "action": "subgraph",
        "hash": a["hash"],
        "radius": a.get("radius", 2)
    }),
}


def transform_deprecated_call(
    tool_name: str, 
    arguments: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """
    Transform a deprecated tool call to the new format.
    
    Args:
        tool_name: Original (deprecated) tool name
        arguments: Original arguments
    
    Returns:
        Tuple of (new_tool_name, transformed_arguments)
    
    Raises:
        KeyError if tool_name not in DEPRECATED_TOOLS
    """
    if tool_name not in DEPRECATED_TOOLS:
        raise KeyError(f"Tool '{tool_name}' is not in deprecation mapping")
    
    new_name, transformer = DEPRECATED_TOOLS[tool_name]
    
    # Log deprecation warning
    logger.warning(
        f"Tool '{tool_name}' is deprecated and will be removed in a future version. "
        f"Use '{new_name}' instead."
    )
    
    # Also emit Python warning for programmatic detection
    warnings.warn(
        f"Tool '{tool_name}' is deprecated. Use '{new_name}' instead.",
        DeprecationWarning,
        stacklevel=3
    )
    
    # Transform arguments
    new_arguments = transformer(arguments or {})
    
    # Remove None values from transformed arguments
    new_arguments = {k: v for k, v in new_arguments.items() if v is not None}
    
    return new_name, new_arguments


def is_deprecated(tool_name: str) -> bool:
    """Check if a tool name is deprecated."""
    return tool_name in DEPRECATED_TOOLS


def get_new_tool_name(deprecated_name: str) -> Optional[str]:
    """Get the new tool name for a deprecated tool."""
    if deprecated_name in DEPRECATED_TOOLS:
        return DEPRECATED_TOOLS[deprecated_name][0]
    return None


def get_deprecation_message(tool_name: str) -> str:
    """Get a human-readable deprecation message for a tool."""
    if tool_name not in DEPRECATED_TOOLS:
        return f"Tool '{tool_name}' is not deprecated."
    
    new_name = DEPRECATED_TOOLS[tool_name][0]
    return f"Tool '{tool_name}' is deprecated. Please use '{new_name}' instead."
```

---

## Task 5.2: Integrate into Server

**Datei:** `src/mcp_memory_service/server_impl.py`

### Amp Prompt

```
Integrate deprecation layer into src/mcp_memory_service/server_impl.py:

1. Add import at top of file:

from .compat import DEPRECATED_TOOLS, transform_deprecated_call, is_deprecated

2. Modify handle_call_tool to use deprecation layer:

async def handle_call_tool(name: str, arguments: dict | None) -> List[types.TextContent]:
    """Handle tool calls with deprecation support."""
    
    # Handle deprecated tools
    if is_deprecated(name):
        name, arguments = transform_deprecated_call(name, arguments or {})
    
    # Ensure arguments is a dict
    arguments = arguments or {}
    
    # Route to handler (using NEW tool names only)
    if name == "memory_store":
        return await self.handle_memory_store(arguments)
    elif name == "memory_search":
        return await self.handle_memory_search(arguments)
    elif name == "memory_list":
        return await self.handle_memory_list(arguments)
    elif name == "memory_delete":
        return await self.handle_memory_delete(arguments)
    elif name == "memory_update":
        return await self.handle_memory_update(arguments)
    elif name == "memory_health":
        return await self.handle_memory_health(arguments)
    elif name == "memory_stats":
        return await self.handle_memory_stats(arguments)
    elif name == "memory_consolidate":
        return await self.handle_memory_consolidate(arguments)
    elif name == "memory_cleanup":
        return await self.handle_memory_cleanup(arguments)
    elif name == "memory_ingest":
        return await self.handle_memory_ingest(arguments)
    elif name == "memory_quality":
        return await self.handle_memory_quality(arguments)
    elif name == "memory_graph":
        return await self.handle_memory_graph(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")

3. In handle_list_tools, ONLY return the 12 new tools:
   - memory_store
   - memory_search
   - memory_list
   - memory_delete
   - memory_update
   - memory_health
   - memory_stats
   - memory_consolidate
   - memory_cleanup
   - memory_ingest
   - memory_quality
   - memory_graph

4. Remove all old tool definitions from handle_list_tools.
   The deprecation layer handles routing old names automatically.
```

---

## Task 5.3: Add Deprecation Documentation

**Datei:** `docs/MIGRATION.md` (neu erstellen)

```markdown
# Tool Migration Guide

## Overview

MCP Memory Service v2.0 consolidates 34 tools into 12 unified tools for better usability.

## Deprecated Tools

All old tool names continue to work but emit deprecation warnings.
They will be removed in v3.0.

### Delete Tools (6 → 1)

| Old | New | Example |
|-----|-----|---------|
| `delete_memory` | `memory_delete` | `{"content_hash": "abc123"}` |
| `delete_by_tag` | `memory_delete` | `{"tags": ["temp"], "tag_match": "any"}` |
| `delete_by_tags` | `memory_delete` | `{"tags": ["a", "b"], "tag_match": "any"}` |
| `delete_by_all_tags` | `memory_delete` | `{"tags": ["a", "b"], "tag_match": "all"}` |
| `delete_by_timeframe` | `memory_delete` | `{"after": "2024-01-01", "before": "2024-12-31"}` |
| `delete_before_date` | `memory_delete` | `{"before": "2024-01-01"}` |

### Search Tools (5 → 1)

| Old | New | Example |
|-----|-----|---------|
| `retrieve_memory` | `memory_search` | `{"query": "python", "limit": 5}` |
| `recall_memory` | `memory_search` | `{"query": "python", "time_expr": "last week"}` |
| `recall_by_timeframe` | `memory_search` | `{"after": "2024-01-01", "before": "2024-06-30"}` |
| `retrieve_with_quality_boost` | `memory_search` | `{"query": "python", "quality_boost": 0.3}` |
| `exact_match_retrieve` | `memory_search` | `{"query": "exact text", "mode": "exact"}` |

### Consolidation Tools (7 → 1)

| Old | New | Example |
|-----|-----|---------|
| `consolidate_memories` | `memory_consolidate` | `{"action": "run", "time_horizon": "weekly"}` |
| `consolidation_status` | `memory_consolidate` | `{"action": "status"}` |
| `consolidation_recommendations` | `memory_consolidate` | `{"action": "recommend", "time_horizon": "monthly"}` |
| `scheduler_status` | `memory_consolidate` | `{"action": "scheduler"}` |
| `trigger_consolidation` | `memory_consolidate` | `{"action": "run", "time_horizon": "daily"}` |
| `pause_consolidation` | `memory_consolidate` | `{"action": "pause"}` |
| `resume_consolidation` | `memory_consolidate` | `{"action": "resume"}` |

## New Unified Tools

1. **memory_store** - Store memories
2. **memory_search** - Semantic, exact, and time-based search
3. **memory_list** - Browse and filter memories
4. **memory_delete** - Flexible deletion with filters
5. **memory_update** - Update metadata
6. **memory_health** - System health check
7. **memory_stats** - Cache and performance stats
8. **memory_consolidate** - Consolidation management
9. **memory_cleanup** - Duplicate removal
10. **memory_ingest** - Document/directory import
11. **memory_quality** - Quality rating and analysis
12. **memory_graph** - Association graph operations
```

---

## Checkliste

- [ ] `compat.py` erstellt mit vollständigem DEPRECATED_TOOLS mapping
- [ ] `transform_deprecated_call` Funktion implementiert
- [ ] `server_impl.py` verwendet Deprecation Layer
- [ ] Alte Tool-Definitionen aus handle_list_tools entfernt
- [ ] Nur 12 neue Tools werden exponiert
- [ ] Deprecation Warnings werden geloggt
- [ ] MIGRATION.md Dokumentation erstellt
- [ ] Tests für Deprecation Layer
