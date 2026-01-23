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
    "delete_by_tag": ("memory_delete", lambda a: {
        "tags": [a["tags"]] if isinstance(a.get("tags"), str) and "," not in a["tags"] else a["tags"],
        "tag_match": "any"
    }),
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
        "limit": a.get("n_results", 5),
        "mode": "semantic",
        "max_response_chars": a.get("max_response_chars")
    }),
    "recall_memory": ("memory_search", lambda a: {
        "query": a.get("query"),
        "time_expr": a.get("query"),  # query contains the time expression
        "limit": a.get("n_results", 5),
        "mode": "semantic",
        "max_response_chars": a.get("max_response_chars")
    }),
    "recall_by_timeframe": ("memory_search", lambda a: {
        "after": a.get("start_date"),
        "before": a.get("end_date"),
        "limit": a.get("n_results", 5),
        "max_response_chars": a.get("max_response_chars")
    }),
    "retrieve_with_quality_boost": ("memory_search", lambda a: {
        "query": a["query"],
        "quality_boost": a.get("quality_weight", 0.3),
        "limit": a.get("n_results", 10),
        "mode": "hybrid",
        "max_response_chars": a.get("max_response_chars")
    }),
    "exact_match_retrieve": ("memory_search", lambda a: {
        "query": a["content"],
        "mode": "exact"
    }),
    "debug_retrieve": ("memory_search", lambda a: {
        "query": a["query"],
        "limit": a.get("n_results", 5),
        "include_debug": True,
        "max_response_chars": a.get("max_response_chars")
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
        "tags": a.get("tags") if a.get("tags") else ([a["tag"]] if a.get("tag") else [])
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
