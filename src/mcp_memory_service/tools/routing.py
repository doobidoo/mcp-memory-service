"""Routing table for MCP tool dispatch.

Maps tool names (and legacy aliases) to their handler functions via lazy
imports. This replaces the 59-branch elif chain in server_impl.py call_tool().

Handler resolution is lazy: modules are imported only on first call to
avoid circular imports and reduce cold-start time.
"""

from __future__ import annotations

from typing import Callable


def _handler(module: str, func: str) -> tuple[str, str]:
    """Create a lazy handler reference (module_path, function_name)."""
    return (f"mcp_memory_service.server.handlers.{module}", func)


# Primary tools (advertised in list_tools via TOOL_REGISTRY)
_PRIMARY: dict[str, tuple[str, str]] = {
    "memory_store": _handler("memory", "handle_store_memory"),
    "memory_store_session": _handler("memory", "handle_store_session"),
    "memory_search": _handler("memory", "handle_memory_search"),
    "memory_list": _handler("memory", "handle_memory_list"),
    "memory_delete": _handler("memory", "handle_memory_delete"),
    "memory_cleanup": _handler("memory", "handle_cleanup_duplicates"),
    "memory_health": _handler("utility", "handle_check_database_health"),
    "memory_stats": _handler("utility", "handle_get_cache_stats"),
    "memory_update": _handler("memory", "handle_update_memory_metadata"),
    "memory_consolidate": _handler("consolidation", "handle_memory_consolidate"),
    "memory_ingest": _handler("documents", "handle_memory_ingest"),
    "memory_harvest": ("__self__", "handle_memory_harvest"),
    "memory_quality": _handler("quality", "handle_memory_quality"),
    "memory_graph": _handler("graph", "handle_memory_graph"),
    "memory_explore": _handler("graph", "handle_memory_explore"),
    "memory_detail": _handler("graph", "handle_memory_detail"),
    "memory_conflicts": _handler("quality", "handle_memory_conflicts"),
    "memory_resolve": _handler("quality", "handle_memory_resolve"),
    "mistake_note_add": _handler("mistake_notes", "handle_mistake_note_add"),
    "mistake_note_search": _handler("mistake_notes", "handle_mistake_note_search"),
    "mistake_note_update": _handler("mistake_notes", "handle_mistake_note_update"),
    "mistake_note_delete": _handler("mistake_notes", "handle_mistake_note_delete"),
    "memory_quarantine_list": ("__self__", "handle_get_quarantined_memories"),
    "memory_quarantine_release": ("__self__", "handle_unquarantine_memory"),
    "memory_commit_session": ("__self__", "handle_commit_session_legacy"),
    "memory_bootstrap_profile": ("__self__", "handle_get_bootstrap_profile"),
    "memory_onboarding_guide": ("__self__", "handle_get_onboarding_guide"),
    "memory_distill": ("__self__", "handle_memory_distill"),
}

# Active, non-advertised handlers. These are NOT deprecated aliases and stay
# in V11. The v10 legacy tool-name aliases (the old compat.DEPRECATED_TOOLS
# set: retrieve_memory, store_memory, delete_by_tag, search_by_tag, ...) were
# removed here in Issue #53 Step 3 — those names are now simply unknown.
_LEGACY: dict[str, tuple[str, str]] = {
    "get_raw_embedding": _handler("memory", "handle_get_raw_embedding"),
    "memory_rate": _handler("quality", "handle_rate_memory"),
    "infer": _handler("graph", "handle_infer"),
    # Inline / non-advertised handlers (e.g. RFC #1008 auto-capture)
    "memory_observe": _handler("memory", "handle_memory_observe"),
    "memory_review": ("__self__", "handle_memory_review"),
    "memory_analysis": ("__self__", "handle_memory_analysis"),
    "knowledge_export": ("__self__", "handle_knowledge_export"),
    "learning_session": ("__self__", "handle_learning_session"),
}

# Combined routing table
ROUTING_TABLE: dict[str, tuple[str, str]] = {**_PRIMARY, **_LEGACY}

# Handler cache (populated on first resolve)
_cache: dict[str, Callable] = {}


def resolve_handler(name: str) -> Callable | tuple | None:
    """Resolve a tool name to its handler function (lazy import + cache).

    Returns a callable for module-based handlers, or a tuple ('__self__', method_name)
    for tools whose logic lives as instance methods on MemoryServer.
    """
    if name in _cache:
        return _cache[name]

    entry = ROUTING_TABLE.get(name)
    if entry is None:
        return None

    module_path, func_name = entry

    # Sentinel: handler is an instance method on MemoryServer
    if module_path == "__self__":
        _cache[name] = entry
        return entry

    import importlib  # inline import: only needed when a handler is first resolved
    mod = importlib.import_module(module_path)
    handler = getattr(mod, func_name, None)
    if handler is not None:
        _cache[name] = handler
    return handler
