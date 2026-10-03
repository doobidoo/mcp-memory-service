"""Tests for tools/registry.py and tools/routing.py — TDD-first (RED)."""

import pytest


# All 28 tools that must be in the registry
EXPECTED_TOOLS = [
    "memory_store",
    "memory_store_session",
    "memory_search",
    "memory_list",
    "memory_delete",
    "memory_cleanup",
    "memory_health",
    "memory_stats",
    "memory_update",
    "memory_consolidate",
    "memory_ingest",
    "memory_harvest",
    "memory_quality",
    "memory_graph",
    "memory_conflicts",
    "memory_resolve",
    "mistake_note_add",
    "mistake_note_search",
    "mistake_note_update",
    "mistake_note_delete",
    "get_quarantined_memories",
    "unquarantine_memory",
    "commit_session_legacy",
    "get_bootstrap_profile",
    "get_onboarding_guide",
    "memory_distill",
    "memory_explore",
    "memory_detail",
]


class TestToolRegistry:
    """Test TOOL_REGISTRY completeness and structure."""

    def test_registry_import(self):
        from mcp_memory_service.tools.registry import TOOL_REGISTRY
        assert TOOL_REGISTRY is not None

    def test_registry_has_all_tools(self):
        from mcp_memory_service.tools.registry import TOOL_REGISTRY
        registry_names = [t.name for t in TOOL_REGISTRY]
        for tool in EXPECTED_TOOLS:
            assert tool in registry_names, f"Missing tool: {tool}"

    def test_registry_no_duplicates(self):
        from mcp_memory_service.tools.registry import TOOL_REGISTRY
        names = [t.name for t in TOOL_REGISTRY]
        assert len(names) == len(set(names)), f"Duplicates: {[n for n in names if names.count(n) > 1]}"

    def test_memory_search_exposes_opt_in_summarize_flag(self):
        from mcp_memory_service.tools.registry import TOOL_REGISTRY

        search_tool = next(
            tool for tool in TOOL_REGISTRY
            if tool.name == "memory_search"
        )

        summarize = search_tool.input_schema["properties"].get("summarize")

        assert summarize is not None
        assert summarize["type"] == "boolean"
        assert summarize.get("default") is False
        assert "summarize" not in search_tool.input_schema.get("required", [])
        assert search_tool.annotations["readOnlyHint"] is True

    def test_tool_has_required_fields(self):
        from mcp_memory_service.tools.registry import TOOL_REGISTRY
        for tool in TOOL_REGISTRY:
            assert tool.name, "Tool missing name"
            assert tool.description, f"Tool {tool.name} missing description"
            assert tool.input_schema, f"Tool {tool.name} missing input_schema"
            assert "type" in tool.input_schema, f"Tool {tool.name} schema missing 'type'"

    def test_annotations_preserved(self):
        """Annotations drive OAuth scope — MUST be preserved (GHSA-2r68)."""
        from mcp_memory_service.tools.registry import TOOL_REGISTRY
        tools_with_annotations = [t for t in TOOL_REGISTRY if t.annotations]
        # At minimum, store/delete tools should have readOnlyHint=False
        store_tool = next(t for t in TOOL_REGISTRY if t.name == "memory_store")
        assert store_tool.annotations.get("destructiveHint") is False

    def test_two_phase_query_tools_read_only(self):
        """memory_explore/memory_detail are read-only — readOnlyHint MUST be set
        so the HTTP /mcp layer treats them as read-scope (GHSA-2r68-g678-7qr3)."""
        from mcp_memory_service.tools.registry import TOOL_REGISTRY
        for name in ("memory_explore", "memory_detail"):
            tool = next((t for t in TOOL_REGISTRY if t.name == name), None)
            assert tool is not None, f"{name} not registered"
            assert tool.annotations.get("readOnlyHint") is True, \
                f"{name} must declare readOnlyHint=True"

    def test_two_phase_query_tools_required_params(self):
        """Contract from #61: explore requires query, detail requires entity_id."""
        from mcp_memory_service.tools.registry import TOOL_REGISTRY
        explore = next(t for t in TOOL_REGISTRY if t.name == "memory_explore")
        detail = next(t for t in TOOL_REGISTRY if t.name == "memory_detail")
        assert explore.input_schema.get("required") == ["query"]
        assert detail.input_schema.get("required") == ["entity_id"]
        # store param accepted on both (composes with #62 multi-store scoping)
        assert "store" in explore.input_schema["properties"]
        assert "store" in detail.input_schema["properties"]

    def test_tooldef_frozen(self):
        """ToolDef should be immutable (frozen dataclass)."""
        from mcp_memory_service.tools.registry import TOOL_REGISTRY
        tool = TOOL_REGISTRY[0]
        with pytest.raises((AttributeError, TypeError)):
            tool.name = "hacked"


class TestRoutingTable:
    """Test ROUTING_TABLE completeness."""

    def test_routing_import(self):
        from mcp_memory_service.tools.routing import ROUTING_TABLE
        assert ROUTING_TABLE is not None

    def test_routing_covers_all_tools(self):
        from mcp_memory_service.tools.registry import TOOL_REGISTRY
        from mcp_memory_service.tools.routing import ROUTING_TABLE
        registry_names = {t.name for t in TOOL_REGISTRY}
        routing_names = set(ROUTING_TABLE.keys())
        missing = registry_names - routing_names
        assert not missing, f"Tools in registry but not in routing: {missing}"

    def test_legacy_aliases_removed_in_v11(self):
        """The v10 deprecated tool-name aliases must NOT route in V11 (Issue #53)."""
        from mcp_memory_service.tools.routing import ROUTING_TABLE
        removed = ["retrieve_memory", "recall_memory", "delete_by_tag", "search_by_tag",
                   "store_memory", "ingest_document", "consolidate_memories"]
        for name in removed:
            assert name not in ROUTING_TABLE, f"Deprecated alias still present in V11: {name}"

    def test_routing_values_are_callable_or_tuple(self):
        from mcp_memory_service.tools.routing import ROUTING_TABLE
        for name, handler in ROUTING_TABLE.items():
            assert callable(handler) or (isinstance(handler, tuple) and len(handler) == 2), \
                f"Routing entry {name} is not callable or (module, func) tuple"

    def test_routing_handlers_resolve(self):
        """All routing entries must resolve to actual functions or __self__ sentinels."""
        from mcp_memory_service.tools.routing import resolve_handler, ROUTING_TABLE
        # Test a sample — don't import ALL (some need server context)
        for name in ["memory_health", "memory_stats", "mistake_note_search"]:
            handler = resolve_handler(name)
            assert callable(handler), f"Handler for {name} not callable"
        # __self__ sentinel entries should resolve to tuple
        for name in ["memory_harvest", "commit_session_legacy", "get_bootstrap_profile"]:
            handler = resolve_handler(name)
            assert isinstance(handler, tuple) and handler[0] == "__self__", \
                f"Handler for {name} should be __self__ sentinel"
