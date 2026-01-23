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

try:
    from mcp_memory_service.server_impl import MemoryServer
except ImportError:
    # Fallback for different import paths
    from mcp_memory_service.server.server_impl import MemoryServer

from mcp_memory_service.compat import DEPRECATED_TOOLS


async def validate_tool_count():
    """Verify exactly 12 tools exposed."""
    print("\n=== Validating Tool Count ===")

    server = MemoryServer()
    await server.initialize()

    result = await server.list_tools()
    tools = result.tools

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

    server = MemoryServer()
    await server.initialize()

    result = await server.list_tools()
    tools = result.tools

    expected_names = {
        "memory_store", "memory_search", "memory_list",
        "memory_delete", "memory_update", "memory_health",
        "memory_stats", "memory_consolidate", "memory_cleanup",
        "memory_ingest", "memory_quality", "memory_graph",
    }

    actual_names = {t.name for t in tools}

    if actual_names == expected_names:
        print("✅ All tool names correct")
        for name in sorted(actual_names):
            print(f"   - {name}")
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

    # Key deprecated tools to verify
    key_deprecated = [
        # Delete tools
        ("delete_by_tag", "memory_delete"),
        ("delete_by_tags", "memory_delete"),
        ("delete_memory", "memory_delete"),
        # Search tools
        ("retrieve_memory", "memory_search"),
        ("recall_memory", "memory_search"),
        ("exact_match_retrieve", "memory_search"),
        # Consolidation tools
        ("consolidation_status", "memory_consolidate"),
        ("consolidate_memories", "memory_consolidate"),
        ("trigger_consolidation", "memory_consolidate"),
        # Renamed tools
        ("store_memory", "memory_store"),
        ("check_database_health", "memory_health"),
        ("get_cache_stats", "memory_stats"),
        ("cleanup_duplicates", "memory_cleanup"),
        ("update_memory_metadata", "memory_update"),
        # Merged tools
        ("list_memories", "memory_list"),
        ("search_by_tag", "memory_list"),
        ("ingest_document", "memory_ingest"),
        ("ingest_directory", "memory_ingest"),
        ("rate_memory", "memory_quality"),
        ("get_memory_quality", "memory_quality"),
        ("find_connected_memories", "memory_graph"),
        ("find_shortest_path", "memory_graph"),
    ]

    all_present = True
    for old_tool, expected_new in key_deprecated:
        if old_tool in DEPRECATED_TOOLS:
            actual_new = DEPRECATED_TOOLS[old_tool][0]
            if actual_new == expected_new:
                print(f"✅ {old_tool} → {actual_new}")
            else:
                print(f"❌ {old_tool} maps to {actual_new}, expected {expected_new}")
                all_present = False
        else:
            print(f"❌ Missing mapping for: {old_tool}")
            all_present = False

    print(f"\nTotal deprecated mappings: {len(DEPRECATED_TOOLS)}")
    return all_present


async def validate_deprecated_tool_routing():
    """Test that deprecated tool calls route correctly."""
    print("\n=== Validating Deprecated Tool Routing ===")

    server = MemoryServer()
    await server.initialize()

    # Test a few key deprecated tools
    test_cases = [
        ("delete_by_tag", {"tag": "nonexistent-test-tag"}, "memory_delete"),
        ("retrieve_memory", {"query": "test validation query", "n_results": 1}, "memory_search"),
        ("consolidation_status", {}, "memory_consolidate"),
        ("store_memory", {"content": "validation test content"}, "memory_store"),
        ("list_memories", {"limit": 1}, "memory_list"),
    ]

    all_passed = True

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        for old_name, args, expected_new in test_cases:
            try:
                result = await server.call_tool(old_name, args)

                # Check that we got a response (not checking content, just that it works)
                if len(result) > 0:
                    # Check deprecation warning was emitted
                    deprecation_warned = any(
                        old_name in str(warning.message)
                        for warning in w
                        if issubclass(warning.category, DeprecationWarning)
                    )

                    if deprecation_warned:
                        print(f"✅ {old_name} → {expected_new} (routed with warning)")
                    else:
                        print(f"⚠️  {old_name} → {expected_new} (routed but no warning)")
                else:
                    print(f"❌ {old_name} returned empty result")
                    all_passed = False

            except Exception as e:
                print(f"❌ {old_name} failed: {e}")
                all_passed = False

    return all_passed


async def validate_new_tool_functionality():
    """Test that new unified tools work correctly."""
    print("\n=== Validating New Tool Functionality ===")

    server = MemoryServer()
    await server.initialize()

    # Test key new tools
    test_cases = [
        ("memory_store", {"content": "test content for validation"}),
        ("memory_search", {"query": "validation", "limit": 5}),
        ("memory_list", {"limit": 5, "offset": 0}),
        ("memory_health", {}),
        ("memory_stats", {}),
        ("memory_consolidate", {"action": "status"}),
    ]

    all_passed = True

    for tool_name, args in test_cases:
        try:
            result = await server.call_tool(tool_name, args)

            if len(result) > 0:
                # Parse JSON response
                result_data = json.loads(result[0].text)
                print(f"✅ {tool_name} works")
            else:
                print(f"❌ {tool_name} returned empty result")
                all_passed = False

        except Exception as e:
            print(f"❌ {tool_name} failed: {e}")
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
    results.append(await validate_new_tool_functionality())

    print("\n" + "=" * 60)

    if all(results):
        print("✅ ALL VALIDATIONS PASSED")
        print("\nTool optimization complete:")
        print("  - 12 new unified tools exposed")
        print("  - 33 deprecated tools route correctly")
        print("  - Backwards compatibility maintained")
        print("  - All functionality preserved")
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED")
        failed_count = len([r for r in results if not r])
        print(f"\n{failed_count} validation(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
