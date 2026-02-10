#!/usr/bin/env python3
"""
Quick test script to verify bug fixes for Issues #443, #444, #445, #446
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp_memory_service.storage.sqlite_vec import SQLiteVecStorage
from mcp_memory_service.models import Memory


async def test_exact_search_substring():
    """Test Issue #445: Exact search should do substring matching"""
    print("\n" + "="*60)
    print("Testing Issue #445: Exact Search Substring Matching")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        storage = SQLiteVecStorage(str(db_path))

        # Store test memories
        test_memories = [
            ("Python async programming is powerful", ["python", "async"]),
            ("Data validation failed in the pipeline", ["validation", "error"]),
            ("JavaScript code needs refactoring", ["javascript", "code"]),
        ]

        for content, tags in test_memories:
            memory = Memory(content=content, tags=tags)
            await storage.store_memory(memory)

        # Test 1: Substring match (should work after fix)
        print("\n1. Testing substring match: 'validation'")
        result = await storage.search_memories(query="validation", mode="exact")
        found = len(result["memories"])
        print(f"   Found {found} memories")
        assert found == 1, f"Expected 1 memory, got {found}"
        assert "validation" in result["memories"][0]["content"].lower()
        print("   ✅ Substring match works!")

        # Test 2: Case-insensitive match
        print("\n2. Testing case-insensitive match: 'PYTHON'")
        result = await storage.search_memories(query="PYTHON", mode="exact")
        found = len(result["memories"])
        print(f"   Found {found} memories")
        assert found == 1, f"Expected 1 memory, got {found}"
        print("   ✅ Case-insensitive match works!")

        # Test 3: No match for non-existent
        print("\n3. Testing non-existent string: 'nonexistent123'")
        result = await storage.search_memories(query="nonexistent123", mode="exact")
        found = len(result["memories"])
        print(f"   Found {found} memories")
        assert found == 0, f"Expected 0 memories, got {found}"
        print("   ✅ Empty result for non-existent works!")

    print("\n✅ All exact search tests passed!")


async def test_memory_search_dict_handling():
    """Test Issues #444, #446: Handle dict results from storage.search_memories()"""
    print("\n" + "="*60)
    print("Testing Issues #444, #446: Dict Access in Memory Search")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        storage = SQLiteVecStorage(str(db_path))

        # Store test memory
        memory = Memory(
            content="Test memory for dict access",
            tags=["test", "dict-access"],
            memory_type="note"
        )
        await storage.store_memory(memory)

        # Test semantic search returns dicts
        print("\n1. Testing semantic search returns dicts")
        result = await storage.search_memories(query="test memory", mode="semantic")
        memories = result["memories"]

        print(f"   Found {len(memories)} memories")
        assert len(memories) > 0, "Expected at least 1 memory"

        # Verify it's a dict, not a Memory object
        first_memory = memories[0]
        print(f"   Memory type: {type(first_memory)}")
        assert isinstance(first_memory, dict), f"Expected dict, got {type(first_memory)}"

        # Verify dict has expected keys
        print(f"   Memory keys: {list(first_memory.keys())[:5]}...")
        assert "content" in first_memory
        assert "tags" in first_memory
        assert "content_hash" in first_memory
        print("   ✅ Dict structure correct!")

        # Test hybrid search
        print("\n2. Testing hybrid search returns dicts")
        result = await storage.search_memories(query="test", mode="hybrid")
        memories = result["memories"]

        if memories:
            first_memory = memories[0]
            assert isinstance(first_memory, dict), f"Expected dict, got {type(first_memory)}"
            print("   ✅ Hybrid search returns dicts!")
        else:
            print("   ⚠️  No results (hybrid search may need configuration)")

    print("\n✅ All dict handling tests passed!")


async def test_import_fix():
    """Test Issue #443: Import path fix"""
    print("\n" + "="*60)
    print("Testing Issue #443: Import Path Fix")
    print("="*60)

    try:
        # Try the import that was broken
        from mcp_memory_service.server.utils.response_limiter import truncate_memories
        print("✅ Import successful: truncate_memories function available")

        # Test basic functionality
        test_memories = [
            {"content": "Memory 1", "content_hash": "hash1", "tags": ["test"], "created_at": "2024-01-01"},
            {"content": "Memory 2", "content_hash": "hash2", "tags": ["test"], "created_at": "2024-01-02"},
        ]

        truncated, meta = truncate_memories(test_memories, max_chars=100)
        print(f"   Truncation works: {len(truncated)} memories, {meta['total_chars']} chars")
        print("   ✅ response_limiter module works!")

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        raise

    print("\n✅ Import fix verified!")


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MCP Memory Service - Bug Fix Verification")
    print("Issues: #443, #444, #445, #446")
    print("="*60)

    try:
        await test_import_fix()
        await test_exact_search_substring()
        await test_memory_search_dict_handling()

        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
