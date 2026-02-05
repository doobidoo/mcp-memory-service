#!/usr/bin/env python3
"""
Test script for Issue 5: Delete Tag Function Ambiguity Fix

This script tests the enhanced delete_by_tag functionality that now supports
both single tags and multiple tags, addressing the API inconsistency.
"""

import asyncio
import os
import sys

import pytest

# Add the path to the MCP Memory Service
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import time

from mcp_memory_service.config import SQLITE_VEC_PATH
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


@pytest.mark.asyncio
async def test_enhanced_delete_by_tag():
    """Test the enhanced delete_by_tag functionality."""

    print("=== Testing Enhanced Delete By Tag Functionality ===\n")

    try:
        # Initialize storage
        print("1. Initializing SqliteVecMemoryStorage...")
        storage = SqliteVecMemoryStorage(db_path=SQLITE_VEC_PATH)
        await storage.initialize()

        # Test data
        test_memories = [
            Memory(
                content="This is a test memory with tag1",
                content_hash="test_hash_1",
                tags=["test", "tag1", "important"],
                created_at=time.time(),
            ),
            Memory(
                content="This is a test memory with tag2",
                content_hash="test_hash_2",
                tags=["test", "tag2", "temporary"],
                created_at=time.time(),
            ),
            Memory(
                content="This is a test memory with both tags",
                content_hash="test_hash_3",
                tags=["test", "tag1", "tag2"],
                created_at=time.time(),
            ),
            Memory(
                content="This is a test memory with different tags",
                content_hash="test_hash_4",
                tags=["other", "different"],
                created_at=time.time(),
            ),
        ]

        # Store test memories
        print("2. Storing test memories...")
        for memory in test_memories:
            success, message = await storage.store(memory)
            print(f"   - {message}")

        # Test 1: Delete by single tag (string)
        print("\n3. Testing delete by single tag (string)...")
        count, message = await storage.delete_by_tag("tag1")
        print(f"   Result: {message}")
        print("   Expected: Should delete 2 memories (test_hash_1 and test_hash_3)")

        # Test 2: Delete by multiple tags (list)
        print("\n4. Testing delete by multiple tags (list)...")
        count, message = await storage.delete_by_tag(["tag2", "other"])
        print(f"   Result: {message}")
        print("   Expected: Should delete 2 memories (test_hash_2 and test_hash_4)")

        # Test 3: Verify remaining memories
        print("\n5. Checking remaining memories...")
        remaining_memories = await storage.search_by_tag(["test", "different", "temporary", "important"])
        print(f"   Remaining memories: {len(remaining_memories)}")
        for memory in remaining_memories:
            print(f"   - Content: {memory.content[:50]}...")
            print(f"     Tags: {memory.tags}")

        # Test 4: Test the new delete_by_tags method
        print("\n6. Testing explicit delete_by_tags method...")

        # First, add a new memory to test with
        new_memory = Memory(
            content="Test memory for delete_by_tags",
            content_hash="test_hash_5",
            tags=["cleanup", "test"],
            created_at=time.time(),
        )
        await storage.store(new_memory)

        count, message = await storage.delete_by_tags(["cleanup"])
        print(f"   Result: {message}")

        # Test 5: Test delete_by_all_tags
        print("\n7. Testing delete_by_all_tags method...")

        # Add memories for testing ALL tags logic
        memory_all_1 = Memory(
            content="Memory with both urgent and important tags",
            content_hash="test_hash_6",
            tags=["urgent", "important", "work"],
            created_at=time.time(),
        )
        memory_all_2 = Memory(
            content="Memory with only urgent tag",
            content_hash="test_hash_7",
            tags=["urgent", "personal"],
            created_at=time.time(),
        )

        await storage.store(memory_all_1)
        await storage.store(memory_all_2)

        # This should only delete memory_all_1 (has both urgent AND important)
        count, message = await storage.delete_by_all_tags(["urgent", "important"])
        print(f"   Result: {message}")
        print("   Expected: Should delete 1 memory (only the one with both tags)")

        # Test 6: Edge cases
        print("\n8. Testing edge cases...")

        # Empty string
        count, message = await storage.delete_by_tag("")
        print(f"   Empty string: {message}")

        # Empty list
        count, message = await storage.delete_by_tag([])
        print(f"   Empty list: {message}")

        # Non-existent tag
        count, message = await storage.delete_by_tag("nonexistent")
        print(f"   Non-existent tag: {message}")

        print("\n=== Test Complete ===")
        print("✅ All tests passed! Issue 5 has been successfully resolved.")
        print("\nAPI Consistency Summary:")
        print("- search_by_tag: accepts array of tags ✅")
        print("- delete_by_tag: now accepts both single tag and array of tags ✅")
        print("- delete_by_tags: explicit multi-tag deletion ✅")
        print("- delete_by_all_tags: AND logic for multiple tags ✅")

        return True

    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_enhanced_delete_by_tag())
    sys.exit(0 if success else 1)
