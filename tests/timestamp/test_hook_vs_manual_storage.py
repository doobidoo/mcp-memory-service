#!/usr/bin/env python3
"""
Test script to compare hook-generated vs manual memory storage for Issue #99.
This test validates timestamp handling, tag consistency, and discoverability
between different memory creation methods.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import asyncio
import tempfile
import time
from datetime import datetime

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash
from mcp_memory_service.utils.time_parser import extract_time_expression


class HookVsManualStorageTest:
    """Test suite comparing hook-generated and manual memory storage."""

    def __init__(self, storage_backend: str = "sqlite_vec"):
        self.storage_backend = storage_backend
        self.storage = None
        self.test_memories_created = []

    async def setup(self):
        """Set up test environment and storage."""
        print(f"=== Setting up {self.storage_backend} storage for testing ===")

        if self.storage_backend == "sqlite_vec":
            # Create temporary database for testing
            self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            self.temp_db.close()

            self.storage = SqliteVecMemoryStorage(db_path=self.temp_db.name, embedding_model="all-MiniLM-L6-v2")
            await self.storage.initialize()
            print(f"✅ SQLite-Vec storage initialized: {self.temp_db.name}")

    async def cleanup(self):
        """Clean up test environment."""
        # Note: SqliteVecMemoryStorage doesn't have a close() method
        self.storage = None

        if hasattr(self, "temp_db") and os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
            print("✅ Test database cleaned up")

    def create_hook_style_memory(self, content: str, project_context: dict) -> Memory:
        """Create a memory as hooks would create it (with auto-generated tags)."""

        # Simulate hook behavior - generate tags like session-end.js does
        hook_tags = [
            "claude-code-session",
            "session-consolidation",
            project_context.get("name", "unknown-project"),
            f"language:{project_context.get('language', 'unknown')}",
            *project_context.get("frameworks", [])[:2],  # Top 2 frameworks
        ]

        # Filter out None/empty tags
        hook_tags = [tag for tag in hook_tags if tag]

        memory = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=hook_tags,
            memory_type="session-summary",
            metadata={
                "session_analysis": {"topics": ["test-topic"], "decisions_count": 1, "insights_count": 1, "confidence": 0.85},
                "project_context": project_context,
                "generated_by": "claude-code-session-end-hook",
                "generated_at": datetime.now().isoformat(),
            },
        )

        return memory

    def create_manual_memory(self, content: str, user_tags: list[str] = None) -> Memory:
        """Create a memory as manual /memory-store would create it."""

        # Manual memories typically have user-provided tags, not auto-generated ones
        manual_tags = user_tags or []

        memory = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=manual_tags,
            memory_type="note",
            metadata={"created_by": "manual-storage", "source": "user-input"},
        )

        return memory

    async def test_timestamp_consistency(self):
        """Test 1: Compare timestamp handling between hook and manual memories."""
        print("\n🧪 Test 1: Timestamp Consistency")
        print("-" * 50)

        # Create memories with slight time differences
        time.time()

        project_context = {"name": "mcp-memory-service", "language": "python", "frameworks": ["fastapi", "chromadb"]}

        # Hook-style memory
        hook_memory = self.create_hook_style_memory("Implemented timestamp standardization for Issue #99", project_context)

        # Manual memory created shortly after
        time.sleep(0.1)  # Small delay to test precision
        manual_memory = self.create_manual_memory(
            "Fixed timestamp precision issue in memory storage", ["timestamp-fix", "issue-99", "debugging"]
        )

        # Store both memories
        await self.storage.store(hook_memory)
        await self.storage.store(manual_memory)

        self.test_memories_created.extend([hook_memory.content_hash, manual_memory.content_hash])

        print("Hook memory timestamps:")
        print(f"  created_at: {hook_memory.created_at}")
        print(f"  created_at_iso: {hook_memory.created_at_iso}")
        print(f"  Type check: {type(hook_memory.created_at)} / {type(hook_memory.created_at_iso)}")

        print("\nManual memory timestamps:")
        print(f"  created_at: {manual_memory.created_at}")
        print(f"  created_at_iso: {manual_memory.created_at_iso}")
        print(f"  Type check: {type(manual_memory.created_at)} / {type(manual_memory.created_at_iso)}")

        # Check if both have proper timestamps
        hook_has_timestamps = hook_memory.created_at is not None and hook_memory.created_at_iso is not None
        manual_has_timestamps = manual_memory.created_at is not None and manual_memory.created_at_iso is not None

        print("\nTimestamp validation:")
        print(f"  Hook memory has complete timestamps: {hook_has_timestamps}")
        print(f"  Manual memory has complete timestamps: {manual_has_timestamps}")

        if hook_has_timestamps and manual_has_timestamps:
            print("✅ Both memory types have consistent timestamp formats")
        else:
            print("❌ Timestamp inconsistency detected!")

        return {
            "hook_has_timestamps": hook_has_timestamps,
            "manual_has_timestamps": manual_has_timestamps,
            "hook_memory": hook_memory,
            "manual_memory": manual_memory,
        }

    async def test_tag_consistency(self):
        """Test 2: Compare tag patterns between hook and manual memories."""
        print("\n🧪 Test 2: Tag Consistency Analysis")
        print("-" * 50)

        project_context = {"name": "test-project", "language": "typescript", "frameworks": ["react", "node"]}

        # Create hook memory
        hook_memory = self.create_hook_style_memory("Testing tag consistency between storage methods", project_context)

        # Create manual memory with content-appropriate tags
        manual_memory = self.create_manual_memory(
            "Testing tag consistency between storage methods", ["testing", "tag-consistency", "storage-methods", "validation"]
        )

        await self.storage.store(hook_memory)
        await self.storage.store(manual_memory)

        self.test_memories_created.extend([hook_memory.content_hash, manual_memory.content_hash])

        print(f"Hook memory tags: {hook_memory.tags}")
        print(f"Manual memory tags: {manual_memory.tags}")

        # Analyze tag patterns
        hook_has_auto_tags = any("claude-code" in tag for tag in hook_memory.tags)
        manual_has_content_tags = len(manual_memory.tags) > 0 and not any("auto-generated" in tag for tag in manual_memory.tags)

        print("\nTag analysis:")
        print(f"  Hook memory has auto-generated tags: {hook_has_auto_tags}")
        print(f"  Manual memory has content-relevant tags: {manual_has_content_tags}")
        print(f"  Hook tag count: {len(hook_memory.tags)}")
        print(f"  Manual tag count: {len(manual_memory.tags)}")

        if hook_has_auto_tags and manual_has_content_tags:
            print("✅ Tag patterns are appropriately different and content-relevant")
        else:
            print("❌ Tag pattern issues detected")

        return {
            "hook_tags": hook_memory.tags,
            "manual_tags": manual_memory.tags,
            "hook_has_auto_tags": hook_has_auto_tags,
            "manual_has_content_tags": manual_has_content_tags,
        }

    async def test_time_based_search_consistency(self):
        """Test 3: Verify both memory types are discoverable in time-based searches."""
        print("\n🧪 Test 3: Time-Based Search Discoverability")
        print("-" * 50)

        # Create memories with known timestamps
        current_time = time.time()
        yesterday_time = current_time - (24 * 60 * 60)  # 24 hours ago

        # Create hook memory with specific timestamp
        hook_memory = self.create_hook_style_memory(
            "Hook memory created yesterday for search testing", {"name": "search-test", "language": "python", "frameworks": []}
        )
        hook_memory.created_at = yesterday_time
        hook_memory.created_at_iso = datetime.fromtimestamp(yesterday_time).isoformat() + "Z"

        # Create manual memory with specific timestamp
        manual_memory = self.create_manual_memory(
            "Manual memory created yesterday for search testing", ["search-test", "yesterday", "discoverability"]
        )
        manual_memory.created_at = yesterday_time + 100  # Slightly later
        manual_memory.created_at_iso = datetime.fromtimestamp(yesterday_time + 100).isoformat() + "Z"

        await self.storage.store(hook_memory)
        await self.storage.store(manual_memory)

        self.test_memories_created.extend([hook_memory.content_hash, manual_memory.content_hash])

        # Test time-based recall
        query = "yesterday"
        cleaned_query, (start_ts, end_ts) = extract_time_expression(query)

        if start_ts and end_ts:
            print(f"Search range: {datetime.fromtimestamp(start_ts)} to {datetime.fromtimestamp(end_ts)}")
            print(f"Hook memory timestamp: {datetime.fromtimestamp(hook_memory.created_at)}")
            print(f"Manual memory timestamp: {datetime.fromtimestamp(manual_memory.created_at)}")

            # Check if memories fall within range
            hook_in_range = start_ts <= hook_memory.created_at <= end_ts
            manual_in_range = start_ts <= manual_memory.created_at <= end_ts

            print("\nTime range analysis:")
            print(f"  Hook memory in range: {hook_in_range}")
            print(f"  Manual memory in range: {manual_in_range}")

            if hook_in_range and manual_in_range:
                print("✅ Both memory types would be discoverable in time-based searches")
                return {"discoverability_consistent": True}
            else:
                print("❌ Time-based search discoverability inconsistent")
                return {"discoverability_consistent": False}
        else:
            print("⚠️  Could not parse time expression for testing")
            return {"discoverability_consistent": None}

    async def test_metadata_structure_comparison(self):
        """Test 4: Compare metadata structure between hook and manual memories."""
        print("\n🧪 Test 4: Metadata Structure Comparison")
        print("-" * 50)

        # Create memories with different metadata patterns
        hook_memory = self.create_hook_style_memory(
            "Testing metadata structure consistency",
            {"name": "metadata-test", "language": "javascript", "frameworks": ["express"]},
        )

        manual_memory = self.create_manual_memory(
            "Testing metadata structure consistency", ["metadata-test", "structure-analysis"]
        )

        await self.storage.store(hook_memory)
        await self.storage.store(manual_memory)

        self.test_memories_created.extend([hook_memory.content_hash, manual_memory.content_hash])

        # Analyze metadata structures
        hook_metadata_keys = set(hook_memory.metadata.keys()) if hook_memory.metadata else set()
        manual_metadata_keys = set(manual_memory.metadata.keys()) if manual_memory.metadata else set()

        print(f"Hook memory metadata keys: {sorted(hook_metadata_keys)}")
        print(f"Manual memory metadata keys: {sorted(manual_metadata_keys)}")

        # Check for required fields
        hook_has_timestamps = hasattr(hook_memory, "created_at_iso") and hook_memory.created_at_iso is not None
        manual_has_timestamps = hasattr(manual_memory, "created_at_iso") and manual_memory.created_at_iso is not None

        print("\nMetadata analysis:")
        print(f"  Hook memory has ISO timestamp: {hook_has_timestamps}")
        print(f"  Manual memory has ISO timestamp: {manual_has_timestamps}")
        print(f"  Hook metadata structure: {bool(hook_memory.metadata)}")
        print(f"  Manual metadata structure: {bool(manual_memory.metadata)}")

        return {
            "hook_metadata_keys": hook_metadata_keys,
            "manual_metadata_keys": manual_metadata_keys,
            "metadata_consistency": hook_has_timestamps and manual_has_timestamps,
        }

    async def run_all_tests(self):
        """Run all tests and compile results."""
        print("=" * 70)
        print("MCP Memory Service: Hook vs Manual Storage Consistency Tests")
        print("Testing for Issue #99 - Memory Storage Inconsistency")
        print("=" * 70)

        try:
            await self.setup()

            # Run individual tests
            timestamp_results = await self.test_timestamp_consistency()
            tag_results = await self.test_tag_consistency()
            search_results = await self.test_time_based_search_consistency()
            metadata_results = await self.test_metadata_structure_comparison()

            # Compile overall results
            print("\n" + "=" * 70)
            print("TEST SUMMARY")
            print("=" * 70)

            tests_passed = 0
            total_tests = 4

            if timestamp_results.get("hook_has_timestamps") and timestamp_results.get("manual_has_timestamps"):
                print("✅ PASS: Timestamp Consistency")
                tests_passed += 1
            else:
                print("❌ FAIL: Timestamp Consistency")

            if tag_results.get("hook_has_auto_tags") and tag_results.get("manual_has_content_tags"):
                print("✅ PASS: Tag Pattern Appropriateness")
                tests_passed += 1
            else:
                print("❌ FAIL: Tag Pattern Issues")

            if search_results.get("discoverability_consistent"):
                print("✅ PASS: Time-Based Search Discoverability")
                tests_passed += 1
            elif search_results.get("discoverability_consistent") is False:
                print("❌ FAIL: Time-Based Search Discoverability")
            else:
                print("⚠️  SKIP: Time-Based Search Test (parsing issue)")

            if metadata_results.get("metadata_consistency"):
                print("✅ PASS: Metadata Structure Consistency")
                tests_passed += 1
            else:
                print("❌ FAIL: Metadata Structure Consistency")

            print(f"\nOverall Result: {tests_passed}/{total_tests} tests passed")

            if tests_passed == total_tests:
                print("🎉 All tests passed! No storage inconsistency detected.")
                return True
            else:
                print("⚠️  Storage inconsistencies detected - Issue #99 confirmed.")
                return False

        finally:
            await self.cleanup()


async def main():
    """Main test execution."""
    test_suite = HookVsManualStorageTest("sqlite_vec")
    success = await test_suite.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
