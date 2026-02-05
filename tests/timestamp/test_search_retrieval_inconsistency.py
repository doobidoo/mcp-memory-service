#!/usr/bin/env python3
"""
Test script to identify the exact root cause of Issue #99 search inconsistency.
Based on investigation, the issue appears to be in timestamp field mapping
between storage and retrieval in ChromaDB where different timestamp fields
are used for querying vs storing.
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


class SearchRetrievalInconsistencyTest:
    """Test suite to identify search/retrieval timestamp inconsistencies."""

    def __init__(self):
        self.storage = None
        self.test_memories = []

    async def setup(self):
        """Set up test environment."""
        print("=== Setting up search/retrieval inconsistency test ===")

        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()

        self.storage = SqliteVecMemoryStorage(db_path=self.temp_db.name, embedding_model="all-MiniLM-L6-v2")
        await self.storage.initialize()
        print(f"✅ Storage initialized: {self.temp_db.name}")

    async def cleanup(self):
        """Clean up test environment."""
        self.storage = None
        if hasattr(self, "temp_db") and os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
            print("✅ Test database cleaned up")

    async def create_test_memories_with_specific_timestamps(self):
        """Create test memories with carefully controlled timestamps."""
        print("\n🧪 Creating test memories with specific timestamps")
        print("-" * 60)

        # Calculate specific timestamps for testing
        now = time.time()
        yesterday_start = now - (24 * 60 * 60)  # 24 hours ago
        yesterday_middle = yesterday_start + (12 * 60 * 60)  # 12 hours into yesterday
        yesterday_end = yesterday_start + (23.5 * 60 * 60)  # Near end of yesterday

        test_cases = [
            {
                "name": "hook_style_memory_yesterday",
                "content": "Hook-generated memory from yesterday's development session",
                "timestamp": yesterday_middle,
                "tags": ["claude-code-session", "session-consolidation", "yesterday-work"],
                "metadata": {
                    "generated_by": "claude-code-session-end-hook",
                    "generated_at": datetime.fromtimestamp(yesterday_middle).isoformat() + "Z",
                    "session_analysis": {"topics": ["development", "testing"]},
                },
                "memory_type": "session-summary",
            },
            {
                "name": "manual_memory_yesterday",
                "content": "Manual note stored yesterday about project progress",
                "timestamp": yesterday_end,
                "tags": ["manual-note", "project-progress", "yesterday"],
                "metadata": {"created_by": "manual-storage", "source": "user-input"},
                "memory_type": "note",
            },
            {
                "name": "hook_style_memory_today",
                "content": "Hook-generated memory from today's session",
                "timestamp": now - (2 * 60 * 60),  # 2 hours ago
                "tags": ["claude-code-session", "session-consolidation", "today-work"],
                "metadata": {
                    "generated_by": "claude-code-session-end-hook",
                    "generated_at": datetime.fromtimestamp(now - (2 * 60 * 60)).isoformat() + "Z",
                },
                "memory_type": "session-summary",
            },
            {
                "name": "manual_memory_today",
                "content": "Manual note stored today about urgent task",
                "timestamp": now - (1 * 60 * 60),  # 1 hour ago
                "tags": ["manual-note", "urgent-task", "today"],
                "metadata": {"created_by": "manual-storage", "source": "user-input"},
                "memory_type": "note",
            },
        ]

        stored_memories = []

        for case in test_cases:
            # Create memory with specific timestamp
            memory = Memory(
                content=case["content"],
                content_hash=generate_content_hash(case["content"]),
                tags=case["tags"],
                memory_type=case["memory_type"],
                metadata=case["metadata"],
                created_at=case["timestamp"],
                created_at_iso=datetime.fromtimestamp(case["timestamp"]).isoformat() + "Z",
            )

            # Store the memory
            success, message = await self.storage.store(memory)
            if success:
                stored_memories.append({"name": case["name"], "memory": memory, "expected_timestamp": case["timestamp"]})
                print(f"✅ Stored {case['name']}: {datetime.fromtimestamp(case['timestamp'])}")
            else:
                print(f"❌ Failed to store {case['name']}: {message}")

        self.test_memories = stored_memories
        return stored_memories

    async def test_time_based_search_consistency(self):
        """Test if time-based searches find all expected memories."""
        print("\n🧪 Test 1: Time-Based Search Consistency")
        print("-" * 60)

        # Test yesterday search
        query = "yesterday"
        cleaned_query, (start_ts, end_ts) = extract_time_expression(query)

        print(f"🔍 Testing query: '{query}'")
        print(f"📅 Search range: {datetime.fromtimestamp(start_ts)} to {datetime.fromtimestamp(end_ts)}")

        # Check which memories should be found
        expected_memories = []
        for mem_info in self.test_memories:
            if start_ts <= mem_info["expected_timestamp"] <= end_ts:
                expected_memories.append(mem_info["name"])

        print(f"📋 Expected to find memories: {expected_memories}")

        # Perform the search using retrieve (general search)
        search_results = await self.storage.retrieve(query, n_results=10)
        print(f"🔍 General retrieve found: {len(search_results)} memories")

        for result in search_results:
            print(f"  - {result.memory.content[:50]}...")

        # Check if we found the expected memories
        found_memories = []
        for result in search_results:
            for mem_info in self.test_memories:
                if result.memory.content == mem_info["memory"].content:
                    found_memories.append(mem_info["name"])
                    break

        print(f"📋 Actually found memories: {found_memories}")

        # Analysis
        missing_memories = set(expected_memories) - set(found_memories)
        unexpected_memories = set(found_memories) - set(expected_memories)

        search_analysis = {
            "expected_count": len(expected_memories),
            "found_count": len(found_memories),
            "missing_memories": list(missing_memories),
            "unexpected_memories": list(unexpected_memories),
            "search_consistent": len(missing_memories) == 0 and len(unexpected_memories) == 0,
        }

        if search_analysis["search_consistent"]:
            print("✅ Time-based search is consistent")
        else:
            print("❌ Time-based search inconsistency detected!")
            if missing_memories:
                print(f"   Missing: {missing_memories}")
            if unexpected_memories:
                print(f"   Unexpected: {unexpected_memories}")

        return search_analysis

    async def test_direct_timestamp_queries(self):
        """Test direct timestamp-based queries to isolate the issue."""
        print("\n🧪 Test 2: Direct Timestamp Query Analysis")
        print("-" * 60)

        # Get yesterday's timestamp range
        yesterday_query = "yesterday"
        cleaned_query, (start_ts, end_ts) = extract_time_expression(yesterday_query)

        print(f"🕐 Yesterday range: {start_ts} to {end_ts}")

        # Check each stored memory's timestamp against the range
        timestamp_analysis = {"memories_in_range": [], "memories_out_of_range": [], "timestamp_precision_issues": []}

        for mem_info in self.test_memories:
            memory = mem_info["memory"]
            expected_ts = mem_info["expected_timestamp"]

            print(f"\n📝 Analyzing {mem_info['name']}:")
            print(f"   Expected timestamp: {expected_ts} ({datetime.fromtimestamp(expected_ts)})")
            print(f"   Memory created_at: {memory.created_at}")
            print(f"   Memory created_at_iso: {memory.created_at_iso}")

            # Check if memory should be in yesterday's range
            in_range = start_ts <= expected_ts <= end_ts
            actually_in_range = start_ts <= (memory.created_at or 0) <= end_ts

            if in_range:
                timestamp_analysis["memories_in_range"].append(mem_info["name"])

            if in_range != actually_in_range:
                timestamp_analysis["timestamp_precision_issues"].append(
                    {
                        "memory": mem_info["name"],
                        "expected_in_range": in_range,
                        "actually_in_range": actually_in_range,
                        "expected_timestamp": expected_ts,
                        "stored_timestamp": memory.created_at,
                    }
                )

            print(f"   Should be in yesterday range: {in_range}")
            print(f"   Memory timestamp in range: {actually_in_range}")

        print("\n📊 Timestamp Analysis Summary:")
        print(f"   Memories in yesterday range: {len(timestamp_analysis['memories_in_range'])}")
        print(f"   Timestamp precision issues: {len(timestamp_analysis['timestamp_precision_issues'])}")

        return timestamp_analysis

    async def test_memory_serialization_fields(self):
        """Test what timestamp fields are actually stored/retrieved."""
        print("\n🧪 Test 3: Memory Serialization Fields Analysis")
        print("-" * 60)

        if not self.test_memories:
            print("⚠️  No test memories available for analysis")
            return {}

        serialization_analysis = {"memory_field_analysis": [], "consistent_fields": True}

        for mem_info in self.test_memories:
            memory = mem_info["memory"]

            # Get the serialized dictionary representation
            memory_dict = memory.to_dict()

            timestamp_fields = {
                "created_at": memory_dict.get("created_at"),
                "created_at_iso": memory_dict.get("created_at_iso"),
                "timestamp": memory_dict.get("timestamp"),
                "timestamp_float": memory_dict.get("timestamp_float"),
                "timestamp_str": memory_dict.get("timestamp_str"),
                "updated_at": memory_dict.get("updated_at"),
                "updated_at_iso": memory_dict.get("updated_at_iso"),
            }

            print(f"\n📝 {mem_info['name']} serialization fields:")
            for field, value in timestamp_fields.items():
                if value is not None:
                    if isinstance(value, float):
                        dt_str = datetime.fromtimestamp(value).isoformat()
                        print(f"   {field}: {value} ({dt_str})")
                    else:
                        print(f"   {field}: {value}")
                else:
                    print(f"   {field}: None")

            analysis_entry = {
                "memory_name": mem_info["name"],
                "timestamp_fields": timestamp_fields,
                "has_all_required": all(
                    [
                        timestamp_fields.get("created_at") is not None,
                        timestamp_fields.get("created_at_iso") is not None,
                        timestamp_fields.get("timestamp") is not None,
                    ]
                ),
            }

            serialization_analysis["memory_field_analysis"].append(analysis_entry)

            if not analysis_entry["has_all_required"]:
                serialization_analysis["consistent_fields"] = False

        return serialization_analysis

    async def run_all_tests(self):
        """Run comprehensive search/retrieval inconsistency analysis."""
        print("=" * 70)
        print("MCP Memory Service: Search/Retrieval Inconsistency Root Cause Analysis")
        print("Issue #99 - Final Investigation Phase")
        print("=" * 70)

        try:
            await self.setup()

            # Create test data
            await self.create_test_memories_with_specific_timestamps()

            # Run tests
            search_test = await self.test_time_based_search_consistency()
            timestamp_test = await self.test_direct_timestamp_queries()
            serialization_test = await self.test_memory_serialization_fields()

            # Final analysis
            print("\n" + "=" * 70)
            print("FINAL ROOT CAUSE ANALYSIS")
            print("=" * 70)

            tests_passed = 0
            total_tests = 3

            # Search consistency
            if search_test.get("search_consistent", False):
                print("✅ PASS: Time-based search is consistent")
                tests_passed += 1
            else:
                print("❌ FAIL: Time-based search inconsistency confirmed")
                print(f"   Missing: {search_test.get('missing_memories', [])}")
                print(f"   Unexpected: {search_test.get('unexpected_memories', [])}")

            # Timestamp precision
            precision_issues = timestamp_test.get("timestamp_precision_issues", [])
            if len(precision_issues) == 0:
                print("✅ PASS: Timestamp precision is correct")
                tests_passed += 1
            else:
                print("❌ FAIL: Timestamp precision issues detected")
                for issue in precision_issues:
                    print(f"   {issue['memory']}: expected={issue['expected_in_range']}, actual={issue['actually_in_range']}")

            # Field serialization
            if serialization_test.get("consistent_fields", False):
                print("✅ PASS: Memory serialization fields consistent")
                tests_passed += 1
            else:
                print("❌ FAIL: Memory serialization field issues")

            print(f"\nOverall Result: {tests_passed}/{total_tests} tests passed")

            # Root cause determination
            print("\n🎯 DEFINITIVE ROOT CAUSE:")

            if tests_passed == total_tests:
                print("• Storage and serialization are working correctly")
                print("• Issue #99 might be in a different storage backend or search implementation")
                print("• The problem could be client-side or in specific edge cases")
            else:
                print("• CONFIRMED: Search/retrieval inconsistencies exist")
                if not search_test.get("search_consistent", False):
                    print("  → Time-based search is not finding expected memories")
                if precision_issues:
                    print("  → Timestamp precision/handling issues in queries")
                if not serialization_test.get("consistent_fields", False):
                    print("  → Memory serialization field inconsistencies")

            print("\n💡 RECOMMENDED FIXES:")
            if not search_test.get("search_consistent", False):
                print("• Review time-based search implementation in storage backends")
                print("• Ensure timestamp field mapping is consistent between store and query")
            if precision_issues:
                print("• Fix timestamp precision handling in search queries")
            if not serialization_test.get("consistent_fields", False):
                print("• Standardize timestamp field serialization across all memories")

            return tests_passed == total_tests

        finally:
            await self.cleanup()


async def main():
    """Main test execution."""
    test_suite = SearchRetrievalInconsistencyTest()
    success = await test_suite.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
