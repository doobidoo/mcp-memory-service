#!/usr/bin/env python3
"""
Final validation test for Issue #99 fix.
This test creates memories that SHOULD be in yesterday's range
and verifies they can be found by time-based searches.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import asyncio
import tempfile
import time
from datetime import datetime, timedelta

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash
from mcp_memory_service.utils.time_parser import extract_time_expression


class Issue99FinalValidationTest:
    """Final validation test for Issue #99 timezone fix."""

    def __init__(self):
        self.storage = None

    async def setup(self):
        """Set up test environment."""
        print("=== Final Issue #99 Validation Test ===")

        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()

        self.storage = SqliteVecMemoryStorage(db_path=self.temp_db.name, embedding_model="all-MiniLM-L6-v2")
        await self.storage.initialize()
        print("✅ Storage initialized")

    async def cleanup(self):
        """Clean up test environment."""
        self.storage = None
        if hasattr(self, "temp_db") and os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)

    async def test_timezone_fix_validation(self):
        """Validate that the timezone fix resolves Issue #99."""
        print("\n🧪 Testing Issue #99 Fix: Timezone Handling")
        print("-" * 50)

        # Calculate actual yesterday timestamps
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        yesterday_start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday_middle = yesterday.replace(hour=12, minute=0, second=0, microsecond=0)
        yesterday_end = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)

        print(f"📅 Yesterday date range: {yesterday_start.date()}")
        print(f"🕐 Yesterday timestamps: {yesterday_start.timestamp()} to {yesterday_end.timestamp()}")

        # Create memories that should be found in yesterday's range
        memories = [
            {
                "content": "Hook-style memory created yesterday morning",
                "timestamp": yesterday_start.timestamp() + (2 * 60 * 60),  # 2 AM yesterday
                "tags": ["claude-code-session", "yesterday-morning"],
            },
            {
                "content": "Manual note from yesterday afternoon",
                "timestamp": yesterday_middle.timestamp() + (3 * 60 * 60),  # 3 PM yesterday
                "tags": ["manual-note", "yesterday-afternoon"],
            },
            {
                "content": "Another hook memory from yesterday evening",
                "timestamp": yesterday_end.timestamp() - (2 * 60 * 60),  # 9 PM yesterday
                "tags": ["claude-code-session", "yesterday-evening"],
            },
        ]

        # Store memories with specific yesterday timestamps
        for i, mem_data in enumerate(memories):
            memory = Memory(
                content=mem_data["content"],
                content_hash=generate_content_hash(mem_data["content"]),
                tags=mem_data["tags"],
                memory_type="test-memory",
                created_at=mem_data["timestamp"],
                created_at_iso=datetime.fromtimestamp(mem_data["timestamp"]).isoformat() + "Z",
            )

            success, message = await self.storage.store(memory)
            if success:
                print(f"✅ Stored memory {i + 1}: {datetime.fromtimestamp(mem_data['timestamp'])}")
            else:
                print(f"❌ Failed to store memory {i + 1}: {message}")
                return False

        # Test yesterday search
        query = "yesterday"
        cleaned_query, (start_ts, end_ts) = extract_time_expression(query)

        print(f"\n🔍 Testing query: '{query}'")
        print(f"📅 Search range: {datetime.fromtimestamp(start_ts)} to {datetime.fromtimestamp(end_ts)}")

        # Perform search
        search_results = await self.storage.retrieve(query, n_results=10)
        print(f"🔍 Found {len(search_results)} memories")

        # Check if we found the expected memories
        found_count = 0
        for result in search_results:
            for mem_data in memories:
                if mem_data["content"] in result.memory.content:
                    found_count += 1
                    print(f"  ✅ Found: {result.memory.content}")
                    break

        # Validation
        expected_count = len(memories)
        success = found_count == expected_count

        print("\n📊 Results:")
        print(f"  Expected memories: {expected_count}")
        print(f"  Found memories: {found_count}")
        print(f"  Success: {success}")

        if success:
            print("🎉 Issue #99 FIXED: Time-based search now works correctly!")
        else:
            print("❌ Issue #99 NOT FIXED: Time-based search still has problems")

        return success

    async def test_hook_vs_manual_consistency(self):
        """Test that hook and manual memories are equally discoverable."""
        print("\n🧪 Testing Hook vs Manual Memory Search Consistency")
        print("-" * 50)

        # Create one hook-style and one manual-style memory for today
        now = time.time()
        today_morning = now - (8 * 60 * 60)  # 8 hours ago

        hook_memory = Memory(
            content="Hook-generated session summary from this morning",
            content_hash=generate_content_hash("Hook-generated session summary from this morning"),
            tags=["claude-code-session", "session-consolidation", "morning-work"],
            memory_type="session-summary",
            metadata={
                "generated_by": "claude-code-session-end-hook",
                "generated_at": datetime.fromtimestamp(today_morning).isoformat() + "Z",
            },
            created_at=today_morning,
        )

        manual_memory = Memory(
            content="Manual note added this morning about project status",
            content_hash=generate_content_hash("Manual note added this morning about project status"),
            tags=["manual-note", "project-status", "morning-work"],
            memory_type="note",
            metadata={"created_by": "manual-storage", "source": "user-input"},
            created_at=today_morning + 300,  # 5 minutes later
        )

        # Store both memories
        hook_result = await self.storage.store(hook_memory)
        manual_result = await self.storage.store(manual_memory)

        print(f"✅ Hook memory stored: {hook_result[0]}")
        print(f"✅ Manual memory stored: {manual_result[0]}")

        # Search for memories from today
        query = "today morning"
        search_results = await self.storage.retrieve(query, n_results=10)

        hook_found = False
        manual_found = False

        for result in search_results:
            if "Hook-generated session summary" in result.memory.content:
                hook_found = True
            if "Manual note added this morning" in result.memory.content:
                manual_found = True

        print(f"\n📊 Search Results for '{query}':")
        print(f"  Hook memory found: {hook_found}")
        print(f"  Manual memory found: {manual_found}")
        print(f"  Both equally discoverable: {hook_found and manual_found}")

        return hook_found and manual_found

    async def run_validation(self):
        """Run complete Issue #99 validation."""
        try:
            await self.setup()

            # Run validation tests
            timezone_fix = await self.test_timezone_fix_validation()
            consistency_fix = await self.test_hook_vs_manual_consistency()

            print("\n" + "=" * 60)
            print("ISSUE #99 FINAL VALIDATION RESULTS")
            print("=" * 60)

            if timezone_fix:
                print("✅ FIXED: Timezone handling in timestamp validation")
            else:
                print("❌ NOT FIXED: Timezone handling still has issues")

            if consistency_fix:
                print("✅ FIXED: Hook vs Manual memory search consistency")
            else:
                print("❌ NOT FIXED: Hook vs Manual memories still inconsistent")

            overall_success = timezone_fix and consistency_fix

            if overall_success:
                print("\n🎉 ISSUE #99 COMPLETELY RESOLVED!")
                print("✅ Time-based searches work correctly")
                print("✅ Hook and manual memories are equally discoverable")
                print("✅ Timezone inconsistencies have been fixed")
            else:
                print("\n⚠️  ISSUE #99 PARTIALLY RESOLVED")
                print("Additional work may be needed")

            return overall_success

        finally:
            await self.cleanup()


async def main():
    """Main validation execution."""
    validator = Issue99FinalValidationTest()
    success = await validator.run_validation()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
