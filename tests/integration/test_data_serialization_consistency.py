#!/usr/bin/env python3
"""
Test script to examine data serialization differences that could cause Issue #99.
This focuses on how Memory objects are serialized/deserialized in different contexts
without requiring the full MCP server stack.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import asyncio
import json
import tempfile
import time
from datetime import datetime

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


class DataSerializationTest:
    """Test suite examining memory data serialization consistency."""

    def __init__(self):
        self.storage = None
        self.test_memories = []

    async def setup(self):
        """Set up test environment."""
        print("=== Setting up data serialization test environment ===")

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

    def create_hook_style_memory_with_metadata(self) -> Memory:
        """Create a memory simulating hook-generated content with rich metadata."""
        content = "Implemented comprehensive testing for Issue #99 memory storage inconsistency"

        memory = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["claude-code-session", "session-consolidation", "issue-99", "mcp-memory-service", "language:python"],
            memory_type="session-summary",
            metadata={
                "session_analysis": {
                    "topics": ["testing", "storage", "consistency"],
                    "decisions_count": 3,
                    "insights_count": 5,
                    "confidence": 0.92,
                },
                "project_context": {"name": "mcp-memory-service", "language": "python", "frameworks": ["fastapi", "chromadb"]},
                "generated_by": "claude-code-session-end-hook",
                "generated_at": "2025-09-14T04:42:15.123Z",
            },
        )

        print(f"🔧 Created hook-style memory with {len(memory.tags)} tags and rich metadata")
        return memory

    def create_manual_memory_with_minimal_metadata(self) -> Memory:
        """Create a memory simulating manual /memory-store with minimal metadata."""
        content = "Manual note about Issue #99 analysis findings"

        memory = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["issue-99", "analysis", "findings", "manual-note"],
            memory_type="note",
            metadata={"source": "user-input", "created_by": "manual-storage"},
        )

        print(f"📝 Created manual-style memory with {len(memory.tags)} tags and minimal metadata")
        return memory

    async def test_memory_serialization_roundtrip(self):
        """Test 1: Examine serialization/deserialization consistency."""
        print("\n🧪 Test 1: Memory Serialization Roundtrip Analysis")
        print("-" * 60)

        # Create test memories
        hook_memory = self.create_hook_style_memory_with_metadata()
        manual_memory = self.create_manual_memory_with_minimal_metadata()

        # Test serialization to dict and back
        hook_dict = hook_memory.to_dict()
        manual_dict = manual_memory.to_dict()

        print(f"📊 Hook memory dict keys: {sorted(hook_dict.keys())}")
        print(f"📊 Manual memory dict keys: {sorted(manual_dict.keys())}")

        # Test deserialization from dict
        hook_restored = Memory.from_dict(hook_dict)
        manual_restored = Memory.from_dict(manual_dict)

        # Compare original vs restored
        hook_consistency = {
            "content_match": hook_memory.content == hook_restored.content,
            "tags_match": hook_memory.tags == hook_restored.tags,
            "metadata_match": hook_memory.metadata == hook_restored.metadata,
            "created_at_preserved": abs((hook_memory.created_at or 0) - (hook_restored.created_at or 0)) < 0.001,
            "created_at_iso_preserved": hook_memory.created_at_iso == hook_restored.created_at_iso,
        }

        manual_consistency = {
            "content_match": manual_memory.content == manual_restored.content,
            "tags_match": manual_memory.tags == manual_restored.tags,
            "metadata_match": manual_memory.metadata == manual_restored.metadata,
            "created_at_preserved": abs((manual_memory.created_at or 0) - (manual_restored.created_at or 0)) < 0.001,
            "created_at_iso_preserved": manual_memory.created_at_iso == manual_restored.created_at_iso,
        }

        print("\n📋 Hook memory serialization consistency:")
        for key, value in hook_consistency.items():
            status = "✅" if value else "❌"
            print(f"  {status} {key}: {value}")

        print("\n📋 Manual memory serialization consistency:")
        for key, value in manual_consistency.items():
            status = "✅" if value else "❌"
            print(f"  {status} {key}: {value}")

        return {
            "hook_consistency": hook_consistency,
            "manual_consistency": manual_consistency,
            "hook_dict": hook_dict,
            "manual_dict": manual_dict,
        }

    async def test_storage_backend_handling(self):
        """Test 2: Examine how storage backend handles different memory types."""
        print("\n🧪 Test 2: Storage Backend Handling Analysis")
        print("-" * 60)

        # Create and store different memory types
        hook_memory = self.create_hook_style_memory_with_metadata()
        manual_memory = self.create_manual_memory_with_minimal_metadata()

        # Store both memories
        hook_store_result = await self.storage.store(hook_memory)
        manual_store_result = await self.storage.store(manual_memory)

        print(f"📤 Hook memory storage result: {hook_store_result}")
        print(f"📤 Manual memory storage result: {manual_store_result}")

        # Retrieve memories back
        hook_retrieved = await self.storage.retrieve(hook_memory.content, n_results=1)
        manual_retrieved = await self.storage.retrieve(manual_memory.content, n_results=1)

        storage_analysis = {
            "hook_stored_successfully": hook_store_result[0],
            "manual_stored_successfully": manual_store_result[0],
            "hook_retrieved_count": len(hook_retrieved),
            "manual_retrieved_count": len(manual_retrieved),
        }

        if hook_retrieved:
            retrieved_hook = hook_retrieved[0].memory
            storage_analysis["hook_retrieval_analysis"] = {
                "content_preserved": retrieved_hook.content == hook_memory.content,
                "tags_preserved": retrieved_hook.tags == hook_memory.tags,
                "timestamp_preserved": (retrieved_hook.created_at is not None and retrieved_hook.created_at_iso is not None),
                "metadata_preserved": bool(retrieved_hook.metadata),
            }

            print("\n📥 Retrieved hook memory analysis:")
            for key, value in storage_analysis["hook_retrieval_analysis"].items():
                status = "✅" if value else "❌"
                print(f"  {status} {key}: {value}")

        if manual_retrieved:
            retrieved_manual = manual_retrieved[0].memory
            storage_analysis["manual_retrieval_analysis"] = {
                "content_preserved": retrieved_manual.content == manual_memory.content,
                "tags_preserved": retrieved_manual.tags == manual_memory.tags,
                "timestamp_preserved": (retrieved_manual.created_at is not None and retrieved_manual.created_at_iso is not None),
                "metadata_preserved": bool(retrieved_manual.metadata),
            }

            print("\n📥 Retrieved manual memory analysis:")
            for key, value in storage_analysis["manual_retrieval_analysis"].items():
                status = "✅" if value else "❌"
                print(f"  {status} {key}: {value}")

        return storage_analysis

    async def test_timestamp_precision_handling(self):
        """Test 3: Examine timestamp precision across serialization boundaries."""
        print("\n🧪 Test 3: Timestamp Precision Analysis")
        print("-" * 60)

        # Create memory with very specific timestamp
        precise_timestamp = time.time()
        precise_iso = datetime.fromtimestamp(precise_timestamp).isoformat() + "Z"

        memory = Memory(
            content="Testing timestamp precision across storage boundaries",
            content_hash=generate_content_hash("Testing timestamp precision"),
            tags=["timestamp-test", "precision"],
            memory_type="test",
            created_at=precise_timestamp,
            created_at_iso=precise_iso,
        )

        print(f"🕐 Original timestamp (float): {precise_timestamp}")
        print(f"🕐 Original timestamp (ISO): {precise_iso}")

        # Test serialization
        memory_dict = memory.to_dict()
        print(
            f"🔄 Serialized timestamp fields: {json.dumps({k: v for k, v in memory_dict.items() if 'timestamp' in k or 'created_at' in k}, indent=2)}"
        )

        # Test deserialization
        restored_memory = Memory.from_dict(memory_dict)
        print(f"🔄 Restored timestamp (float): {restored_memory.created_at}")
        print(f"🔄 Restored timestamp (ISO): {restored_memory.created_at_iso}")

        # Test storage roundtrip
        store_result = await self.storage.store(memory)
        retrieved_results = await self.storage.retrieve(memory.content, n_results=1)

        precision_analysis = {
            "serialization_preserves_precision": abs(precise_timestamp - (restored_memory.created_at or 0)) < 0.001,
            "iso_format_preserved": precise_iso == restored_memory.created_at_iso,
            "storage_successful": store_result[0],
            "retrieval_successful": len(retrieved_results) > 0,
        }

        if retrieved_results:
            stored_memory = retrieved_results[0].memory
            precision_analysis.update(
                {
                    "storage_preserves_float_precision": abs(precise_timestamp - (stored_memory.created_at or 0)) < 0.001,
                    "storage_preserves_iso_format": precise_iso == stored_memory.created_at_iso,
                }
            )

            print(f"💾 Storage preserved timestamp (float): {stored_memory.created_at}")
            print(f"💾 Storage preserved timestamp (ISO): {stored_memory.created_at_iso}")

        print("\n📊 Precision analysis:")
        for key, value in precision_analysis.items():
            status = "✅" if value else "❌"
            print(f"  {status} {key}: {value}")

        return precision_analysis

    async def run_all_tests(self):
        """Run all data serialization tests."""
        print("=" * 70)
        print("MCP Memory Service: Data Serialization Consistency Analysis")
        print("Investigating Issue #99 - Root Cause Analysis")
        print("=" * 70)

        try:
            await self.setup()

            # Run individual tests
            serialization_test = await self.test_memory_serialization_roundtrip()
            storage_test = await self.test_storage_backend_handling()
            precision_test = await self.test_timestamp_precision_handling()

            # Analyze overall results
            print("\n" + "=" * 70)
            print("COMPREHENSIVE ANALYSIS SUMMARY")
            print("=" * 70)

            tests_passed = 0
            total_tests = 3

            # Serialization consistency
            hook_serialization_ok = all(serialization_test["hook_consistency"].values())
            manual_serialization_ok = all(serialization_test["manual_consistency"].values())

            if hook_serialization_ok and manual_serialization_ok:
                print("✅ PASS: Memory serialization/deserialization consistent")
                tests_passed += 1
            else:
                print("❌ FAIL: Serialization inconsistencies detected")

            # Storage backend handling
            storage_ok = (
                storage_test.get("hook_stored_successfully", False)
                and storage_test.get("manual_stored_successfully", False)
                and storage_test.get("hook_retrieved_count", 0) > 0
                and storage_test.get("manual_retrieved_count", 0) > 0
            )

            if storage_ok:
                print("✅ PASS: Storage backend handles both memory types consistently")
                tests_passed += 1
            else:
                print("❌ FAIL: Storage backend handling inconsistencies")

            # Timestamp precision
            precision_ok = precision_test.get("serialization_preserves_precision", False) and precision_test.get(
                "storage_preserves_float_precision", False
            )

            if precision_ok:
                print("✅ PASS: Timestamp precision maintained across boundaries")
                tests_passed += 1
            else:
                print("❌ FAIL: Timestamp precision issues detected")

            print(f"\nOverall Result: {tests_passed}/{total_tests} tests passed")

            # Root cause analysis
            print("\n🔍 ROOT CAUSE ANALYSIS:")

            if tests_passed == total_tests:
                print("• ✅ Memory objects themselves are consistent")
                print("• ✅ Storage backend handles both types properly")
                print("• ✅ Timestamp precision is maintained")
                print("\n🎯 CONCLUSION: Issue #99 is likely in:")
                print("  - Search/retrieval query logic")
                print("  - Time-based filtering implementation")
                print("  - Tag-based search differences")
                print("  - Client-side display logic")
                print("\n💡 RECOMMENDATION: Focus investigation on search and retrieval functions")
            else:
                print("• ❌ Detected inconsistencies in core data handling")
                print("• This confirms the storage-level issues described in Discussion #98")

            return tests_passed == total_tests

        finally:
            await self.cleanup()


async def main():
    """Main test execution."""
    test_suite = DataSerializationTest()
    success = await test_suite.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
