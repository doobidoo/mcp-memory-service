#!/usr/bin/env python3
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
Test script to verify ChromaDB to SQLite-vec migration.

This script compares data between ChromaDB and SQLite-vec to ensure
the migration was successful.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mcp_memory_service.config import CHROMA_PATH, EMBEDDING_MODEL_NAME
from mcp_memory_service.storage.chroma import ChromaMemoryStorage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


async def compare_storage_backends(chroma_path: str, sqlite_path: str):
    """Compare data between ChromaDB and SQLite-vec backends."""

    print("🔍 Testing Migration Results")
    print("=" * 50)

    # Initialize both storages
    chroma_storage = ChromaMemoryStorage(persist_directory=chroma_path, embedding_model_name=EMBEDDING_MODEL_NAME)
    await chroma_storage.initialize()

    sqlite_storage = SqliteVecMemoryStorage(db_path=sqlite_path, embedding_model=EMBEDDING_MODEL_NAME)
    await sqlite_storage.initialize()

    try:
        # Get all memories from both storages
        print("📥 Fetching memories from ChromaDB...")
        chroma_memories = await chroma_storage.retrieve("", n_results=10000)

        print("📥 Fetching memories from SQLite-vec...")
        sqlite_memories = await sqlite_storage.retrieve("", n_results=10000)

        # Compare counts
        chroma_count = len(chroma_memories)
        sqlite_count = len(sqlite_memories)

        print(f"📊 ChromaDB memories: {chroma_count}")
        print(f"📊 SQLite-vec memories: {sqlite_count}")

        if sqlite_count >= chroma_count:
            print("✅ Memory count check: PASSED")
        else:
            print("❌ Memory count check: FAILED")
            print(f"   Missing {chroma_count - sqlite_count} memories")

        # Compare content hashes
        chroma_hashes = {m.memory.content_hash for m in chroma_memories}
        sqlite_hashes = {m.memory.content_hash for m in sqlite_memories}

        missing_in_sqlite = chroma_hashes - sqlite_hashes
        extra_in_sqlite = sqlite_hashes - chroma_hashes

        if not missing_in_sqlite:
            print("✅ Content hash check: PASSED")
        else:
            print("❌ Content hash check: FAILED")
            print(f"   {len(missing_in_sqlite)} memories missing in SQLite-vec")
            if len(missing_in_sqlite) <= 5:
                for hash_val in list(missing_in_sqlite)[:5]:
                    print(f"   - {hash_val[:12]}...")

        if extra_in_sqlite:
            print(f"ℹ️  SQLite-vec has {len(extra_in_sqlite)} additional memories")

        # Test search functionality
        print("\\n🔍 Testing search functionality...")

        if chroma_memories:
            # Use first memory's content as search query
            test_query = chroma_memories[0].memory.content[:50]

            chroma_results = await chroma_storage.retrieve(test_query, n_results=5)
            sqlite_results = await sqlite_storage.retrieve(test_query, n_results=5)

            print(f"📊 Search results - ChromaDB: {len(chroma_results)}, SQLite-vec: {len(sqlite_results)}")

            if len(sqlite_results) > 0:
                print("✅ Search functionality: WORKING")
            else:
                print("❌ Search functionality: FAILED")

        print("\\n🎉 Migration test completed!")

    finally:
        await chroma_storage.close()
        await sqlite_storage.close()


async def main():
    """Main test function."""
    import os

    # Default paths
    chroma_path = CHROMA_PATH
    sqlite_path = os.path.join(os.path.dirname(chroma_path), "memory_migrated.db")

    # Allow custom paths via command line
    if len(sys.argv) > 1:
        sqlite_path = sys.argv[1]

    print(f"📂 ChromaDB path: {chroma_path}")
    print(f"📂 SQLite-vec path: {sqlite_path}")
    print()

    # Check if files exist
    if not os.path.exists(chroma_path):
        print(f"❌ ChromaDB not found at: {chroma_path}")
        return 1

    if not os.path.exists(sqlite_path):
        print(f"❌ SQLite-vec database not found at: {sqlite_path}")
        print("💡 Run the migration script first: python scripts/migrate_chroma_to_sqlite.py")
        return 1

    await compare_storage_backends(chroma_path, sqlite_path)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
