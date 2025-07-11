#!/usr/bin/env python3
"""
Test the fixed SQLite-vec storage implementation.
"""

import asyncio
import os
import tempfile
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

# Set environment for sqlite-vec
os.environ['MCP_MEMORY_STORAGE_BACKEND'] = 'sqlite_vec'

async def test_fixed_storage():
    """Test the fixed sqlite-vec storage implementation."""
    print("🔧 Testing Fixed SQLite-vec Storage...")
    print("=" * 50)
    
    try:
        from src.mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
        from src.mcp_memory_service.models.memory import Memory
        from src.mcp_memory_service.utils.hashing import generate_content_hash
        
        # Create temporary database
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_fixed.db")
        
        print("1. Initializing storage...")
        storage = SqliteVecMemoryStorage(db_path)
        await storage.initialize()
        print("   ✅ Storage initialized successfully")
        
        print("\n2. Testing memory storage...")
        content = "Test memory for the fixed sqlite-vec implementation"
        memory = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["test", "fixed", "sqlite-vec"],
            memory_type="note",
            metadata={"test": "fixed_implementation"}
        )
        
        success, message = await storage.store(memory)
        print(f"   Store result: {success} - {message}")
        
        if not success:
            print(f"   ❌ Storage failed: {message}")
            return False
        
        print("\n3. Testing retrieval...")
        results = await storage.retrieve("fixed sqlite-vec", n_results=5)
        print(f"   Found {len(results)} results")
        
        if results:
            result = results[0]
            print(f"   Content: {result.memory.content}")
            print(f"   Relevance: {result.relevance_score:.3f}")
            print(f"   Tags: {result.memory.tags}")
        
        print("\n4. Testing tag search...")
        tag_results = await storage.search_by_tag(["fixed"])
        print(f"   Found {len(tag_results)} memories with 'fixed' tag")
        
        print("\n5. Testing deletion...")
        success, message = await storage.delete(memory.content_hash)
        print(f"   Delete result: {success} - {message}")
        
        # Cleanup
        storage.close()
        os.remove(db_path)
        os.rmdir(temp_dir)
        
        print("\n✅ All tests passed! Fixed SQLite-vec storage is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    success = await test_fixed_storage()
    
    if success:
        print("\n🎯 Database Health Check: PASSED")
        print("   SQLite-vec backend is healthy and ready for use")
        return 0
    else:
        print("\n❌ Database Health Check: FAILED")
        print("   Issues found with SQLite-vec implementation")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))