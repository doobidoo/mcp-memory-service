#!/usr/bin/env python3
"""
Quick test script for sqlite-vec backend functionality on local Ubuntu setup.
"""

import asyncio
import os
import tempfile
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set environment for sqlite-vec
os.environ['MCP_MEMORY_STORAGE_BACKEND'] = 'sqlite_vec'

async def test_sqlite_vec_backend():
    """Test the sqlite-vec backend functionality."""
    print("🔧 Testing SQLite-vec backend on Ubuntu...")
    print("=" * 50)
    
    try:
        # Test imports
        print("1. Testing imports...")
        import sqlite_vec
        print("   ✅ sqlite_vec imported successfully")
        
        # Mock chromadb to avoid import error
        sys.modules['chromadb'] = type(sys)('chromadb')
        
        from src.mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
        from src.mcp_memory_service.models.memory import Memory
        from src.mcp_memory_service.utils.hashing import generate_content_hash
        print("   ✅ MCP Memory Service modules imported successfully")
        
        # Create temporary database
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_memory.db")
        print(f"   📁 Using temporary database: {db_path}")
        
        # Initialize storage
        print("\n2. Initializing storage...")
        storage = SqliteVecMemoryStorage(db_path)
        await storage.initialize()
        print("   ✅ Storage initialized successfully")
        
        # Test storing a memory
        print("\n3. Testing memory storage...")
        content = "This is a test memory for local Ubuntu setup with Claude Code integration"
        memory = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["test", "ubuntu", "claude-code"],
            memory_type="note",
            metadata={"environment": "local", "integration": "claude-code"}
        )
        
        success, message = await storage.store(memory)
        print(f"   Store result: {success} - {message}")
        
        if not success:
            print("   ❌ Failed to store memory")
            return False
        
        # Test retrieval
        print("\n4. Testing memory retrieval...")
        results = await storage.retrieve("ubuntu claude code", n_results=5)
        print(f"   Found {len(results)} results")
        
        if results:
            result = results[0]
            print(f"   Content: {result.memory.content[:50]}...")
            print(f"   Relevance: {result.relevance_score:.3f}")
            print(f"   Tags: {result.memory.tags}")
        
        # Test tag search
        print("\n5. Testing tag search...")
        tag_results = await storage.search_by_tag(["ubuntu"])
        print(f"   Found {len(tag_results)} memories with 'ubuntu' tag")
        
        # Get statistics
        print("\n6. Getting storage statistics...")
        stats = storage.get_stats()
        print(f"   Backend: {stats['backend']}")
        print(f"   Total memories: {stats['total_memories']}")
        print(f"   Database size: {stats['database_size_mb']} MB")
        print(f"   Embedding model: {stats['embedding_model']}")
        
        # Cleanup
        storage.close()
        os.remove(db_path)
        os.rmdir(temp_dir)
        
        print("\n✅ All tests passed! SQLite-vec backend is ready for use.")
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print("   💡 Make sure dependencies are installed:")
        print("      pip install sqlite-vec sentence-transformers")
        return False
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

async def main():
    """Main test function."""
    success = await test_sqlite_vec_backend()
    
    if success:
        print("\n🚀 Next Steps for Claude Code Integration:")
        print("   1. Start the MCP Memory Service:")
        print("      source venv/bin/activate")
        print("      export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec")
        print("      python -m src.mcp_memory_service.server")
        print("")
        print("   2. The service will use SQLite-vec backend automatically")
        print("   3. Memory database will be created at:")
        print(f"      {Path.home()}/.local/share/mcp-memory/sqlite_vec.db")
        print("")
        print("   4. For VS Code integration, install MCP extensions")
        
        return 0
    else:
        print("\n❌ Setup incomplete. Please resolve issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))