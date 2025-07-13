#\!/usr/bin/env python3
"""
Test script for MCP Memory Service with Homebrew PyTorch.
"""
import os
import sys
import asyncio
import time
from datetime import datetime

# Configure environment variables
os.environ["MCP_MEMORY_STORAGE_BACKEND"] = "sqlite_vec"
os.environ["MCP_MEMORY_SQLITE_PATH"] = os.path.expanduser("~/Library/Application Support/mcp-memory/sqlite_vec.db")
os.environ["MCP_MEMORY_BACKUPS_PATH"] = os.path.expanduser("~/Library/Application Support/mcp-memory/backups")
os.environ["MCP_MEMORY_USE_ONNX"] = "1"

# Import the MCP Memory Service modules
try:
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
    from mcp_memory_service.models.memory import Memory
    from mcp_memory_service.utils.hashing import generate_content_hash
except ImportError as e:
    print(f"Error importing MCP Memory Service modules: {e}")
    sys.exit(1)

async def main():
    print("=== MCP Memory Service Test ===")
    
    # Initialize the storage
    db_path = os.environ["MCP_MEMORY_SQLITE_PATH"]
    print(f"Using SQLite-vec database at: {db_path}")
    
    storage = SqliteVecMemoryStorage(db_path)
    await storage.initialize()
    
    # Check database health
    print("\n=== Database Health Check ===")
    if storage.conn is None:
        print("Database connection is not initialized")
    else:
        try:
            cursor = storage.conn.execute('SELECT COUNT(*) FROM memories')
            memory_count = cursor.fetchone()[0]
            print(f"Database connected successfully. Contains {memory_count} memories.")
            
            cursor = storage.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"Database tables: {', '.join(tables)}")
            
            print(f"Embedding model availability: {storage.embedding_model is not None}")
            if not storage.embedding_model:
                print("No embedding model available. Limited functionality.")
                
        except Exception as e:
            print(f"Database error: {str(e)}")
    
    # Get database stats
    print("\n=== Database Stats ===")
    stats = storage.get_stats()
    import json
    print(json.dumps(stats, indent=2))
    
    # Store a test memory
    print("\n=== Creating Test Memory ===")
    test_content = f"MCP Test memory created at {datetime.now().isoformat()} with Homebrew PyTorch"
    
    test_memory = Memory(
        content=test_content,
        content_hash=generate_content_hash(test_content),
        tags=["mcp-test", "homebrew-pytorch"],
        memory_type="note",
        metadata={"source": "mcp_test_script"}
    )
    print(f"Memory content: {test_memory.content}")
    print(f"Content hash: {test_memory.content_hash}")
    
    success, message = await storage.store(test_memory)
    print(f"Store success: {success}")
    print(f"Message: {message}")
    
    # Try to retrieve the memory
    print("\n=== Retrieving by Tag ===")
    memories = await storage.search_by_tag(["mcp-test"])
    
    if memories:
        print(f"Found {len(memories)} memories with tag 'mcp-test'")
        for i, memory in enumerate(memories):
            print(f"  Memory {i+1}: {memory.content[:60]}...")
    else:
        print("No memories found with tag 'mcp-test'")
    
    # Try semantic search
    print("\n=== Semantic Search ===")
    results = await storage.retrieve("test memory homebrew pytorch", n_results=5)
    
    if results:
        print(f"Found {len(results)} memories via semantic search")
        for i, result in enumerate(results):
            print(f"  Result {i+1}:")
            print(f"    Content: {result.memory.content[:60]}...")
            print(f"    Score: {result.relevance_score}")
    else:
        print("No memories found via semantic search")
    
    print("\n=== Test Complete ===")
    storage.close()

if __name__ == "__main__":
    asyncio.run(main())
