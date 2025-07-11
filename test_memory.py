#!/usr/bin/env python3
"""
Simple test script to verify memory service functionality.
"""
import asyncio
import json
import sys
import os
from datetime import datetime

# Set environment variables for testing
os.environ["MCP_MEMORY_STORAGE_BACKEND"] = "sqlite_vec"
os.environ["MCP_MEMORY_SQLITE_PATH"] = os.path.expanduser("~/Library/Application Support/mcp-memory/sqlite_vec.db")
os.environ["MCP_MEMORY_BACKUPS_PATH"] = os.path.expanduser("~/Library/Application Support/mcp-memory/backups")
os.environ["MCP_MEMORY_USE_ONNX"] = "1"

# Import our modules
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.utils.db_utils import validate_database, get_database_stats

async def main():
    print("=== MCP Memory Service Test ===")
    
    # Initialize the storage
    db_path = os.environ["MCP_MEMORY_SQLITE_PATH"]
    print(f"Using SQLite-vec database at: {db_path}")
    
    storage = SqliteVecMemoryStorage(db_path)
    await storage.initialize()
    
    # Run our own database health check
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
            
            if not storage.embedding_model:
                print("No embedding model available. Limited functionality.")
                
        except Exception as e:
            print(f"Database error: {str(e)}")
    
    # Get database stats directly
    print("\n=== Database Stats ===")
    try:
        # Simple stats
        cursor = storage.conn.execute('SELECT COUNT(*) FROM memories')
        memory_count = cursor.fetchone()[0]
        
        # Get database file size
        db_path = storage.db_path
        file_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
        
        stats = {
            "backend": "sqlite-vec",
            "total_memories": memory_count,
            "database_size_bytes": file_size,
            "database_size_mb": round(file_size / (1024 * 1024), 2),
            "embedding_model": storage.embedding_model_name if hasattr(storage, 'embedding_model_name') else "none",
            "embedding_dimension": storage.embedding_dimension if hasattr(storage, 'embedding_dimension') else 0
        }
        print(json.dumps(stats, indent=2))
    except Exception as e:
        print(f"Error getting stats: {str(e)}")
    
    # Store a test memory
    print("\n=== Creating Test Memory ===")
    test_content = "This is a test memory created on " + datetime.now().isoformat()
    
    # Import the hash function
    from mcp_memory_service.utils.hashing import generate_content_hash
    
    test_memory = Memory(
        content=test_content,
        content_hash=generate_content_hash(test_content),
        tags=["test", "example"],
        memory_type="note",
        metadata={"source": "test_script"}
    )
    print(f"Memory content: {test_memory.content}")
    print(f"Content hash: {test_memory.content_hash}")
    
    success, message = await storage.store(test_memory)
    print(f"Store success: {success}")
    print(f"Message: {message}")
    
    # Try to retrieve the memory
    print("\n=== Retrieving Memories ===")
    results = await storage.retrieve("test memory", n_results=5)
    
    if results:
        print(f"Found {len(results)} memories")
        for i, result in enumerate(results):
            print(f"  Result {i+1}:")
            print(f"    Content: {result.memory.content}")
            print(f"    Tags: {result.memory.tags}")
            print(f"    Score: {result.relevance_score}")
    else:
        print("No memories found")
    
    print("\n=== Test Complete ===")
    storage.close()

if __name__ == "__main__":
    asyncio.run(main())