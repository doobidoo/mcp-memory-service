#\!/usr/bin/env python3
"""
Test script for MCP Memory Service with Homebrew PyTorch embeddings.
"""
import os
import sys
import asyncio
import time
from datetime import datetime
import json

# Import our custom embedder
import homebrew_pytorch_embeddings

# Configure environment variables
os.environ["MCP_MEMORY_STORAGE_BACKEND"] = "sqlite_vec"
os.environ["MCP_MEMORY_SQLITE_PATH"] = os.path.expanduser("~/Library/Application Support/mcp-memory/sqlite_vec.db")
os.environ["MCP_MEMORY_BACKUPS_PATH"] = os.path.expanduser("~/Library/Application Support/mcp-memory/backups")
os.environ["MCP_MEMORY_USE_ONNX"] = "1"

# Import the MCP Memory Service modules
try:
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
    from mcp_memory_service.models.memory import Memory, MemoryQueryResult
    from mcp_memory_service.utils.hashing import generate_content_hash
    import sqlite_vec
except ImportError as e:
    print(f"Error importing MCP Memory Service modules: {e}")
    sys.exit(1)

# Create a custom storage class that uses our Homebrew PyTorch embedder
class HomebrewEmbeddingStorage(SqliteVecMemoryStorage):
    """SQLite-vec storage that uses Homebrew PyTorch for embeddings."""
    
    def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for text using Homebrew PyTorch."""
        try:
            # Check cache first
            if self.enable_cache:
                cache_key = hash(text)
                if cache_key in _EMBEDDING_CACHE:
                    return _EMBEDDING_CACHE[cache_key]
            
            # Generate embedding using Homebrew PyTorch
            embedding = homebrew_pytorch_embeddings.encode([text])[0]
            embedding_list = embedding.tolist()
            
            # Cache the result
            if self.enable_cache:
                _EMBEDDING_CACHE[cache_key] = embedding_list
            
            return embedding_list
            
        except Exception as e:
            print(f"Failed to generate embedding: {str(e)}")
            return [0.0] * self.embedding_dimension

# Global model cache for performance optimization
_EMBEDDING_CACHE = {}

async def main():
    print("=== MCP Memory Service Test with Homebrew PyTorch Embeddings ===")
    
    # Initialize the storage
    db_path = os.environ["MCP_MEMORY_SQLITE_PATH"]
    print(f"Using SQLite-vec database at: {db_path}")
    
    storage = HomebrewEmbeddingStorage(db_path)
    await storage.initialize()
    
    # Override the embedding model status
    storage.embedding_model = True  # Hack to make the storage think it has an embedding model
    
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
                
        except Exception as e:
            print(f"Database error: {str(e)}")
    
    # Get database stats
    print("\n=== Database Stats ===")
    stats = storage.get_stats()
    print(json.dumps(stats, indent=2))
    
    # Store a test memory
    print("\n=== Creating Test Memory ===")
    test_content = f"Homebrew PyTorch Test memory created at {datetime.now().isoformat()}"
    
    test_memory = Memory(
        content=test_content,
        content_hash=generate_content_hash(test_content),
        tags=["homebrew-test", "embeddings-test"],
        memory_type="note",
        metadata={"source": "homebrew_test_script"}
    )
    print(f"Memory content: {test_memory.content}")
    print(f"Content hash: {test_memory.content_hash}")
    
    success, message = await storage.store(test_memory)
    print(f"Store success: {success}")
    print(f"Message: {message}")
    
    # Try to retrieve the memory by tag
    print("\n=== Retrieving by Tag ===")
    memories = await storage.search_by_tag(["homebrew-test"])
    
    if memories:
        print(f"Found {len(memories)} memories with tag 'homebrew-test'")
        for i, memory in enumerate(memories):
            print(f"  Memory {i+1}: {memory.content[:60]}...")
    else:
        print("No memories found with tag 'homebrew-test'")
    
    # Try semantic search
    print("\n=== Semantic Search ===")
    results = await storage.retrieve("homebrew pytorch test", n_results=5)
    
    if results:
        print(f"Found {len(results)} memories via semantic search")
        for i, result in enumerate(results):
            print(f"  Result {i+1}:")
            print(f"    Content: {result.memory.content[:60]}...")
            print(f"    Score: {result.relevance_score}")
    else:
        print("No memories found via semantic search")
    
    # Try more general semantic search
    print("\n=== General Semantic Search ===")
    results = await storage.retrieve("test memory", n_results=5)
    
    if results:
        print(f"Found {len(results)} memories via general semantic search")
        for i, result in enumerate(results):
            print(f"  Result {i+1}:")
            print(f"    Content: {result.memory.content[:60]}...")
            print(f"    Score: {result.relevance_score}")
    else:
        print("No memories found via general semantic search")
    
    print("\n=== Test Complete ===")
    storage.close()

if __name__ == "__main__":
    asyncio.run(main())
