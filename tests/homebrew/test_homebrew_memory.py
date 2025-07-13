#!/usr/bin/env python3
"""
Test script for MCP Memory Service with Homebrew PyTorch integration.
"""
import asyncio
import json
import sys
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variables for testing
os.environ["MCP_MEMORY_STORAGE_BACKEND"] = "sqlite_vec"
os.environ["MCP_MEMORY_SQLITE_PATH"] = os.path.expanduser("~/Library/Application Support/mcp-memory/sqlite_vec.db")
os.environ["MCP_MEMORY_BACKUPS_PATH"] = os.path.expanduser("~/Library/Application Support/mcp-memory/backups")
os.environ["MCP_MEMORY_USE_ONNX"] = "1"
os.environ["MCP_MEMORY_USE_HOMEBREW_PYTORCH"] = "1"
os.environ["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))

# Import our modules
from src.mcp_memory_service.server_patch import apply_patches
from src.mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from src.mcp_memory_service.models.memory import Memory
from src.mcp_memory_service.utils.hashing import generate_content_hash

async def main():
    logger.info("=== MCP Memory Service with Homebrew PyTorch Test ===")
    
    # Apply patches
    apply_patches()
    
    # Initialize the storage
    db_path = os.environ["MCP_MEMORY_SQLITE_PATH"]
    logger.info(f"Using SQLite-vec database at: {db_path}")
    
    storage = SqliteVecMemoryStorage(db_path)
    await storage.initialize()
    
    # Run our own database health check
    logger.info("=== Database Health Check ===")
    if storage.conn is None:
        logger.error("Database connection is not initialized")
    else:
        try:
            cursor = storage.conn.execute('SELECT COUNT(*) FROM memories')
            memory_count = cursor.fetchone()[0]
            logger.info(f"Database connected successfully. Contains {memory_count} memories.")
            
            cursor = storage.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            logger.info(f"Database tables: {', '.join(tables)}")
            
            if not storage.embedding_model:
                logger.warning("No embedding model available. Limited functionality.")
                
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
    
    # Get database stats directly
    logger.info("=== Database Stats ===")
    try:
        stats = storage.get_stats()
        logger.info(json.dumps(stats, indent=2))
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
    
    # Store a test memory
    logger.info("=== Creating Test Memory ===")
    test_content = "This is a test memory created on " + datetime.now().isoformat()
    
    test_memory = Memory(
        content=test_content,
        content_hash=generate_content_hash(test_content),
        tags=["test", "homebrew"],
        memory_type="note",
        metadata={"source": "homebrew_test_script"}
    )
    logger.info(f"Memory content: {test_memory.content}")
    logger.info(f"Content hash: {test_memory.content_hash}")
    
    success, message = await storage.store(test_memory)
    logger.info(f"Store success: {success}")
    logger.info(f"Message: {message}")
    
    # Try to retrieve the memory by tag
    logger.info("=== Retrieving Memories by Tag ===")
    tag_results = await storage.search_by_tag(["homebrew"])
    
    if tag_results:
        logger.info(f"Found {len(tag_results)} memories with tag 'homebrew'")
        for i, memory in enumerate(tag_results):
            logger.info(f"  Result {i+1}:")
            logger.info(f"    Content: {memory.content}")
            logger.info(f"    Tags: {memory.tags}")
    else:
        logger.warning("No memories found with tag 'homebrew'")
    
    # Try to retrieve the memory by semantic search
    logger.info("=== Retrieving Memories by Semantic Search ===")
    results = await storage.retrieve("test memory homebrew", n_results=5)
    
    if results:
        logger.info(f"Found {len(results)} memories")
        for i, result in enumerate(results):
            logger.info(f"  Result {i+1}:")
            logger.info(f"    Content: {result.memory.content}")
            logger.info(f"    Tags: {result.memory.tags}")
            logger.info(f"    Score: {result.relevance_score}")
    else:
        logger.warning("No memories found by semantic search")
    
    logger.info("=== Test Complete ===")
    storage.close()

if __name__ == "__main__":
    asyncio.run(main())