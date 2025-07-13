#!/usr/bin/env python3
"""
Health check utility for MCP Memory Service.
Tests the critical functions to verify the service is working properly.
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime

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
from src.mcp_memory_service.utils.debug import check_embedding_model

async def run_health_check():
    """Run comprehensive health checks on the memory service."""
    logger.info("=== MCP Memory Service Health Check ===")
    
    # Apply patches
    apply_patches()
    
    # Initialize the storage
    db_path = os.environ["MCP_MEMORY_SQLITE_PATH"]
    logger.info(f"Using SQLite-vec database at: {db_path}")
    
    storage = SqliteVecMemoryStorage(db_path)
    await storage.initialize()
    
    # Check 1: Database connectivity
    logger.info("=== Check 1: Database Connectivity ===")
    if storage.conn is None:
        logger.error("❌ Database connection is not initialized")
        return False
    else:
        try:
            cursor = storage.conn.execute('SELECT COUNT(*) FROM memories')
            memory_count = cursor.fetchone()[0]
            logger.info(f"✅ Database connected successfully. Contains {memory_count} memories.")
        except Exception as e:
            logger.error(f"❌ Database error: {str(e)}")
            return False
    
    # Check 2: Embedding model
    logger.info("=== Check 2: Embedding Model ===")
    try:
        model_status = check_embedding_model(storage)
        logger.info(f"Model status: {json.dumps(model_status, indent=2)}")
        
        if model_status.get("model_loaded", False):
            logger.info("✅ Embedding model loaded successfully")
        else:
            logger.warning(f"⚠️ Embedding model not loaded: {model_status.get('error', 'Unknown error')}")
    except Exception as e:
        logger.error(f"❌ Error checking embedding model: {str(e)}")
    
    # Check 3: Memory storage
    logger.info("=== Check 3: Memory Storage ===")
    test_content = f"Health check test memory created on {datetime.now().isoformat()}"
    test_memory = Memory(
        content=test_content,
        content_hash=generate_content_hash(test_content),
        tags=["health-check", "test"],
        memory_type="test",
        metadata={"source": "health_check"}
    )
    
    try:
        success, message = await storage.store(test_memory)
        if success:
            logger.info(f"✅ Memory storage successful: {message}")
        else:
            logger.error(f"❌ Memory storage failed: {message}")
            return False
    except Exception as e:
        logger.error(f"❌ Exception during memory storage: {str(e)}")
        return False
    
    # Check 4: Tag-based retrieval
    logger.info("=== Check 4: Tag-based Retrieval ===")
    try:
        tag_results = await storage.search_by_tag(["health-check"])
        if tag_results:
            logger.info(f"✅ Tag search successful. Found {len(tag_results)} memories with tag 'health-check'")
        else:
            logger.warning("⚠️ Tag search returned no results")
    except Exception as e:
        logger.error(f"❌ Error during tag search: {str(e)}")
        return False
    
    # Check 5: Semantic search
    logger.info("=== Check 5: Semantic Search ===")
    try:
        results = await storage.retrieve("health check test", n_results=3)
        if results:
            logger.info(f"✅ Semantic search successful. Found {len(results)} memories")
            for i, result in enumerate(results):
                logger.info(f"  Result {i+1}: score={result.relevance_score:.3f}, content={result.memory.content[:30]}...")
        else:
            logger.warning("⚠️ Semantic search returned no results")
    except Exception as e:
        logger.error(f"❌ Error during semantic search: {str(e)}")
    
    # Clean up
    logger.info("=== Health Check Complete ===")
    storage.close()
    
    return True

if __name__ == "__main__":
    success = asyncio.run(run_health_check())
    if not success:
        sys.exit(1)  # Exit with error code if health check failed