#\!/usr/bin/env python3
"""
MCP Memory Service Server with Homebrew PyTorch Integration
"""
import os
import sys
import asyncio
import logging
import json
from pathlib import Path

# Set environment variables to enable Homebrew PyTorch integration
os.environ["MCP_MEMORY_USE_HOMEBREW_PYTORCH"] = "1"
os.environ["MCP_MEMORY_STORAGE_BACKEND"] = "sqlite_vec"
os.environ["MCP_MEMORY_USE_ONNX"] = "1"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("memory-server")

# Set default paths if not set
if not os.environ.get("MCP_MEMORY_SQLITE_PATH"):
    home = str(Path.home())
    if sys.platform == "darwin":  # macOS
        base_dir = os.path.join(home, "Library", "Application Support", "mcp-memory")
    elif sys.platform == "win32":  # Windows
        base_dir = os.path.join(os.environ.get("LOCALAPPDATA", ""), "mcp-memory")
    else:  # Linux and others
        base_dir = os.path.join(home, ".local", "share", "mcp-memory")
    
    os.environ["MCP_MEMORY_SQLITE_PATH"] = os.path.join(base_dir, "sqlite_vec.db")
    os.environ["MCP_MEMORY_BACKUPS_PATH"] = os.path.join(base_dir, "backups")
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(os.environ["MCP_MEMORY_SQLITE_PATH"]), exist_ok=True)
    os.makedirs(os.environ["MCP_MEMORY_BACKUPS_PATH"], exist_ok=True)

async def main():
    """Run the memory server with Homebrew PyTorch integration."""
    try:
        # Import after setting environment variables
        from mcp_memory_service.server import main as server_main
        from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
        from mcp_memory_service.homebrew_integration import patch_storage
        
        print("=" * 80)
        print(" MCP Memory Service with Homebrew PyTorch Integration")
        print("=" * 80)
        print(f"SQLite-vec database: {os.environ.get('MCP_MEMORY_SQLITE_PATH')}")
        print(f"Backups path: {os.environ.get('MCP_MEMORY_BACKUPS_PATH')}")
        print(f"ONNX Runtime enabled: {os.environ.get('MCP_MEMORY_USE_ONNX', '1')}")
        print(f"Storage backend: {os.environ.get('MCP_MEMORY_STORAGE_BACKEND', 'sqlite_vec')}")
        print("=" * 80)
        
        # Monkey patch the SqliteVecMemoryStorage._initialize_embedding_model method
        original_init_model = SqliteVecMemoryStorage._initialize_embedding_model
        
        async def patched_init_model(self):
            """Patched method to use Homebrew PyTorch if available."""
            try:
                # First try to use the Homebrew PyTorch integration
                if patch_storage(self):
                    logger.info("Using Homebrew PyTorch for embeddings")
                    return
                # Fall back to the original method if patching fails
                await original_init_model(self)
            except Exception as e:
                logger.error(f"Error in patched _initialize_embedding_model: {e}")
                # Still try the original as a last resort
                try:
                    await original_init_model(self)
                except Exception as inner_e:
                    logger.error(f"Original _initialize_embedding_model also failed: {inner_e}")
        
        # Apply the monkey patch
        SqliteVecMemoryStorage._initialize_embedding_model = patched_init_model
        
        # Run the server
        await server_main()
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
