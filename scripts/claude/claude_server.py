#!/usr/bin/env python3
"""
Simple wrapper server for Claude Desktop that doesn't require Homebrew PyTorch.
"""

import os
import sys
import logging

# Set environment variables for the service
os.environ["MCP_MEMORY_STORAGE_BACKEND"] = "sqlite_vec"
os.environ["MCP_MEMORY_SQLITE_PATH"] = os.path.expanduser("~/Library/Application Support/mcp-memory/sqlite_vec.db")
os.environ["MCP_MEMORY_BACKUPS_PATH"] = os.path.expanduser("~/Library/Application Support/mcp-memory/backups")
os.environ["MCP_MEMORY_USE_ONNX"] = "1"

# Add source directory to PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import required modules
from src.mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

# Add sanitized method to SqliteVecMemoryStorage
if not hasattr(SqliteVecMemoryStorage, 'sanitized'):
    def sanitized(self, tags):
        """Sanitize and normalize tags to a JSON string."""
        import json
        if tags is None:
            return json.dumps([])
        
        # If we get a string, split it into an array
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
        # If we get an array, use it directly
        elif isinstance(tags, list):
            tags = [str(tag).strip() for tag in tags if str(tag).strip()]
        else:
            return json.dumps([])
                
        # Return JSON string representation of the array
        return json.dumps(tags)
        
    # Add the method to the class
    SqliteVecMemoryStorage.sanitized = sanitized

# Run the server
if __name__ == "__main__":
    print("Starting MCP Memory Service for Claude Desktop...")
    
    # Create directories if they don't exist
    db_dir = os.path.dirname(os.environ["MCP_MEMORY_SQLITE_PATH"])
    backups_dir = os.environ["MCP_MEMORY_BACKUPS_PATH"]
    os.makedirs(db_dir, exist_ok=True)
    os.makedirs(backups_dir, exist_ok=True)
    
    # Import and run the server
    from src.mcp_memory_service.server import main
    main()