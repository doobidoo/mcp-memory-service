"""
Homebrew PyTorch version of the MCP Memory Service server.

This module provides a version of the MCP Memory Service that uses the Homebrew PyTorch
installation for generating embeddings.
"""

import os
import sys
import logging
import importlib
import subprocess
from typing import Optional

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

def apply_module_override():
    """Override the SqliteVecMemoryStorage with our Homebrew-enabled version."""
    # First, check if we should use Homebrew PyTorch
    if os.environ.get('MCP_MEMORY_USE_HOMEBREW_PYTORCH', '').lower() not in ('1', 'true', 'yes'):
        logger.info("Homebrew PyTorch integration is disabled. Set MCP_MEMORY_USE_HOMEBREW_PYTORCH=1 to enable.")
        return False
        
    logger.info("MCP_MEMORY_USE_HOMEBREW_PYTORCH is enabled, proceeding with checks...")
        
    # Check if PyTorch is installed via Homebrew
    try:
        # Check if pytorch is in the brew list
        result = subprocess.run(
            ['brew', 'list'],
            capture_output=True,
            text=True,
            check=False
        )
        
        if 'pytorch' not in result.stdout:
            logger.error("PyTorch is not installed via Homebrew. Please install PyTorch first: brew install pytorch")
            return False
            
        logger.info("PyTorch is installed via Homebrew. Proceeding with Homebrew integration.")
    except Exception as e:
        logger.error(f"Error checking for Homebrew PyTorch: {e}")
        return False
    
    # Import the storage classes
    try:
        from .storage.sqlite_vec import SqliteVecMemoryStorage
        from .storage.homebrew_sqlite_vec import HomebrewSqliteVecMemoryStorage
        
        logger.info(f"Original class: {SqliteVecMemoryStorage}")
        logger.info(f"Homebrew class: {HomebrewSqliteVecMemoryStorage}")
        
        # Replace the class with our homebrew version
        sys.modules['mcp_memory_service.storage.sqlite_vec'].SqliteVecMemoryStorage = HomebrewSqliteVecMemoryStorage
        logger.info("Successfully overrode SqliteVecMemoryStorage with HomebrewSqliteVecMemoryStorage")
        
        # Verify the override worked
        import importlib
        storage_module = importlib.import_module('mcp_memory_service.storage.sqlite_vec')
        actual_class = storage_module.SqliteVecMemoryStorage
        logger.info(f"Verification - class after override: {actual_class}")
        logger.info(f"Override verification successful: {actual_class == HomebrewSqliteVecMemoryStorage}")
        
        return True
    except ImportError as e:
        logger.error(f"Failed to import storage modules: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to override SqliteVecMemoryStorage: {e}")
        return False

def main():
    """Run the MCP Memory Service with Homebrew PyTorch integration."""
    try:
        # Apply our module override first
        if not apply_module_override():
            logger.warning("Continuing with original SqliteVecMemoryStorage class")
        
        # Then import and run the server
        # Import only after we've overridden the storage class
        from .server import main as server_main
        logger.info("Starting MCP Memory Service with Homebrew PyTorch integration...")
        server_main()
    except Exception as e:
        logger.error(f"Failed to start server with Homebrew PyTorch integration: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()