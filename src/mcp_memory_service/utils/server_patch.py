"""
Server patch to use Homebrew PyTorch integration.

This module patches the SQLite-vec storage backend to use Homebrew PyTorch
for generating embeddings.
"""

import os
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

def apply_patches():
    """Apply patches to the server components."""
    try:
        # Check if we should apply Homebrew PyTorch patches
        if os.environ.get('MCP_MEMORY_USE_HOMEBREW_PYTORCH', '').lower() in ('1', 'true', 'yes'):
            logger.info("Applying Homebrew PyTorch patches...")
            _patch_sqlite_vec_storage()
            logger.info("Homebrew PyTorch patches applied successfully")
        else:
            logger.info("Homebrew PyTorch integration disabled")
    except Exception as e:
        logger.error(f"Failed to apply patches: {e}")

def _patch_sqlite_vec_storage():
    """Patch SQLite-vec storage to use Homebrew PyTorch."""
    try:
        # Import the storage class
        from .storage.sqlite_vec import SqliteVecMemoryStorage
        # Import the Homebrew integration
        from .homebrew_integration import patch_storage
        
        # Save the original initialization method
        original_init_method = SqliteVecMemoryStorage._initialize_embedding_model
        
        # Define a patched method
        async def patched_initialize_embedding_model(self):
            """Patched method to initialize embedding model with Homebrew PyTorch support."""
            logger.info("Using patched embedding model initialization with Homebrew PyTorch support")
            
            # Try to use Homebrew PyTorch integration
            success = patch_storage(self)
            
            if success:
                logger.info("Successfully patched storage to use Homebrew PyTorch")
                return
                
            # Fall back to original method
            logger.info("Falling back to original embedding model initialization")
            await original_init_method(self)
        
        # Apply the patch
        SqliteVecMemoryStorage._initialize_embedding_model = patched_initialize_embedding_model
        logger.info("SQLite-vec storage patched successfully")
        
    except Exception as e:
        logger.error(f"Failed to patch SQLite-vec storage: {e}")