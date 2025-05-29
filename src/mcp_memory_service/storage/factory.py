"""
Storage Factory for MCP Memory Service
Copyright (c) 2025 EchoVault
Licensed under the MIT License.

This module provides a factory for creating storage implementations
based on environment configuration.
"""

import os
import logging
from typing import Optional

from .base import MemoryStorage
from .chroma import ChromaMemoryStorage

logger = logging.getLogger(__name__)

def create_storage(path: Optional[str] = None) -> MemoryStorage:
    """
    Create a storage implementation based on environment configuration.
    
    Args:
        path: Optional path for local storage
        
    Returns:
        An implementation of MemoryStorage
    """
    # Check if EchoVault is enabled
    use_echovault = os.environ.get("USE_ECHOVAULT", "").lower() in ("true", "1", "yes")
    
    if use_echovault:
        try:
            # Import here to avoid circular imports
            from .echovault import EchoVaultStorage
            logger.info("Using EchoVault storage implementation")
            return EchoVaultStorage(path)
        except ImportError as e:
            logger.warning(f"Failed to import EchoVaultStorage: {e}")
            logger.warning("Falling back to ChromaMemoryStorage")
            return ChromaMemoryStorage(path)
        except Exception as e:
            logger.error(f"Error initializing EchoVaultStorage: {e}")
            logger.warning("Falling back to ChromaMemoryStorage")
            return ChromaMemoryStorage(path)
    else:
        logger.info("Using standard ChromaMemoryStorage")
        return ChromaMemoryStorage(path)