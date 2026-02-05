#!/usr/bin/env python3
"""
Shared storage manager for MCP Memory Service.

This module provides a singleton storage instance that can be shared between
HTTP and MCP servers, preventing duplicate model loading and initialization.

The storage is initialized once and reused by both servers, saving ~500MB RAM
per additional server instance and avoiding race conditions.
"""

import asyncio
import logging
from threading import Lock
from typing import Optional

from .config import DATABASE_PATH, SQLITE_VEC_PATH
from .storage.base import MemoryStorage
from .storage.factory import create_storage_instance

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages a singleton storage instance for shared access."""

    _instance: Optional["StorageManager"] = None
    _lock: Lock = Lock()

    def __init__(self):
        """Initialize storage manager."""
        self._storage: MemoryStorage | None = None
        self._initialization_lock: asyncio.Lock = asyncio.Lock()
        self._initialized: bool = False

    @classmethod
    def get_instance(cls) -> "StorageManager":
        """Get singleton instance of StorageManager.

        Thread-safe singleton pattern ensures only one instance exists.

        Returns:
            StorageManager: The singleton instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    logger.info("Created new StorageManager singleton instance")
        return cls._instance

    async def get_storage(self) -> MemoryStorage:
        """Get or create the shared storage instance.

        This method is idempotent and thread-safe. Multiple concurrent calls
        will result in only one storage initialization.

        Returns:
            MemoryStorage: The shared storage instance
        """
        # Fast path - already initialized
        if self._initialized and self._storage is not None:
            return self._storage

        # Slow path - need to initialize
        async with self._initialization_lock:
            # Double-check after acquiring lock
            if self._initialized and self._storage is not None:
                return self._storage

            logger.info("Initializing shared storage instance...")

            # Determine which path to use based on backend
            # SQLite-vec and Hybrid use SQLITE_VEC_PATH
            # Web interface uses DATABASE_PATH for compatibility
            storage_path = SQLITE_VEC_PATH or DATABASE_PATH

            # Create storage using factory
            self._storage = await create_storage_instance(storage_path)
            self._initialized = True

            logger.info(f"✓ Shared storage initialized successfully: {type(self._storage).__name__}")

            return self._storage

    async def close(self) -> None:
        """Close the storage instance if it exists.

        Safe to call even if storage was never initialized.
        """
        if self._storage is not None:
            try:
                logger.info("Closing shared storage instance...")
                await self._storage.close()
                self._storage = None
                self._initialized = False
                logger.info("✓ Shared storage closed successfully")
            except Exception as e:
                logger.error(f"Error closing shared storage: {e}")

    def is_initialized(self) -> bool:
        """Check if storage has been initialized.

        Returns:
            bool: True if storage is initialized, False otherwise
        """
        return self._initialized and self._storage is not None


# Module-level convenience functions
_manager = StorageManager.get_instance()


async def get_shared_storage() -> MemoryStorage:
    """Get the shared storage instance.

    Convenience function that uses the singleton StorageManager.

    Returns:
        MemoryStorage: The shared storage instance
    """
    return await _manager.get_storage()


async def close_shared_storage() -> None:
    """Close the shared storage instance.

    Convenience function that uses the singleton StorageManager.
    """
    await _manager.close()


def is_storage_initialized() -> bool:
    """Check if shared storage has been initialized.

    Convenience function that uses the singleton StorageManager.

    Returns:
        bool: True if storage is initialized, False otherwise
    """
    return _manager.is_initialized()
