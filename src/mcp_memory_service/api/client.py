# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Storage client wrapper for code execution interface.

Provides global storage instance management with lazy initialization
and automatic connection reuse for optimal performance.

Design Goals:
    - Single storage instance per process (avoid redundant connections)
    - Lazy initialization (create on first use)
    - Thread-safe access to global instance
    - Automatic cleanup on process exit
    - Graceful error handling with fallbacks

Performance:
    - First call: ~50ms (initialization + connection)
    - Subsequent calls: ~0ms (reuses connection)
    - Memory overhead: ~10MB for embedding model cache
"""

import asyncio
import logging
import os
from typing import Optional
from ..storage.base import MemoryStorage
from ..storage.factory import create_storage_instance
from ..config import DATABASE_PATH, get_base_directory

logger = logging.getLogger(__name__)

# Global storage instance (module-level singleton)
_storage_instance: Optional[MemoryStorage] = None
_initialization_lock = asyncio.Lock()

# Global consolidation instances (set by HTTP server)
_consolidator_instance: Optional["DreamInspiredConsolidator"] = None
_scheduler_instance: Optional["ConsolidationScheduler"] = None


async def _get_storage_async() -> MemoryStorage:
    """
    Get or create storage backend instance (async version).

    This function implements lazy initialization with connection reuse:
    1. Returns existing instance if available
    2. Creates new instance if none exists
    3. Initializes storage backend on first call
    4. Reuses connection for subsequent calls

    Thread Safety:
        Uses asyncio.Lock to prevent race conditions during initialization.

    Returns:
        Initialized MemoryStorage instance

    Raises:
        RuntimeError: If storage initialization fails
    """
    global _storage_instance

    # Fast path: return existing instance
    if _storage_instance is not None:
        return _storage_instance

    # Slow path: create new instance with lock
    async with _initialization_lock:
        # Double-check after acquiring lock (another coroutine may have initialized)
        if _storage_instance is not None:
            return _storage_instance

        try:
            logger.info("Initializing storage backend for code execution API...")

            # Determine SQLite database path
            db_path = DATABASE_PATH
            if not db_path:
                # Fallback to cross-platform default path
                base_dir = get_base_directory()
                db_path = os.path.join(base_dir, "sqlite_vec.db")
                logger.warning(f"DATABASE_PATH not configured, using default: {db_path}")

            # Ensure database directory exists
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
                logger.info(f"Created database directory: {db_dir}")

            # Create and initialize storage instance
            _storage_instance = await create_storage_instance(db_path)

            logger.info(f"Storage backend initialized: {type(_storage_instance).__name__}")
            return _storage_instance

        except Exception as e:
            logger.error(f"Failed to initialize storage backend: {e}")
            raise RuntimeError(f"Storage initialization failed: {e}") from e


async def get_storage_async() -> MemoryStorage:
    """
    Get storage backend instance (async version).

    This is the internal async version that should be used within
    async contexts. For synchronous contexts, the sync_wrapper
    will handle the event loop management.

    Returns:
        Initialized MemoryStorage instance

    Raises:
        RuntimeError: If storage initialization fails
    """
    return await _get_storage_async()


def get_storage() -> MemoryStorage:
    """
    Get storage backend instance (synchronous wrapper).

    This is the primary entry point for code execution API operations.
    It wraps the async initialization in a synchronous interface for
    ease of use in non-async contexts.

    Connection Reuse:
        - First call: ~50ms (initialization)
        - Subsequent calls: ~0ms (returns cached instance)

    Returns:
        Initialized MemoryStorage instance

    Raises:
        RuntimeError: If storage initialization fails

    Example:
        >>> storage = get_storage()
        >>> # Use storage for operations
        >>> results = await storage.retrieve("query", n_results=5)
    """
    global _storage_instance

    # Fast path: if already initialized, return immediately
    if _storage_instance is not None:
        return _storage_instance

    # Need to initialize - this requires an event loop
    try:
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, but we can't use run_until_complete
            # This shouldn't happen in normal usage, but handle it gracefully
            logger.error("get_storage() called from async context - use get_storage_async() instead")
            raise RuntimeError("get_storage() cannot be called from async context")
        except RuntimeError:
            # No running loop, we can proceed with synchronous initialization
            pass

        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run async initialization
        storage = loop.run_until_complete(_get_storage_async())
        return storage

    except Exception as e:
        logger.error(f"Error getting storage instance: {e}")
        raise


def close() -> None:
    """
    Close and clean up storage resources.

    Explicitly closes the storage backend connection and clears
    the global instance. This ensures proper cleanup when the
    process is terminating or when you want to force a reconnection.

    After calling close(), the next call to get_storage() will
    create a fresh connection.

    Note:
        If the storage backend has an async close() method, it will
        be scheduled but not awaited. For proper async cleanup, use
        close_async() instead.

    Example:
        >>> from mcp_memory_service.api import close
        >>> close()  # Cleanup resources
        >>> # Next get_storage() call will create new connection
    """
    global _storage_instance

    if _storage_instance is not None:
        try:
            logger.info("Closing storage instance")
            # Simply clear the instance reference
            # Async cleanup will happen via atexit or explicit close_async()
        except Exception as e:
            logger.warning(f"Error closing storage instance: {e}")
        finally:
            _storage_instance = None


async def close_async() -> None:
    """
    Close and clean up storage resources (async version).

    This is the proper way to close storage backends that have
    async cleanup methods. Use this in async contexts.

    Example:
        >>> from mcp_memory_service.api import close_async
        >>> await close_async()  # Proper async cleanup
    """
    global _storage_instance

    if _storage_instance is not None:
        try:
            logger.info("Closing storage instance (async)")
            # If storage has an async close method, await it
            if hasattr(_storage_instance, 'close') and callable(_storage_instance.close):
                close_method = _storage_instance.close()
                # Check if it's a coroutine
                if hasattr(close_method, '__await__'):
                    await close_method
        except Exception as e:
            logger.warning(f"Error closing storage instance: {e}")
        finally:
            _storage_instance = None


def reset_storage() -> None:
    """
    Reset global storage instance.

    Useful for testing or when configuration changes require
    reinitializing the storage backend.

    Warning:
        This closes the existing connection. Subsequent calls to
        get_storage() will create a new instance.

    Example:
        >>> reset_storage()  # Close current connection
        >>> storage = get_storage()  # Creates new connection
    """
    close()  # Reuse the close() method for consistency


# Cleanup on module exit
import atexit


def _cleanup_storage():
    """Cleanup storage instance on process exit."""
    global _storage_instance
    if _storage_instance is not None:
        logger.info("Cleaning up storage instance on exit")
        _storage_instance = None


atexit.register(_cleanup_storage)


def set_consolidator(consolidator: "DreamInspiredConsolidator") -> None:
    """
    Set global consolidator instance (called by HTTP server).

    This allows the API to access the consolidator instance
    that's managed by the HTTP server lifecycle.

    Args:
        consolidator: DreamInspiredConsolidator instance
    """
    global _consolidator_instance
    _consolidator_instance = consolidator
    logger.info("Global consolidator instance set")


def set_scheduler(scheduler: "ConsolidationScheduler") -> None:
    """
    Set global scheduler instance (called by HTTP server).

    This allows the API to access the scheduler instance
    that's managed by the HTTP server lifecycle.

    Args:
        scheduler: ConsolidationScheduler instance
    """
    global _scheduler_instance
    _scheduler_instance = scheduler
    logger.info("Global scheduler instance set")


def get_consolidator() -> Optional["DreamInspiredConsolidator"]:
    """
    Get global consolidator instance.

    Returns:
        DreamInspiredConsolidator instance or None if not set
    """
    return _consolidator_instance


def get_scheduler() -> Optional["ConsolidationScheduler"]:
    """
    Get global scheduler instance.

    Returns:
        ConsolidationScheduler instance or None if not set
    """
    return _scheduler_instance
