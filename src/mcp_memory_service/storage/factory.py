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
Shared storage backend factory for the MCP Memory Service.

This module provides a single, shared factory function for creating storage backends,
eliminating code duplication between the MCP server and web interface initialization.
"""

import logging
from typing import Type

from .base import MemoryStorage

logger = logging.getLogger(__name__)


def _fallback_to_sqlite_vec() -> Type[MemoryStorage]:
    """
    Helper function to fallback to SQLite-vec storage when other backends fail to import.

    Returns:
        SqliteVecMemoryStorage class
    """
    logger.warning("Falling back to SQLite-vec storage")
    from .sqlite_vec import SqliteVecMemoryStorage
    return SqliteVecMemoryStorage


def get_storage_backend_class() -> Type[MemoryStorage]:
    """
    Get storage backend class based on configuration.

    Returns:
        Storage backend class
    """
    from ..config import STORAGE_BACKEND

    backend = STORAGE_BACKEND.lower()

    if backend == "sqlite-vec" or backend == "sqlite_vec":
        from .sqlite_vec import SqliteVecMemoryStorage
        return SqliteVecMemoryStorage
    elif backend == "cloudflare":
        try:
            from .cloudflare import CloudflareStorage
            return CloudflareStorage
        except ImportError as e:
            logger.error(f"Failed to import Cloudflare storage: {e}")
            raise
    elif backend == "hybrid":
        try:
            from .hybrid import HybridMemoryStorage
            return HybridMemoryStorage
        except ImportError as e:
            logger.error(f"Failed to import Hybrid storage: {e}")
            return _fallback_to_sqlite_vec()
    else:
        logger.warning(f"Unknown storage backend '{backend}', defaulting to SQLite-vec")
        from .sqlite_vec import SqliteVecMemoryStorage
        return SqliteVecMemoryStorage


async def create_storage_instance(sqlite_path: str, server_type: str = None) -> MemoryStorage:
    """
    Create and initialize storage backend instance based on configuration.

    Args:
        sqlite_path: Path to SQLite database file (used for SQLite-vec and Hybrid backends)
        server_type: Optional server type identifier ("mcp" or "http") to control hybrid sync ownership

    Returns:
        Initialized storage backend instance
    """
    from ..config import (
        STORAGE_BACKEND, EMBEDDING_MODEL_NAME,
        CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID,
        CLOUDFLARE_VECTORIZE_INDEX, CLOUDFLARE_D1_DATABASE_ID,
        CLOUDFLARE_R2_BUCKET, CLOUDFLARE_EMBEDDING_MODEL,
        CLOUDFLARE_LARGE_CONTENT_THRESHOLD, CLOUDFLARE_MAX_RETRIES,
        CLOUDFLARE_BASE_DELAY,
        HYBRID_SYNC_INTERVAL, HYBRID_BATCH_SIZE, HYBRID_SYNC_OWNER
    )

    logger.info(f"Creating storage backend instance (sqlite_path: {sqlite_path}, server_type: {server_type})...")

    # Check if we should override hybrid backend based on sync ownership (v8.27.0+)
    effective_backend = STORAGE_BACKEND
    if STORAGE_BACKEND == 'hybrid' and server_type and HYBRID_SYNC_OWNER != 'both':
        if HYBRID_SYNC_OWNER != server_type:
            logger.info(
                f"Sync ownership configured for '{HYBRID_SYNC_OWNER}' but this is '{server_type}' server. "
                f"Using SQLite-vec storage instead of Hybrid to avoid duplicate sync queues."
            )
            effective_backend = 'sqlite_vec'

    # Get storage class based on effective configuration
    if effective_backend == 'sqlite_vec':
        # Intentional switch to SQLite-vec (not a fallback/error case)
        from .sqlite_vec import SqliteVecMemoryStorage
        StorageClass = SqliteVecMemoryStorage
    else:
        # Use configured backend (hybrid or cloudflare)
        StorageClass = get_storage_backend_class()

    # Create storage instance based on backend type
    if StorageClass.__name__ == "SqliteVecMemoryStorage":
        storage = StorageClass(
            db_path=sqlite_path,
            embedding_model=EMBEDDING_MODEL_NAME
        )
        logger.info(f"Initialized SQLite-vec storage at {sqlite_path}")

    elif StorageClass.__name__ == "CloudflareStorage":
        storage = StorageClass(
            api_token=CLOUDFLARE_API_TOKEN,
            account_id=CLOUDFLARE_ACCOUNT_ID,
            vectorize_index=CLOUDFLARE_VECTORIZE_INDEX,
            d1_database_id=CLOUDFLARE_D1_DATABASE_ID,
            r2_bucket=CLOUDFLARE_R2_BUCKET,
            embedding_model=CLOUDFLARE_EMBEDDING_MODEL,
            large_content_threshold=CLOUDFLARE_LARGE_CONTENT_THRESHOLD,
            max_retries=CLOUDFLARE_MAX_RETRIES,
            base_delay=CLOUDFLARE_BASE_DELAY
        )
        logger.info(f"Initialized Cloudflare storage with vectorize index: {CLOUDFLARE_VECTORIZE_INDEX}")

    elif StorageClass.__name__ == "HybridMemoryStorage":
        # Prepare Cloudflare configuration dict
        cloudflare_config = None
        if all([CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_VECTORIZE_INDEX, CLOUDFLARE_D1_DATABASE_ID]):
            cloudflare_config = {
                'api_token': CLOUDFLARE_API_TOKEN,
                'account_id': CLOUDFLARE_ACCOUNT_ID,
                'vectorize_index': CLOUDFLARE_VECTORIZE_INDEX,
                'd1_database_id': CLOUDFLARE_D1_DATABASE_ID,
                'r2_bucket': CLOUDFLARE_R2_BUCKET,
                'embedding_model': CLOUDFLARE_EMBEDDING_MODEL,
                'large_content_threshold': CLOUDFLARE_LARGE_CONTENT_THRESHOLD,
                'max_retries': CLOUDFLARE_MAX_RETRIES,
                'base_delay': CLOUDFLARE_BASE_DELAY
            }

        storage = StorageClass(
            sqlite_db_path=sqlite_path,
            embedding_model=EMBEDDING_MODEL_NAME,
            cloudflare_config=cloudflare_config,
            sync_interval=HYBRID_SYNC_INTERVAL,
            batch_size=HYBRID_BATCH_SIZE
        )
        logger.info(f"Initialized hybrid storage with SQLite at {sqlite_path}")

    else:
        # Unknown storage backend - this should not happen as get_storage_backend_class
        # already handles unknown backends by falling back to SQLite-vec
        raise ValueError(f"Unsupported storage backend class: {StorageClass.__name__}")

    # Initialize storage backend
    await storage.initialize()
    logger.info(f"Storage backend {StorageClass.__name__} initialized successfully")

    return storage