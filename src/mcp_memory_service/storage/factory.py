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
    elif backend == "chroma":
        try:
            from .chroma import ChromaStorage
            return ChromaStorage
        except ImportError as e:
            logger.error(f"ChromaDB not installed. Install with: pip install mcp-memory-service[chromadb]")
            logger.error(f"Import error: {str(e)}")
            return _fallback_to_sqlite_vec()
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


async def create_storage_instance(sqlite_path: str) -> MemoryStorage:
    """
    Create and initialize storage backend instance based on configuration.

    Args:
        sqlite_path: Path to SQLite database file (used for SQLite-vec and Hybrid backends)

    Returns:
        Initialized storage backend instance
    """
    from ..config import (
        EMBEDDING_MODEL_NAME,
        CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID,
        CLOUDFLARE_VECTORIZE_INDEX, CLOUDFLARE_D1_DATABASE_ID,
        CLOUDFLARE_R2_BUCKET, CLOUDFLARE_EMBEDDING_MODEL,
        CLOUDFLARE_LARGE_CONTENT_THRESHOLD, CLOUDFLARE_MAX_RETRIES,
        CLOUDFLARE_BASE_DELAY, CHROMA_PATH, COLLECTION_METADATA,
        HYBRID_SYNC_INTERVAL, HYBRID_BATCH_SIZE
    )

    logger.info(f"Creating storage backend instance (sqlite_path: {sqlite_path})...")

    # Get storage class based on configuration
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

    else:  # ChromaStorage
        storage = StorageClass(
            path=str(CHROMA_PATH),
            collection_name=COLLECTION_METADATA.get("name", "memories")
        )
        logger.info(f"Initialized ChromaDB storage at {CHROMA_PATH}")

    # Initialize storage backend
    await storage.initialize()
    logger.info(f"Storage backend {StorageClass.__name__} initialized successfully")

    return storage