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
FastAPI dependencies for the HTTP interface.
"""

import logging
from typing import Optional
from fastapi import HTTPException

from ..storage.base import MemoryStorage

logger = logging.getLogger(__name__)

# Global storage instance
_storage: Optional[MemoryStorage] = None


def set_storage(storage: MemoryStorage) -> None:
    """Set the global storage instance."""
    global _storage
    _storage = storage


def get_storage() -> MemoryStorage:
    """Get the global storage instance."""
    if _storage is None:
        raise HTTPException(status_code=503, detail="Storage not initialized")
    return _storage


def _get_storage_backend_class() -> type:
    """Get storage backend class based on configuration (web-only version)."""
    from ..config import STORAGE_BACKEND

    backend = STORAGE_BACKEND.lower()

    if backend == "sqlite-vec" or backend == "sqlite_vec":
        from ..storage.sqlite_vec import SqliteVecMemoryStorage
        return SqliteVecMemoryStorage
    elif backend == "chroma":
        try:
            from ..storage.chroma import ChromaStorage
            return ChromaStorage
        except ImportError as e:
            logger.error(f"ChromaDB not installed. Install with: pip install mcp-memory-service[chromadb]")
            logger.error(f"Import error: {str(e)}")
            logger.warning("Falling back to SQLite-vec storage")
            from ..storage.sqlite_vec import SqliteVecMemoryStorage
            return SqliteVecMemoryStorage
    elif backend == "cloudflare":
        try:
            from ..storage.cloudflare import CloudflareStorage
            return CloudflareStorage
        except ImportError as e:
            logger.error(f"Failed to import Cloudflare storage: {e}")
            raise
    elif backend == "hybrid":
        try:
            from ..storage.hybrid import HybridMemoryStorage
            return HybridMemoryStorage
        except ImportError as e:
            logger.error(f"Failed to import Hybrid storage: {e}")
            logger.warning("Falling back to SQLite-vec storage")
            from ..storage.sqlite_vec import SqliteVecMemoryStorage
            return SqliteVecMemoryStorage
    else:
        logger.warning(f"Unknown storage backend '{backend}', defaulting to SQLite-vec")
        from ..storage.sqlite_vec import SqliteVecMemoryStorage
        return SqliteVecMemoryStorage


async def create_storage_backend() -> MemoryStorage:
    """
    Create and initialize storage backend for web interface based on configuration.

    Returns:
        Initialized storage backend
    """
    from ..config import (
        SQLITE_VEC_PATH, EMBEDDING_MODEL_NAME, DATABASE_PATH,
        CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID,
        CLOUDFLARE_VECTORIZE_INDEX, CLOUDFLARE_D1_DATABASE_ID,
        CLOUDFLARE_R2_BUCKET, CLOUDFLARE_EMBEDDING_MODEL,
        CLOUDFLARE_LARGE_CONTENT_THRESHOLD, CLOUDFLARE_MAX_RETRIES,
        CLOUDFLARE_BASE_DELAY, CHROMA_PATH, COLLECTION_METADATA,
        HYBRID_SYNC_INTERVAL, HYBRID_BATCH_SIZE
    )

    logger.info("Creating storage backend for web interface...")

    # Get storage class based on configuration
    StorageClass = _get_storage_backend_class()

    if StorageClass.__name__ == "SqliteVecMemoryStorage":
        # For HTTP interface, use DATABASE_PATH which is configured for web interface
        storage = StorageClass(
            db_path=DATABASE_PATH,
            embedding_model=EMBEDDING_MODEL_NAME
        )
        logger.info(f"Initialized SQLite-vec storage at {DATABASE_PATH}")

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
            sqlite_db_path=DATABASE_PATH,  # Use DATABASE_PATH for web interface
            embedding_model=EMBEDDING_MODEL_NAME,
            cloudflare_config=cloudflare_config,
            sync_interval=HYBRID_SYNC_INTERVAL,
            batch_size=HYBRID_BATCH_SIZE
        )
        logger.info(f"Initialized hybrid storage with SQLite at {DATABASE_PATH}")

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