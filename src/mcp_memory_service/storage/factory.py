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
Storage backend factory for the MCP Memory Service.

Provides factory functions for creating storage backends.
Supported backends: Qdrant (production), SQLite-vec (development).
"""

import logging

from .base import MemoryStorage

logger = logging.getLogger(__name__)


def get_storage_backend_class() -> type[MemoryStorage]:
    """
    Get storage backend class based on configuration.

    Returns:
        Storage backend class (QdrantStorage or SqliteVecMemoryStorage)
    """
    from ..config import STORAGE_BACKEND

    backend = STORAGE_BACKEND.lower()

    if backend in ("sqlite-vec", "sqlite_vec"):
        from .sqlite_vec import SqliteVecMemoryStorage

        return SqliteVecMemoryStorage

    elif backend == "qdrant":
        from .qdrant_storage import QdrantStorage

        return QdrantStorage

    else:
        # Fail fast on invalid configuration - don't silently default
        supported = ["sqlite_vec", "sqlite-vec", "qdrant"]
        raise ValueError(
            f"Unknown storage backend '{backend}'. "
            f"Supported backends: {', '.join(supported)}. "
            f"Set MCP_MEMORY_STORAGE_BACKEND environment variable."
        )


async def create_storage_instance(sqlite_path: str) -> MemoryStorage:
    """
    Create and initialize storage backend instance based on configuration.

    Args:
        sqlite_path: Path to SQLite database file (used for SQLite-vec backend)

    Returns:
        Initialized storage backend instance
    """
    from ..config import EMBEDDING_MODEL_NAME, settings

    logger.info("Creating storage backend instance...")

    StorageClass = get_storage_backend_class()

    if StorageClass.__name__ == "SqliteVecMemoryStorage":
        storage = StorageClass(db_path=sqlite_path, embedding_model=EMBEDDING_MODEL_NAME)
        logger.info(f"Initialized SQLite-vec storage at {sqlite_path}")

    elif StorageClass.__name__ == "QdrantStorage":
        # Determine mode: server (URL) or embedded (path)
        if settings.qdrant.url:
            # Server mode - network client
            storage = StorageClass(
                url=settings.qdrant.url,
                embedding_model=EMBEDDING_MODEL_NAME,
                collection_name=settings.qdrant.COLLECTION_NAME,
                quantization_enabled=settings.qdrant.quantization_enabled,
                distance_metric=settings.qdrant.DISTANCE_METRIC,
            )
            logger.info(f"Initialized Qdrant storage in server mode: {settings.qdrant.url}")
        else:
            # Embedded mode - file-based
            storage = StorageClass(
                storage_path=settings.qdrant.storage_path,
                embedding_model=EMBEDDING_MODEL_NAME,
                collection_name=settings.qdrant.COLLECTION_NAME,
                quantization_enabled=settings.qdrant.quantization_enabled,
                distance_metric=settings.qdrant.DISTANCE_METRIC,
            )
            logger.info(f"Initialized Qdrant storage in embedded mode: {settings.qdrant.storage_path}")

    else:
        raise ValueError(f"Unsupported storage backend class: {StorageClass.__name__}")

    await storage.initialize()
    logger.info(f"Storage backend {StorageClass.__name__} initialized successfully")

    return storage
