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
CLI utilities for MCP Memory Service.
"""

import os

from ..storage.base import MemoryStorage


async def get_storage(backend: str | None = None) -> MemoryStorage:
    """
    Get storage backend for CLI operations.

    Args:
        backend: Storage backend name ('sqlite_vec' or 'qdrant')

    Returns:
        Initialized storage backend
    """
    if backend is None:
        backend = os.getenv("MCP_MEMORY_STORAGE_BACKEND", "sqlite_vec").lower()

    backend = backend.lower()

    if backend in ("sqlite_vec", "sqlite-vec"):
        from ..config import EMBEDDING_MODEL_NAME, SQLITE_VEC_PATH
        from ..storage.sqlite_vec import SqliteVecMemoryStorage

        storage = SqliteVecMemoryStorage(db_path=SQLITE_VEC_PATH, embedding_model=EMBEDDING_MODEL_NAME)
        await storage.initialize()
        return storage

    elif backend == "qdrant":
        from ..config import EMBEDDING_MODEL_NAME, settings
        from ..storage.qdrant_storage import QdrantStorage

        if settings.qdrant.url:
            storage = QdrantStorage(
                url=settings.qdrant.url,
                embedding_model=EMBEDDING_MODEL_NAME,
                collection_name=settings.qdrant.COLLECTION_NAME,
                quantization_enabled=settings.qdrant.quantization_enabled,
                distance_metric=settings.qdrant.DISTANCE_METRIC,
            )
        else:
            storage = QdrantStorage(
                storage_path=settings.qdrant.storage_path,
                embedding_model=EMBEDDING_MODEL_NAME,
                collection_name=settings.qdrant.COLLECTION_NAME,
                quantization_enabled=settings.qdrant.quantization_enabled,
                distance_metric=settings.qdrant.DISTANCE_METRIC,
            )
        await storage.initialize()
        return storage

    else:
        raise ValueError(f"Unsupported storage backend: {backend}. Use 'sqlite_vec' or 'qdrant'.")
