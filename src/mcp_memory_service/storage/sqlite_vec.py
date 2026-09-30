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
SQLite-vec storage backend for MCP Memory Service.
Provides local vector similarity search using sqlite-vec extension.

Decomposed into mixins (§10) — this file composes the final class.
"""

from .base import MemoryStorage
from .mixins import (
    BaseMixin,
    MigrationsMixin,
    EmbeddingsMixin,
    StoreMixin,
    RetrieveMixin,
    HybridMixin,
    DeleteMixin,
    MetadataMixin,
)

# Re-export module-level names for backward compatibility (tests patch these)
from .mixins.embeddings import (
    clear_model_caches,
    get_model_cache_stats,
    _HashEmbeddingModel,
    _MODEL_CACHE,
    _DIMENSION_CACHE,
    SENTENCE_TRANSFORMERS_AVAILABLE,
)

# Re-export SentenceTransformer for test patching compatibility
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc]

# Re-export deserialize_embedding from base mixin
from .mixins.base import deserialize_embedding

# Re-export constants from retrieve mixin
from .mixins.retrieve import _MAX_TAG_SEARCH_CANDIDATES, _SQLITE_VEC_MAX_KNN_K


class SqliteVecMemoryStorage(
    BaseMixin,
    MigrationsMixin,
    EmbeddingsMixin,
    StoreMixin,
    RetrieveMixin,
    HybridMixin,
    DeleteMixin,
    MetadataMixin,
    MemoryStorage,
):
    """
    SQLite-vec based memory storage implementation.

    This backend provides local vector similarity search using sqlite-vec
    while maintaining the same interface as other storage backends.

    Composed from focused mixins:
    - BaseMixin: connection, threading, retry, utilities
    - MigrationsMixin: schema setup and migrations
    - EmbeddingsMixin: model initialization and embedding generation
    - StoreMixin: store and store_batch operations
    - RetrieveMixin: retrieve, search, and query operations
    - HybridMixin: BM25 + vector hybrid search
    - DeleteMixin: soft-delete and purge operations
    - MetadataMixin: metadata updates, conflicts, versioning
    """
    pass
