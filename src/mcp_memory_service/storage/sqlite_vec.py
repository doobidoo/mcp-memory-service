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

from typing import Optional

from ..models import ontology
from .base import MemoryStorage
from .ontology_overlay import (
    OverlaySource,
    tombstone_memory_type,
    tombstone_retention,
    upsert_memory_type,
    upsert_retention,
    validate_memory_type_entry,
    validate_retention_entry,
)
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

    # RetrieveMixin.get_by_hash scopes the lookup to ``store`` (AND store = ?),
    # so a store-scoped hash-addressed operation can be safely pre-checked.
    get_by_hash_is_store_scoped = True

    async def initialize(self, *args, **kwargs):
        await super().initialize(*args, **kwargs)
        # The ontology overlay (#1897) lives in this database. Registering makes
        # every type lookup in the process see it, including other processes' writes.
        source = OverlaySource(self.db_path)
        await self._run_in_thread(source.capture, self.conn)  # in-memory databases only
        self._ontology_overlay = source
        ontology.set_overlay_source(source)

    async def close(self):
        ontology.clear_overlay_source(getattr(self, "_ontology_overlay", None))
        await super().close()

    async def _write_ontology(self, write):
        """Run an overlay write under the connection lock, then refresh readers."""
        source = self._ontology_overlay

        def _op():
            result = write(self.conn)
            source.capture(self.conn)
            return result

        try:
            return await self._execute_with_retry(_op)
        finally:
            # Also on a rejected duplicate: the row that won came from somewhere
            # this process has not polled yet.
            ontology.invalidate_overlay()

    async def add_memory_type(self, name: str, parent: Optional[str] = None) -> str:
        """Add a memory type to the ontology overlay and return its canonical name.

        ``parent=None`` adds a base type; otherwise a subtype of ``parent``,
        which must be a base type. Raises ValueError if the name already exists.
        """
        name, parent = validate_memory_type_entry(name, parent)
        await self._write_ontology(lambda conn: upsert_memory_type(conn, name, parent))
        return name

    async def remove_memory_type(self, name: str) -> bool:
        """Tombstone an overlay memory type. Built-in and env types cannot be removed.

        Stored memories keep the type verbatim; it just stops validating for new writes.
        """
        name = ontology.canonicalize_memory_type(name)
        return await self._write_ontology(lambda conn: tombstone_memory_type(conn, name))

    async def set_retention(self, memory_type: str, days: int) -> str:
        """Store the retention period (days) of a memory type and return its canonical name.

        For types the consolidation config has no period for, such as an
        overlay base type that would otherwise decay on the 30-day fallback.
        Raises ValueError for an unknown type, a type the config already
        covers, or days outside 1..3650.
        """
        memory_type, days = validate_retention_entry(memory_type, days)
        await self._write_ontology(lambda conn: upsert_retention(conn, memory_type, days))
        return memory_type

    async def clear_retention(self, memory_type: str) -> bool:
        """Tombstone a stored retention period; the type falls back to its parent's."""
        memory_type = ontology.canonicalize_memory_type(memory_type)
        return await self._write_ontology(lambda conn: tombstone_retention(conn, memory_type))
