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

"""PostgreSQL + pgvector storage backend for the MCP Memory Service.

Mirrors the sqlite_vec backend's surface but persists to a Postgres database
with the `vector` extension enabled. Designed for team-shared deployments
where concurrent writers, point-in-time backups, and SQL-level joins to
adjacent platform data matter more than a zero-dep single-file DB.

Embeddings are sourced from the same external embedding API the rest of
the service uses (configured via MCP_EXTERNAL_EMBEDDING_URL); the backend
stores them in a `vector(N)` column where N is the embedder's dimension.

Schema: see `_DDL` below. Tags live in `text[]` with a GIN index for O(1)
contains queries; metadata is `jsonb`. An HNSW index on the embedding
column gives sub-millisecond ANN search at team scale.
"""

import asyncio
import json
import logging
import os
import time
import traceback
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .base import MemoryStorage
from ..models.memory import Memory, MemoryQueryResult

logger = logging.getLogger(__name__)

try:
    import asyncpg  # type: ignore
    _ASYNCPG_AVAILABLE = True
except ImportError:
    asyncpg = None  # type: ignore
    _ASYNCPG_AVAILABLE = False

# pgvector's Python helpers register the `vector` codec on a connection.
try:
    from pgvector.asyncpg import register_vector  # type: ignore
    _PGVECTOR_AVAILABLE = True
except ImportError:
    register_vector = None  # type: ignore
    _PGVECTOR_AVAILABLE = False


_DEFAULT_SCHEMA = "public"
_MEMORIES_TABLE = "memories"
_EMBEDDINGS_TABLE = "memory_embeddings"


def _now() -> Tuple[float, str]:
    """Return (unix_seconds, iso_8601_utc)."""
    now = datetime.now(timezone.utc)
    return now.timestamp(), now.isoformat()


def _row_to_memory(row: "asyncpg.Record") -> Memory:
    """Build a Memory from a `memories` row.

    Tolerates both `tags` as text[] (the canonical shape) and as the legacy
    comma-joined string form used by older sqlite-vec exports during
    migration.
    """
    tags = row["tags"] or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    metadata = row["metadata"] or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}

    return Memory(
        content=row["content"],
        content_hash=row["content_hash"],
        tags=list(tags),
        memory_type=row["memory_type"],
        metadata=dict(metadata),
        created_at=float(row["created_at"]) if row["created_at"] is not None else 0.0,
        updated_at=float(row["updated_at"]) if row["updated_at"] is not None else 0.0,
        created_at_iso=row["created_at_iso"] or "",
        updated_at_iso=row["updated_at_iso"] or "",
    )


class PgVectorMemoryStorage(MemoryStorage):
    """PostgreSQL + pgvector storage backend.

    Use ``MCP_MEMORY_STORAGE_BACKEND=pgvector`` and configure a DSN via
    ``MCP_PGVECTOR_DSN``. Optional: ``MCP_PGVECTOR_SCHEMA`` (defaults to
    ``public``).

    The schema is created on first initialize() if missing — the backend
    is safe to point at an empty database. Existing data is left untouched.
    """

    def __init__(
        self,
        dsn: str,
        schema: str = _DEFAULT_SCHEMA,
        embedding_model: Optional[str] = None,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
    ) -> None:
        if not _ASYNCPG_AVAILABLE:
            raise ImportError(
                "asyncpg is not installed. Install with: pip install 'mcp-memory-service[pgvector]'"
            )
        if not _PGVECTOR_AVAILABLE:
            raise ImportError(
                "pgvector is not installed. Install with: pip install 'mcp-memory-service[pgvector]'"
            )

        self.dsn = dsn
        self.schema = schema or _DEFAULT_SCHEMA
        self.requested_embedding_model_name = embedding_model
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size

        self.pool: Optional["asyncpg.Pool"] = None
        self.embedding_model: Optional[Any] = None
        self.embedding_dimension: Optional[int] = None
        self.embedding_model_name: Optional[str] = None

        self._initialized = False

    # ---------------------------------------------------------------- properties

    @property
    def max_content_length(self) -> Optional[int]:
        # Postgres TEXT has no hard limit; cap for the embedder is the practical one.
        # nomic-embed-text-v1.5 has a 2048-token context (~8000 chars). Leave a buffer.
        return None

    @property
    def supports_chunking(self) -> bool:
        return False

    # --------------------------------------------------------------- lifecycle

    async def initialize(self) -> None:
        """Create the connection pool, install the vector extension if
        absent, and ensure tables/indexes exist."""
        if self._initialized:
            return

        logger.info("Initializing pgvector backend (schema=%s)", self.schema)

        self.pool = await asyncpg.create_pool(
            dsn=self.dsn,
            min_size=self.min_pool_size,
            max_size=self.max_pool_size,
            init=self._init_connection,
        )

        await self._initialize_embedding_model()

        # Now that we know the embedding dimension, ensure schema exists.
        async with self.pool.acquire() as conn:
            await self._ensure_schema(conn)

        self._initialized = True
        logger.info(
            "pgvector backend ready (dim=%s, model=%s)",
            self.embedding_dimension,
            self.embedding_model_name,
        )

    async def _init_connection(self, conn: "asyncpg.Connection") -> None:
        """Per-connection setup: register pgvector codec and search_path."""
        await register_vector(conn)
        # Quote the schema to be safe with mixed-case / reserved names.
        await conn.execute(f'SET search_path TO "{self.schema}", public;')

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None
            self._initialized = False

    # --------------------------------------------------------------- embeddings

    async def _initialize_embedding_model(self) -> None:
        """Resolve the embedding model. The pgvector backend only supports
        external embedding APIs (matching the team's deployed setup) — local
        sentence-transformers/ONNX paths are intentionally not wired here."""
        external_api_url = os.environ.get("MCP_EXTERNAL_EMBEDDING_URL")
        if not external_api_url:
            raise RuntimeError(
                "pgvector backend requires MCP_EXTERNAL_EMBEDDING_URL. "
                "Configure your embedding service (Ollama / llama.cpp / vLLM) and retry."
            )

        from ..embeddings.external_api import get_external_embedding_model

        model_name = (
            self.requested_embedding_model_name
            or os.environ.get("MCP_EXTERNAL_EMBEDDING_MODEL")
            or "nomic-embed-text"
        )
        try:
            ext_model = get_external_embedding_model(external_api_url, model_name)
        except (ConnectionError, RuntimeError) as e:
            raise RuntimeError(
                f"External embedding API at {external_api_url} is unreachable: {e}. "
                "Ensure your embedding service is running before starting mcp-memory-service."
            ) from e

        self.embedding_model = ext_model
        self.embedding_dimension = ext_model.embedding_dimension
        self.embedding_model_name = model_name
        logger.info(
            "External embedding API connected (model=%s, dim=%s)",
            self.embedding_model_name,
            self.embedding_dimension,
        )

    async def _embed(self, text: str) -> List[float]:
        if self.embedding_model is None:
            raise RuntimeError("Embedding model not initialized")
        # external_api models expose a sync .encode([texts]) returning ndarray.
        vectors = await asyncio.to_thread(self.embedding_model.encode, [text])
        first = vectors[0]
        return list(map(float, first))

    # --------------------------------------------------------------- DDL

    async def _ensure_schema(self, conn: "asyncpg.Connection") -> None:
        # Extension first; CREATE EXTENSION IF NOT EXISTS is idempotent.
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        if self.schema != _DEFAULT_SCHEMA:
            await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}";')

        dim = self.embedding_dimension
        if not dim:
            raise RuntimeError("embedding_dimension must be known before _ensure_schema")

        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS "{self.schema}"."{_MEMORIES_TABLE}" (
                id              BIGSERIAL PRIMARY KEY,
                content_hash    TEXT        NOT NULL UNIQUE,
                content         TEXT        NOT NULL,
                tags            TEXT[]      NOT NULL DEFAULT '{{}}',
                memory_type     TEXT,
                metadata        JSONB       NOT NULL DEFAULT '{{}}'::jsonb,
                created_at      DOUBLE PRECISION NOT NULL,
                updated_at      DOUBLE PRECISION NOT NULL,
                created_at_iso  TEXT,
                updated_at_iso  TEXT,
                deleted_at      TIMESTAMPTZ
            );
            """
        )

        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS "{self.schema}"."{_EMBEDDINGS_TABLE}" (
                memory_id     BIGINT PRIMARY KEY
                              REFERENCES "{self.schema}"."{_MEMORIES_TABLE}"(id)
                              ON DELETE CASCADE,
                content_hash  TEXT NOT NULL UNIQUE,
                embedding     vector({dim}) NOT NULL
            );
            """
        )

        # Indexes — IF NOT EXISTS so re-runs are no-ops.
        await conn.execute(
            f'CREATE INDEX IF NOT EXISTS memories_tags_gin '
            f'ON "{self.schema}"."{_MEMORIES_TABLE}" USING GIN (tags);'
        )
        await conn.execute(
            f'CREATE INDEX IF NOT EXISTS memories_metadata_gin '
            f'ON "{self.schema}"."{_MEMORIES_TABLE}" USING GIN (metadata);'
        )
        await conn.execute(
            f'CREATE INDEX IF NOT EXISTS memories_created_at '
            f'ON "{self.schema}"."{_MEMORIES_TABLE}" (created_at DESC) '
            f'WHERE deleted_at IS NULL;'
        )
        await conn.execute(
            f'CREATE INDEX IF NOT EXISTS memories_active_hash '
            f'ON "{self.schema}"."{_MEMORIES_TABLE}" (content_hash) '
            f'WHERE deleted_at IS NULL;'
        )
        # HNSW gives strong recall and low latency at team scale; cosine matches
        # how nomic / minilm / bge embeddings are trained.
        await conn.execute(
            f'CREATE INDEX IF NOT EXISTS memory_embeddings_hnsw '
            f'ON "{self.schema}"."{_EMBEDDINGS_TABLE}" '
            f'USING hnsw (embedding vector_cosine_ops);'
        )

    # --------------------------------------------------------------- helpers

    def _t(self, name: str) -> str:
        """Fully-qualified, quoted table name."""
        return f'"{self.schema}"."{name}"'

    async def _ensure_pool(self) -> "asyncpg.Pool":
        if self.pool is None:
            raise RuntimeError("Storage not initialized — call initialize() first")
        return self.pool

    # --------------------------------------------------------------- store

    async def store(self, memory: Memory, skip_semantic_dedup: bool = False) -> Tuple[bool, str]:
        try:
            pool = await self._ensure_pool()

            try:
                embedding = await self._embed(memory.content)
            except Exception as e:
                logger.error("Embedding failed for %s: %s", memory.content_hash, e)
                return False, f"Failed to generate embedding: {e}"

            async with pool.acquire() as conn:
                async with conn.transaction():
                    existing = await conn.fetchrow(
                        f"SELECT id, deleted_at FROM {self._t(_MEMORIES_TABLE)} "
                        f"WHERE content_hash = $1",
                        memory.content_hash,
                    )
                    if existing and existing["deleted_at"] is None:
                        return False, "Duplicate content detected (exact match)"

                    if existing and existing["deleted_at"] is not None:
                        # Resurrect the soft-deleted row by clearing the tombstone
                        # and refreshing the embedding + metadata.
                        memory_id = existing["id"]
                        await conn.execute(
                            f"""
                            UPDATE {self._t(_MEMORIES_TABLE)}
                            SET content = $2, tags = $3, memory_type = $4,
                                metadata = $5, created_at = $6, updated_at = $7,
                                created_at_iso = $8, updated_at_iso = $9,
                                deleted_at = NULL
                            WHERE id = $1
                            """,
                            memory_id, memory.content, memory.tags or [],
                            memory.memory_type, json.dumps(memory.metadata or {}),
                            memory.created_at, memory.updated_at,
                            memory.created_at_iso, memory.updated_at_iso,
                        )
                        await conn.execute(
                            f"""
                            INSERT INTO {self._t(_EMBEDDINGS_TABLE)} (memory_id, content_hash, embedding)
                            VALUES ($1, $2, $3)
                            ON CONFLICT (memory_id) DO UPDATE SET embedding = EXCLUDED.embedding
                            """,
                            memory_id, memory.content_hash, embedding,
                        )
                        return True, "Memory stored successfully (resurrected from tombstone)"

                    row = await conn.fetchrow(
                        f"""
                        INSERT INTO {self._t(_MEMORIES_TABLE)}
                            (content_hash, content, tags, memory_type, metadata,
                             created_at, updated_at, created_at_iso, updated_at_iso)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        RETURNING id
                        """,
                        memory.content_hash, memory.content, memory.tags or [],
                        memory.memory_type, json.dumps(memory.metadata or {}),
                        memory.created_at, memory.updated_at,
                        memory.created_at_iso, memory.updated_at_iso,
                    )
                    await conn.execute(
                        f"""
                        INSERT INTO {self._t(_EMBEDDINGS_TABLE)} (memory_id, content_hash, embedding)
                        VALUES ($1, $2, $3)
                        """,
                        row["id"], memory.content_hash, embedding,
                    )
            return True, "Memory stored successfully"

        except Exception as e:
            logger.error("Failed to store memory: %s\n%s", e, traceback.format_exc())
            return False, f"Failed to store memory: {e}"

    async def store_batch(self, memories: List[Memory]) -> List[Tuple[bool, str]]:
        """Store a batch of memories in a single transaction.

        Embeddings are generated concurrently (the external API typically
        accepts batched inputs, but the encoder API is per-text); the DB
        writes happen inside one transaction so a failure rolls back the
        whole batch.
        """
        if not memories:
            return []
        try:
            pool = await self._ensure_pool()

            embeddings: List[Optional[List[float]]] = []
            for mem in memories:
                try:
                    emb = await self._embed(mem.content)
                except Exception as e:
                    logger.error("Embedding failed in batch for %s: %s", mem.content_hash, e)
                    emb = None
                embeddings.append(emb)

            results: List[Tuple[bool, str]] = []
            async with pool.acquire() as conn:
                async with conn.transaction():
                    for mem, emb in zip(memories, embeddings):
                        if emb is None:
                            results.append((False, "Failed to generate embedding"))
                            continue
                        try:
                            row = await conn.fetchrow(
                                f"""
                                INSERT INTO {self._t(_MEMORIES_TABLE)}
                                    (content_hash, content, tags, memory_type, metadata,
                                     created_at, updated_at, created_at_iso, updated_at_iso)
                                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                                ON CONFLICT (content_hash) DO NOTHING
                                RETURNING id
                                """,
                                mem.content_hash, mem.content, mem.tags or [],
                                mem.memory_type, json.dumps(mem.metadata or {}),
                                mem.created_at, mem.updated_at,
                                mem.created_at_iso, mem.updated_at_iso,
                            )
                            if row is None:
                                results.append((False, "Duplicate content detected (exact match)"))
                                continue
                            await conn.execute(
                                f"""
                                INSERT INTO {self._t(_EMBEDDINGS_TABLE)} (memory_id, content_hash, embedding)
                                VALUES ($1, $2, $3)
                                """,
                                row["id"], mem.content_hash, emb,
                            )
                            results.append((True, "Memory stored successfully"))
                        except Exception as e:
                            results.append((False, f"Failed to store memory: {e}"))
            return results
        except Exception as e:
            logger.error("Batch store failed: %s", e)
            return [(False, f"Batch store failed: {e}") for _ in memories]

    # --------------------------------------------------------------- retrieve

    async def retrieve(
        self,
        query: str,
        n_results: int = 5,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.0,
    ) -> List[MemoryQueryResult]:
        try:
            pool = await self._ensure_pool()
            query_embedding = await self._embed(query)

            tag_filter_sql = ""
            params: List[Any] = [query_embedding]
            if tags:
                tag_filter_sql = " AND m.tags && $2::text[]"
                params.append(list(tags))
            params.append(n_results)
            limit_param_index = len(params)

            sql = f"""
                SELECT m.content_hash, m.content, m.tags, m.memory_type, m.metadata,
                       m.created_at, m.updated_at, m.created_at_iso, m.updated_at_iso,
                       1 - (e.embedding <=> $1::vector) AS similarity
                FROM {self._t(_MEMORIES_TABLE)} m
                JOIN {self._t(_EMBEDDINGS_TABLE)} e ON e.memory_id = m.id
                WHERE m.deleted_at IS NULL{tag_filter_sql}
                ORDER BY e.embedding <=> $1::vector
                LIMIT ${limit_param_index}
            """

            async with pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)

            results: List[MemoryQueryResult] = []
            for row in rows:
                similarity = float(row["similarity"])
                if min_confidence and similarity < min_confidence:
                    continue
                results.append(
                    MemoryQueryResult(
                        memory=_row_to_memory(row),
                        relevance_score=similarity,
                        debug_info={"backend": "pgvector", "metric": "cosine"},
                    )
                )
            return results
        except Exception as e:
            logger.error("retrieve() failed: %s\n%s", e, traceback.format_exc())
            return []

    async def get_by_hash(self, content_hash: str) -> Optional[Memory]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, created_at_iso, updated_at_iso
                FROM {self._t(_MEMORIES_TABLE)}
                WHERE content_hash = $1 AND deleted_at IS NULL
                """,
                content_hash,
            )
        return _row_to_memory(row) if row else None

    async def get_by_exact_content(self, content: str) -> List[Memory]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, created_at_iso, updated_at_iso
                FROM {self._t(_MEMORIES_TABLE)}
                WHERE content = $1 AND deleted_at IS NULL
                ORDER BY created_at DESC
                """,
                content,
            )
        return [_row_to_memory(r) for r in rows]

    # --------------------------------------------------------------- tag search

    async def search_by_tag(
        self, tags: List[str], time_start: Optional[float] = None
    ) -> List[Memory]:
        if not tags:
            return []
        try:
            pool = await self._ensure_pool()
            stripped = [t.strip() for t in tags if t and t.strip()]
            if not stripped:
                return []

            params: List[Any] = [stripped]
            time_clause = ""
            if time_start is not None:
                params.append(time_start)
                time_clause = f" AND created_at >= ${len(params)}"

            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""
                    SELECT content_hash, content, tags, memory_type, metadata,
                           created_at, updated_at, created_at_iso, updated_at_iso
                    FROM {self._t(_MEMORIES_TABLE)}
                    WHERE tags && $1::text[] AND deleted_at IS NULL{time_clause}
                    ORDER BY created_at DESC
                    """,
                    *params,
                )
            return [_row_to_memory(r) for r in rows]
        except Exception as e:
            logger.error("search_by_tag() failed: %s", e)
            return []

    async def search_by_tags(
        self,
        tags: List[str],
        operation: str = "AND",
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
    ) -> List[Memory]:
        if not tags:
            return []
        try:
            pool = await self._ensure_pool()
            stripped = [t.strip() for t in tags if t and t.strip()]
            if not stripped:
                return []

            # `&&` (overlap) for OR semantics, `@>` (contains all) for AND.
            tag_op = "@>" if operation.upper() == "AND" else "&&"
            params: List[Any] = [stripped]
            extras: List[str] = []
            if time_start is not None:
                params.append(time_start)
                extras.append(f"AND created_at >= ${len(params)}")
            if time_end is not None:
                params.append(time_end)
                extras.append(f"AND created_at <= ${len(params)}")

            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""
                    SELECT content_hash, content, tags, memory_type, metadata,
                           created_at, updated_at, created_at_iso, updated_at_iso
                    FROM {self._t(_MEMORIES_TABLE)}
                    WHERE tags {tag_op} $1::text[] AND deleted_at IS NULL
                          {' '.join(extras)}
                    ORDER BY created_at DESC
                    """,
                    *params,
                )
            return [_row_to_memory(r) for r in rows]
        except Exception as e:
            logger.error("search_by_tags() failed: %s", e)
            return []

    # --------------------------------------------------------------- delete

    async def delete(self, content_hash: str) -> Tuple[bool, str]:
        """Soft-delete a memory by hash. Embedding row is kept until
        purge_deleted() runs so accidental deletions can be reversed."""
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                result = await conn.execute(
                    f"""
                    UPDATE {self._t(_MEMORIES_TABLE)}
                    SET deleted_at = NOW()
                    WHERE content_hash = $1 AND deleted_at IS NULL
                    """,
                    content_hash,
                )
            # asyncpg execute() returns "UPDATE n"
            try:
                affected = int(result.split()[-1])
            except Exception:
                affected = 0
            if affected == 0:
                return False, "No active memory found for that hash"
            return True, f"Deleted memory {content_hash}"
        except Exception as e:
            logger.error("delete() failed: %s", e)
            return False, f"Failed to delete memory: {e}"

    async def delete_by_tag(self, tag: str) -> Tuple[int, str]:
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                result = await conn.execute(
                    f"""
                    UPDATE {self._t(_MEMORIES_TABLE)}
                    SET deleted_at = NOW()
                    WHERE $1 = ANY(tags) AND deleted_at IS NULL
                    """,
                    tag,
                )
            try:
                affected = int(result.split()[-1])
            except Exception:
                affected = 0
            return affected, f"Deleted {affected} memories tagged {tag!r}"
        except Exception as e:
            logger.error("delete_by_tag() failed: %s", e)
            return 0, f"Failed to delete by tag: {e}"

    async def is_deleted(self, content_hash: str) -> bool:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT deleted_at FROM {self._t(_MEMORIES_TABLE)} WHERE content_hash = $1",
                content_hash,
            )
        return bool(row and row["deleted_at"] is not None)

    async def purge_deleted(self, older_than_days: int = 30) -> int:
        """Permanently remove tombstones older than `older_than_days`."""
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                result = await conn.execute(
                    f"""
                    DELETE FROM {self._t(_MEMORIES_TABLE)}
                    WHERE deleted_at IS NOT NULL
                      AND deleted_at < NOW() - ($1::int || ' days')::interval
                    """,
                    older_than_days,
                )
            try:
                return int(result.split()[-1])
            except Exception:
                return 0
        except Exception as e:
            logger.error("purge_deleted() failed: %s", e)
            return 0

    # --------------------------------------------------------------- updates

    async def update_memory_metadata(
        self,
        content_hash: str,
        updates: Dict[str, Any],
        preserve_timestamps: bool = True,
    ) -> Tuple[bool, str]:
        """Update tags / metadata / memory_type for an existing memory.

        Tags merge semantics: if `updates['tags']` is provided it REPLACES
        the existing tags (matching sqlite_vec behavior). Pass an empty
        list to clear.
        """
        if not updates:
            return False, "No updates provided"
        try:
            pool = await self._ensure_pool()

            sets: List[str] = []
            params: List[Any] = []

            if "tags" in updates:
                params.append(list(updates["tags"] or []))
                sets.append(f"tags = ${len(params)}")
            if "metadata" in updates:
                params.append(json.dumps(updates["metadata"] or {}))
                sets.append(f"metadata = ${len(params)}::jsonb")
            if "memory_type" in updates:
                params.append(updates["memory_type"])
                sets.append(f"memory_type = ${len(params)}")

            now_unix, now_iso = _now()
            params.append(now_unix)
            sets.append(f"updated_at = ${len(params)}")
            params.append(now_iso)
            sets.append(f"updated_at_iso = ${len(params)}")

            if not preserve_timestamps:
                params.append(now_unix)
                sets.append(f"created_at = ${len(params)}")
                params.append(now_iso)
                sets.append(f"created_at_iso = ${len(params)}")

            params.append(content_hash)
            where_idx = len(params)

            sql = (
                f"UPDATE {self._t(_MEMORIES_TABLE)} "
                f"SET {', '.join(sets)} "
                f"WHERE content_hash = ${where_idx} AND deleted_at IS NULL"
            )

            async with pool.acquire() as conn:
                result = await conn.execute(sql, *params)

            try:
                affected = int(result.split()[-1])
            except Exception:
                affected = 0
            if affected == 0:
                return False, "No active memory found for that hash"
            return True, f"Updated memory {content_hash}"
        except Exception as e:
            logger.error("update_memory_metadata() failed: %s", e)
            return False, f"Failed to update memory: {e}"

    # --------------------------------------------------------------- dedupe

    async def cleanup_duplicates(self) -> Tuple[int, str]:
        """Remove duplicate active rows sharing the same content.

        Postgres already enforces uniqueness on content_hash, so within a
        single namespace this is a no-op for hash-duplicates. We additionally
        deduplicate by raw content (keeping the oldest), matching the
        sqlite_vec behavior.
        """
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                async with conn.transaction():
                    duplicates = await conn.fetch(
                        f"""
                        SELECT id FROM (
                            SELECT id,
                                   ROW_NUMBER() OVER (PARTITION BY content ORDER BY created_at) AS rn
                            FROM {self._t(_MEMORIES_TABLE)}
                            WHERE deleted_at IS NULL
                        ) t
                        WHERE rn > 1
                        """
                    )
                    if not duplicates:
                        return 0, "No duplicate content found"

                    ids = [r["id"] for r in duplicates]
                    await conn.execute(
                        f"""
                        UPDATE {self._t(_MEMORIES_TABLE)}
                        SET deleted_at = NOW()
                        WHERE id = ANY($1::bigint[])
                        """,
                        ids,
                    )
            return len(duplicates), f"Removed {len(duplicates)} duplicate memories"
        except Exception as e:
            logger.error("cleanup_duplicates() failed: %s", e)
            return 0, f"Failed to cleanup duplicates: {e}"

    # --------------------------------------------------------------- listings

    async def get_all_memories(
        self,
        limit: int = None,
        offset: int = 0,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Memory]:
        try:
            pool = await self._ensure_pool()
            params: List[Any] = []
            extras: List[str] = []
            if memory_type:
                params.append(memory_type)
                extras.append(f"AND memory_type = ${len(params)}")
            if tags:
                params.append(list(tags))
                extras.append(f"AND tags && ${len(params)}::text[]")

            limit_clause = ""
            if limit is not None:
                params.append(limit)
                limit_clause = f"LIMIT ${len(params)} "
            if offset:
                params.append(offset)
                limit_clause += f"OFFSET ${len(params)}"

            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""
                    SELECT content_hash, content, tags, memory_type, metadata,
                           created_at, updated_at, created_at_iso, updated_at_iso
                    FROM {self._t(_MEMORIES_TABLE)}
                    WHERE deleted_at IS NULL {' '.join(extras)}
                    ORDER BY created_at DESC
                    {limit_clause}
                    """,
                    *params,
                )
            return [_row_to_memory(r) for r in rows]
        except Exception as e:
            logger.error("get_all_memories() failed: %s", e)
            return []

    async def get_recent_memories(self, n: int = 10) -> List[Memory]:
        return await self.get_all_memories(limit=n)

    async def get_memories_by_time_range(
        self, start_time: float, end_time: float
    ) -> List[Memory]:
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""
                    SELECT content_hash, content, tags, memory_type, metadata,
                           created_at, updated_at, created_at_iso, updated_at_iso
                    FROM {self._t(_MEMORIES_TABLE)}
                    WHERE deleted_at IS NULL
                      AND created_at >= $1 AND created_at <= $2
                    ORDER BY created_at DESC
                    """,
                    start_time, end_time,
                )
            return [_row_to_memory(r) for r in rows]
        except Exception as e:
            logger.error("get_memories_by_time_range() failed: %s", e)
            return []

    # --------------------------------------------------------------- counts/tags

    async def count_memories_by_tag(self, tags: List[str]) -> int:
        if not tags:
            return 0
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    SELECT COUNT(*) AS c
                    FROM {self._t(_MEMORIES_TABLE)}
                    WHERE deleted_at IS NULL AND tags && $1::text[]
                    """,
                    list(tags),
                )
            return int(row["c"]) if row else 0
        except Exception as e:
            logger.error("count_memories_by_tag() failed: %s", e)
            return 0

    async def count_all_memories(
        self,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> int:
        try:
            pool = await self._ensure_pool()
            params: List[Any] = []
            extras: List[str] = []
            if memory_type:
                params.append(memory_type)
                extras.append(f"AND memory_type = ${len(params)}")
            if tags:
                params.append(list(tags))
                extras.append(f"AND tags && ${len(params)}::text[]")

            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    SELECT COUNT(*) AS c
                    FROM {self._t(_MEMORIES_TABLE)}
                    WHERE deleted_at IS NULL {' '.join(extras)}
                    """,
                    *params,
                )
            return int(row["c"]) if row else 0
        except Exception as e:
            logger.error("count_all_memories() failed: %s", e)
            return 0

    async def get_all_tags(self) -> List[str]:
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""
                    SELECT DISTINCT unnest(tags) AS tag
                    FROM {self._t(_MEMORIES_TABLE)}
                    WHERE deleted_at IS NULL
                    ORDER BY tag
                    """
                )
            return [r["tag"] for r in rows]
        except Exception as e:
            logger.error("get_all_tags() failed: %s", e)
            return []

    # --------------------------------------------------------------- stats

    async def get_stats(self) -> Dict[str, Any]:
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                active = await conn.fetchval(
                    f"SELECT COUNT(*) FROM {self._t(_MEMORIES_TABLE)} WHERE deleted_at IS NULL"
                )
                tombstoned = await conn.fetchval(
                    f"SELECT COUNT(*) FROM {self._t(_MEMORIES_TABLE)} WHERE deleted_at IS NOT NULL"
                )
                tag_count = await conn.fetchval(
                    f"""
                    SELECT COUNT(DISTINCT t)
                    FROM {self._t(_MEMORIES_TABLE)}, unnest(tags) AS t
                    WHERE deleted_at IS NULL
                    """
                )
            return {
                "storage_backend": "PgVectorMemoryStorage",
                "status": "operational",
                "total_memories": int(active or 0),
                "tombstoned_memories": int(tombstoned or 0),
                "unique_tags": int(tag_count or 0),
                "embedding_model": self.embedding_model_name,
                "embedding_dimension": self.embedding_dimension,
                "schema": self.schema,
            }
        except Exception as e:
            logger.error("get_stats() failed: %s", e)
            return {
                "storage_backend": "PgVectorMemoryStorage",
                "status": "error",
                "error": str(e),
            }
