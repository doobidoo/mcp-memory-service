"""RetrieveMixin: retrieve, search_by_tag, get_all_memories, and related queries."""

import sqlite3
import json
import logging
import os
import time
import traceback
import asyncio
from collections import Counter
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Set

try:
    from sqlite_vec import serialize_float32
except ImportError:
    pass

from ...models.memory import Memory, MemoryQueryResult

logger = logging.getLogger(__name__)

# Module-level constants
_SQLITE_VEC_MAX_KNN_K = 4096
_MAX_TAG_SEARCH_CANDIDATES = _SQLITE_VEC_MAX_KNN_K
_MAX_TAGS_FOR_SEARCH = 100


def _sanitize_log_value(value: object) -> str:
    return str(value).replace("\n", "\\n").replace("\r", "\\r").replace("\x1b", "\\x1b")


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class RetrieveMixin:
    """Mixin providing memory retrieval and search operations."""

    async def retrieve(self, query: str, n_results: int = 5, tags: Optional[List[str]] = None, min_confidence: float = 0.0, include_superseded: bool = False, start_time: Optional[float] = None, end_time: Optional[float] = None, store: Optional[str] = 'default') -> List[MemoryQueryResult]:
        """Retrieve memories using semantic search."""
        try:
            if not self.conn:
                logger.error("Database not initialized")
                return []

            if not self.embedding_model:
                logger.warning("No embedding model available, cannot perform semantic search")
                return []

            try:
                query_embedding = self._generate_embedding(query)
            except Exception as e:
                logger.error("Failed to generate query embedding: %s", _sanitize_log_value(e))
                return []

            def _count_embeddings():
                cursor = self.conn.execute('SELECT COUNT(*) FROM memory_embeddings')
                return cursor.fetchone()[0]
            embedding_count = await self._execute_with_retry(_count_embeddings)

            if embedding_count == 0:
                logger.warning("No embeddings found in database. Memories may have been stored without embeddings.")
                return []

            if tags:
                if len(tags) > _MAX_TAGS_FOR_SEARCH:
                    logger.warning("Too many tags provided for search (%s), limiting to %s", len(tags), _MAX_TAGS_FOR_SEARCH)
                    tags = tags[:_MAX_TAGS_FOR_SEARCH]
                k_value = min(embedding_count, _MAX_TAG_SEARCH_CANDIDATES)
            elif start_time is not None or end_time is not None:
                k_value = min(embedding_count, _MAX_TAG_SEARCH_CANDIDATES)
            else:
                fetch_n = max(n_results * 3, 20) if min_confidence > 0.0 else n_results
                k_value = min(fetch_n, _MAX_TAG_SEARCH_CANDIDATES)

            def search_memories():
                tag_conditions = ""
                params = [serialize_float32(query_embedding), k_value]

                # Add store filter to vec0 query
                store_condition = ""
                if store is not None:
                    store_condition = " AND store = ?"
                    params.append(store)

                if tags:
                    tag_clauses = []
                    for tag in tags:
                        if not isinstance(tag, str):
                            logger.warning("Skipping non-string tag in search: %s", type(tag).__name__)
                            continue
                        stripped = tag.strip()
                        tag_clauses.append(
                            "(',' || REPLACE(m.tags, ' ', '') || ',') LIKE ? ESCAPE '\\'"
                        )
                        params.append(f"%,{_escape_like(stripped.replace(' ', ''))},%")

                    if not tag_clauses:
                        logger.warning("Tag filter provided but contained no valid tags. Returning empty results.")
                        return []

                    tag_conditions = " AND (" + " OR ".join(tag_clauses) + ")"

                superseded_filter = "" if include_superseded else " AND (m.superseded_by IS NULL OR m.superseded_by = '')"

                time_conditions = ""
                if start_time is not None:
                    time_conditions += " AND m.created_at >= ?"
                    params.append(start_time)
                if end_time is not None:
                    time_conditions += " AND m.created_at <= ?"
                    params.append(end_time)

                sql = f'''
                    SELECT m.content_hash, m.content, m.tags, m.memory_type, m.metadata,
                           m.created_at, m.updated_at, m.created_at_iso, m.updated_at_iso,
                           e.distance
                    FROM memories m
                    INNER JOIN (
                        SELECT rowid, distance
                        FROM memory_embeddings
                        WHERE content_embedding MATCH ? AND k = ?{store_condition}
                          AND rowid IN (
                              SELECT m.id FROM memories m
                              WHERE m.deleted_at IS NULL{superseded_filter}{tag_conditions}{time_conditions}
                          )
                    ) e ON m.id = e.rowid
                    ORDER BY e.distance
                    LIMIT ?
                '''
                params.append(n_results)

                cursor = self.conn.execute(sql, params)
                results = cursor.fetchall()
                if not results:
                    logger.debug("No results from vector search. Checking database state...")
                    mem_count = self.conn.execute('SELECT COUNT(*) FROM memories').fetchone()[0]
                    logger.debug("Memories table has %s rows, embeddings table has %s rows", mem_count, embedding_count)

                return results

            search_results = await self._execute_with_retry(search_memories)

            results = []
            for row in search_results:
                try:
                    content_hash, content, tags_str, memory_type, metadata_str = row[:5]
                    created_at, updated_at, created_at_iso, updated_at_iso, distance = row[5:]

                    tags_list = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
                    metadata = self._safe_json_loads(metadata_str, "memory_metadata")

                    memory = Memory(
                        content=content,
                        content_hash=content_hash,
                        tags=tags_list,
                        memory_type=memory_type,
                        metadata=metadata,
                        created_at=created_at,
                        updated_at=updated_at,
                        created_at_iso=created_at_iso,
                        updated_at_iso=updated_at_iso
                    )

                    relevance_score = max(0.0, 1.0 - (float(distance) / 2.0)) if distance is not None else 0.0

                    effective_confidence = self._effective_confidence(
                        memory.metadata.get("confidence"),
                        memory.metadata.get("last_accessed_at"),
                        created_at,
                    )

                    memory.record_access(query)
                    results.append(MemoryQueryResult(
                        memory=memory,
                        relevance_score=relevance_score,
                        debug_info={
                            "distance": distance,
                            "backend": "sqlite-vec",
                            "effective_confidence": effective_confidence,
                        }
                    ))

                except Exception as parse_error:
                    logger.warning("Failed to parse memory result: %s", _sanitize_log_value(parse_error))
                    continue

            try:
                await self._persist_access_metadata_batch([r.memory for r in results])
            except Exception as e:
                logger.warning("Failed to persist access metadata: %s", _sanitize_log_value(e))

            if min_confidence > 0.0:
                before = len(results)
                results = [
                    r for r in results
                    if r.debug_info.get("effective_confidence", 1.0) >= min_confidence
                ][:n_results]
                logger.debug("min_confidence=%s filtered %s stale memories", _sanitize_log_value(min_confidence), before - len(results))

            logger.info(f"Retrieved {len(results)} memories for query: {_sanitize_log_value(query)}")
            return results

        except Exception as e:
            logger.error("Failed to retrieve memories: %s", _sanitize_log_value(e))
            logger.error(traceback.format_exc())
            return []

    async def search_by_tag(self, tags: List[str], time_start: Optional[float] = None) -> List[Memory]:
        """Search memories by tags with optional time filtering."""
        try:
            if not self.conn:
                logger.error("Database not initialized")
                return []

            if not tags:
                return []

            stripped_tags = [tag.strip() for tag in tags]
            tag_conditions = " OR ".join(["(',' || REPLACE(tags, ' ', '') || ',') LIKE ? ESCAPE '\\'" for _ in stripped_tags])
            tag_params = [f"%,{_escape_like(tag)},%" for tag in stripped_tags]

            where_clause = f"WHERE ({tag_conditions}) AND deleted_at IS NULL"
            if time_start is not None:
                where_clause += " AND created_at >= ?"
                tag_params.append(time_start)

            def _search_by_tag(wc=where_clause, tp=tag_params):
                cursor = self.conn.execute(f'''
                    SELECT content_hash, content, tags, memory_type, metadata,
                           created_at, updated_at, created_at_iso, updated_at_iso
                    FROM memories
                    {wc}
                    ORDER BY created_at DESC
                ''', tp)
                return cursor.fetchall()

            rows = await self._execute_with_retry(_search_by_tag)

            results = []
            for row in rows:
                try:
                    content_hash, content, tags_str, memory_type, metadata_str = row[:5]
                    created_at, updated_at, created_at_iso, updated_at_iso = row[5:]

                    memory_tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
                    metadata = self._safe_json_loads(metadata_str, "memory_metadata")

                    memory = Memory(
                        content=content,
                        content_hash=content_hash,
                        tags=memory_tags,
                        memory_type=memory_type,
                        metadata=metadata,
                        created_at=created_at,
                        updated_at=updated_at,
                        created_at_iso=created_at_iso,
                        updated_at_iso=updated_at_iso
                    )

                    results.append(memory)

                except Exception as parse_error:
                    logger.warning("Failed to parse memory result: %s", _sanitize_log_value(parse_error))
                    continue

            logger.info(f"Found {len(results)} memories with tags: {[_sanitize_log_value(t) for t in tags]}")
            return results

        except Exception as e:
            logger.error("Failed to search by tags: %s", _sanitize_log_value(e))
            logger.error(traceback.format_exc())
            return []

    async def search_by_tags(
        self,
        tags: List[str],
        operation: str = "AND",
        time_start: Optional[float] = None,
        time_end: Optional[float] = None
    ) -> List[Memory]:
        """Search memories by tags with AND/OR operation and optional time filtering."""
        try:
            if not self.conn:
                logger.error("Database not initialized")
                return []

            if not tags:
                return []

            normalized_operation = operation.strip().upper() if isinstance(operation, str) else "AND"
            if normalized_operation not in {"AND", "OR"}:
                logger.warning("Unsupported tag operation %s; defaulting to AND", operation)
                normalized_operation = "AND"

            stripped_tags = [tag.strip() for tag in tags]
            comparator = " AND " if normalized_operation == "AND" else " OR "
            tag_conditions = comparator.join(["(',' || REPLACE(tags, ' ', '') || ',') LIKE ? ESCAPE '\\'" for _ in stripped_tags])
            tag_params = [f"%,{_escape_like(tag)},%" for tag in stripped_tags]

            where_conditions = [f"({tag_conditions})"] if tag_conditions else []
            where_conditions.append("deleted_at IS NULL")
            if time_start is not None:
                where_conditions.append("created_at >= ?")
                tag_params.append(time_start)
            if time_end is not None:
                where_conditions.append("created_at <= ?")
                tag_params.append(time_end)

            where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""

            def _search_by_tags(wc=where_clause, tp=tag_params):
                cursor = self.conn.execute(f'''
                    SELECT content_hash, content, tags, memory_type, metadata,
                           created_at, updated_at, created_at_iso, updated_at_iso
                    FROM memories
                    {wc}
                    ORDER BY updated_at DESC
                ''', tp)
                return cursor.fetchall()

            rows = await self._execute_with_retry(_search_by_tags)

            results = []
            for row in rows:
                try:
                    content_hash, content, tags_str, memory_type, metadata_str, created_at, updated_at, created_at_iso, updated_at_iso = row

                    memory_tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
                    metadata = self._safe_json_loads(metadata_str, "memory_metadata")

                    memory = Memory(
                        content=content,
                        content_hash=content_hash,
                        tags=memory_tags,
                        memory_type=memory_type,
                        metadata=metadata,
                        created_at=created_at,
                        updated_at=updated_at,
                        created_at_iso=created_at_iso,
                        updated_at_iso=updated_at_iso
                    )

                    results.append(memory)

                except Exception as parse_error:
                    logger.warning("Failed to parse memory result: %s", _sanitize_log_value(parse_error))
                    continue

            logger.info("Found %s memories with tags: %s (operation: %s)", len(results), _sanitize_log_value(tags), _sanitize_log_value(operation))
            return results

        except Exception as e:
            logger.error("Failed to search by tags with operation %s: %s", _sanitize_log_value(operation), _sanitize_log_value(e))
            logger.error(traceback.format_exc())
            return []

    async def search_by_tag_chronological(self, tags: List[str], limit: int = None, offset: int = 0) -> List[Memory]:
        """Search memories by tags with chronological ordering and database-level pagination."""
        try:
            if not self.conn:
                logger.error("Database not initialized")
                return []

            if not tags:
                return []

            stripped_tags = [tag.strip() for tag in tags]
            tag_conditions = " OR ".join(["(',' || REPLACE(tags, ' ', '') || ',') LIKE ? ESCAPE '\\'" for _ in stripped_tags])
            tag_params = [f"%,{_escape_like(tag)},%" for tag in stripped_tags]

            query = f"""
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, created_at_iso, updated_at_iso
                FROM memories
                WHERE ({tag_conditions})
                AND deleted_at IS NULL
                ORDER BY created_at DESC
            """

            if limit is not None:
                query += " LIMIT ?"
                tag_params.append(int(limit))
            if offset > 0:
                query += " OFFSET ?"
                tag_params.append(int(offset))

            def _search_chronological(q=query, tp=tag_params):
                cursor = self.conn.execute(q, tp)
                return cursor.fetchall()

            rows = await self._execute_with_retry(_search_chronological)
            results = []

            for row in rows:
                try:
                    content_hash, content, tags_str, memory_type, metadata_str, created_at, updated_at, created_at_iso, updated_at_iso = row

                    memory_tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
                    metadata = self._safe_json_loads(metadata_str, "memory_metadata")

                    memory = Memory(
                        content=content,
                        content_hash=content_hash,
                        tags=memory_tags,
                        memory_type=memory_type,
                        metadata=metadata,
                        created_at=created_at,
                        updated_at=updated_at,
                        created_at_iso=created_at_iso,
                        updated_at_iso=updated_at_iso
                    )

                    results.append(memory)

                except Exception as parse_error:
                    logger.warning("Failed to parse memory result: %s", _sanitize_log_value(parse_error))
                    continue

            logger.info("Found %s memories with tags: %s using database-level pagination (limit=%s, offset=%s)", len(results), _sanitize_log_value(tags), limit, offset)
            return results

        except Exception as e:
            logger.error("Failed to search by tags chronologically: %s", _sanitize_log_value(e))
            logger.error(traceback.format_exc())
            return []

    async def get_by_hash(self, content_hash: str) -> Optional[Memory]:
        """Get a memory by its content hash."""
        try:
            if not self.conn:
                return None

            def _get_by_hash():
                cursor = self.conn.execute('''
                    SELECT content_hash, content, tags, memory_type, metadata,
                           created_at, updated_at, created_at_iso, updated_at_iso
                    FROM memories WHERE content_hash = ? AND deleted_at IS NULL
                ''', (content_hash,))
                return cursor.fetchone()

            row = await self._execute_with_retry(_get_by_hash)
            if not row:
                return None

            content_hash_val, content, tags_str, memory_type, metadata_str = row[:5]
            created_at, updated_at, created_at_iso, updated_at_iso = row[5:]

            tags_list = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
            metadata = self._safe_json_loads(metadata_str, "memory_retrieval")

            memory = Memory(
                content=content,
                content_hash=content_hash_val,
                tags=tags_list,
                memory_type=memory_type,
                metadata=metadata,
                created_at=created_at,
                updated_at=updated_at,
                created_at_iso=created_at_iso,
                updated_at_iso=updated_at_iso
            )

            return memory

        except Exception as e:
            logger.error("Failed to get memory by hash %s: %s", _sanitize_log_value(content_hash), _sanitize_log_value(e))
            return None

    async def get_all_content_hashes(self, include_deleted: bool = False) -> Set[str]:
        """Get all content hashes in database for bulk existence checking."""
        try:
            if not self.conn:
                return set()

            def _get_hashes():
                if include_deleted:
                    cursor = self.conn.execute('SELECT content_hash FROM memories')
                else:
                    cursor = self.conn.execute('SELECT content_hash FROM memories WHERE deleted_at IS NULL')
                return cursor.fetchall()

            rows = await self._execute_with_retry(_get_hashes)
            return {row[0] for row in rows}

        except Exception as e:
            logger.error("Failed to get all content hashes: %s", _sanitize_log_value(e))
            return set()

    async def get_by_exact_content(self, content: str) -> List[Memory]:
        """Retrieve memories by exact content match."""
        try:
            if not self.conn:
                return []

            def _get_by_exact_content():
                cursor = self.conn.execute('''
                    SELECT content, tags, memory_type, metadata, content_hash,
                           created_at, created_at_iso, updated_at, updated_at_iso
                    FROM memories
                    WHERE content LIKE '%' || ? || '%' COLLATE NOCASE
                    AND deleted_at IS NULL
                    ORDER BY created_at DESC
                ''', (content,))
                return cursor.fetchall()

            memories = []
            for row in await self._execute_with_retry(_get_by_exact_content):
                content_str, tags_str, memory_type, metadata_str, content_hash, \
                    created_at, created_at_iso, updated_at, updated_at_iso = row

                metadata = self._safe_json_loads(metadata_str, "get_by_exact_content")
                tags_list = [tag.strip() for tag in tags_str.split(',')] if tags_str else []

                memory = Memory(
                    content=content_str,
                    content_hash=content_hash,
                    tags=tags_list,
                    memory_type=memory_type,
                    metadata=metadata,
                    created_at=created_at,
                    created_at_iso=created_at_iso,
                    updated_at=updated_at,
                    updated_at_iso=updated_at_iso
                )
                memories.append(memory)

            return memories

        except Exception as e:
            logger.error("Error in exact content match: %s", _sanitize_log_value(e))
            return []

    async def get_all_memories(
        self,
        limit: int = None,
        offset: int = 0,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        tag_match: str = "any",
        stale_days: Optional[int] = None,
        include_embeddings: bool = False,
        store: Optional[str] = "default",
    ) -> List[Memory]:
        """Get all memories in storage ordered by creation time (newest first)."""
        try:
            await self.initialize()

            query = '''
                SELECT m.content_hash, m.content, m.tags, m.memory_type, m.metadata,
                       m.created_at, m.updated_at, m.created_at_iso, m.updated_at_iso,
                       e.content_embedding
                FROM memories m
                LEFT JOIN memory_embeddings e ON m.id = e.rowid
            '''

            params = []
            where_conditions = []

            where_conditions.append('m.deleted_at IS NULL')

            if store is not None:
                where_conditions.append('m.store = ?')
                params.append(store)

            if memory_type is not None:
                where_conditions.append('m.memory_type = ?')
                params.append(memory_type)

            if tags and len(tags) > 0:
                stripped_tags = [tag.strip() for tag in tags]
                joiner = " AND " if tag_match == "all" else " OR "
                tag_conditions = joiner.join(
                    [
                        "(',' || REPLACE(m.tags, ' ', '') || ',') LIKE ? ESCAPE '\\'"
                        for _ in stripped_tags
                    ]
                )
                where_conditions.append(f"({tag_conditions})")
                params.extend([f"%,{_escape_like(tag)},%" for tag in stripped_tags])

            self._apply_stale_days_filter(where_conditions, params, stale_days, table_alias="m")

            query += ' WHERE ' + ' AND '.join(where_conditions)
            query += ' ORDER BY m.created_at DESC'

            if limit is not None:
                query += ' LIMIT ?'
                params.append(limit)

            if offset > 0:
                query += ' OFFSET ?'
                params.append(offset)

            def _get_all(q=query, p=params):
                cursor = self.conn.execute(q, p)
                return cursor.fetchall()

            memories = []
            for row in await self._execute_with_retry(_get_all):
                memory = self._row_to_memory(row)
                if memory:
                    memories.append(memory)

            return memories

        except Exception as e:
            logger.error("Error getting all memories: %s", _sanitize_log_value(e))
            return []

    async def get_recent_memories(self, n: int = 10) -> List[Memory]:
        """Get n most recent memories."""
        return await self.get_all_memories(limit=n, offset=0)

    async def get_largest_memories(self, n: int = 10) -> List[Memory]:
        """Get n largest memories by content length."""
        try:
            await self.initialize()

            query = """
                SELECT content_hash, content, tags, memory_type, metadata, created_at, updated_at
                FROM memories
                WHERE deleted_at IS NULL
                ORDER BY LENGTH(content) DESC
                LIMIT ?
            """

            def _get_largest(q=query, _n=n):
                cursor = self.conn.execute(q, (_n,))
                return cursor.fetchall()

            rows = await self._execute_with_retry(_get_largest)
            memories = []
            for row in rows:
                try:
                    memory = Memory(
                        content_hash=row[0],
                        content=row[1],
                        tags=[t.strip() for t in row[2].split(",") if t.strip()] if row[2] else [],
                        memory_type=row[3],
                        metadata=self._safe_json_loads(row[4], "get_largest_memories"),
                        created_at=row[5],
                        updated_at=row[6]
                    )
                    memories.append(memory)
                except Exception as parse_error:
                    logger.warning("Failed to parse memory %s: %s", row[0], _sanitize_log_value(parse_error))
                    continue

            return memories

        except Exception as e:
            logger.error("Error getting largest memories: %s", _sanitize_log_value(e))
            return []

    async def get_memory_timestamps(self, days: Optional[int] = None) -> List[float]:
        """Get memory creation timestamps only, without loading full memory objects."""
        try:
            await self.initialize()

            def _get_timestamps():
                if days is not None:
                    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
                    cutoff_timestamp = cutoff.timestamp()
                    cursor = self.conn.execute(
                        """
                        SELECT created_at
                        FROM memories
                        WHERE created_at >= ? AND deleted_at IS NULL
                        ORDER BY created_at DESC
                        """,
                        (cutoff_timestamp,)
                    )
                else:
                    cursor = self.conn.execute(
                        """
                        SELECT created_at
                        FROM memories
                        WHERE deleted_at IS NULL
                        ORDER BY created_at DESC
                        """
                    )
                return cursor.fetchall()

            rows = await self._execute_with_retry(_get_timestamps)
            timestamps = [row[0] for row in rows if row[0] is not None]

            return timestamps

        except Exception as e:
            logger.error("Error getting memory timestamps: %s", _sanitize_log_value(e))
            return []

    async def count_all_memories(self, memory_type: Optional[str] = None, tags: Optional[List[str]] = None, tag_match: str = "any", stale_days: Optional[int] = None, store: Optional[str] = "default") -> int:
        """Get total count of memories in storage."""
        try:
            await self.initialize()

            conditions = []
            params = []

            if store is not None:
                conditions.append('store = ?')
                params.append(store)

            if memory_type is not None:
                conditions.append('memory_type = ?')
                params.append(memory_type)

            if tags:
                stripped_tags = [tag.strip() for tag in tags]
                joiner = " AND " if tag_match == "all" else " OR "
                tag_conditions = joiner.join(
                    [
                        "(',' || REPLACE(tags, ' ', '') || ',') LIKE ? ESCAPE '\\'"
                        for _ in stripped_tags
                    ]
                )
                conditions.append(f"({tag_conditions})")
                params.extend([f"%,{_escape_like(tag)},%" for tag in stripped_tags])

            self._apply_stale_days_filter(conditions, params, stale_days)

            conditions.append('deleted_at IS NULL')
            count_query = 'SELECT COUNT(*) FROM memories WHERE ' + ' AND '.join(conditions)
            count_params = tuple(params)

            def _count():
                cursor = self.conn.execute(count_query, count_params)
                result = cursor.fetchone()
                return result[0] if result else 0

            return await self._execute_with_retry(_count)

        except Exception as e:
            logger.error("Error counting memories: %s", _sanitize_log_value(e))
            return 0

    async def get_all_tags_with_counts(self) -> List[Dict[str, Any]]:
        """Get all tags with their usage counts."""
        try:
            await self.initialize()

            def _get_tags():
                cursor = self.conn.execute('''
                    SELECT tags
                    FROM memories
                    WHERE tags IS NOT NULL AND tags != '' AND deleted_at IS NULL
                ''')
                return cursor.fetchall()

            rows = await self._execute_with_retry(_get_tags)

            await asyncio.sleep(0)

            tag_counter = Counter(
                tag.strip()
                for (tag_string,) in rows
                if tag_string
                for tag in tag_string.split(",")
                if tag.strip()
            )

            return [{"tag": tag, "count": count} for tag, count in tag_counter.most_common()]

        except sqlite3.Error as e:
            logger.error("Database error getting tags with counts: %s", _sanitize_log_value(e))
            return []
        except Exception as e:
            logger.error("Unexpected error getting tags with counts: %s", _sanitize_log_value(e))
            raise

    async def get_relationship_type_distribution(self) -> Dict[str, int]:
        """Get distribution of relationship types in the knowledge graph."""
        try:
            if not self.conn:
                logger.error("Database not initialized")
                return {}

            def _get_rel_distribution():
                cursor = self.conn.execute("""
                    SELECT
                        CASE
                            WHEN relationship_type IS NULL OR relationship_type = '' THEN 'untyped'
                            ELSE relationship_type
                        END as rel_type,
                        COUNT(*) as count
                    FROM memory_graph
                    GROUP BY rel_type
                    ORDER BY count DESC
                """)
                return cursor.fetchall()

            rows = await self._execute_with_retry(_get_rel_distribution)
            distribution = {row[0]: row[1] for row in rows}
            return distribution

        except sqlite3.Error as e:
            logger.error("Database error getting relationship type distribution: %s", _sanitize_log_value(e))
            return {}
        except Exception as e:
            logger.error("Unexpected error getting relationship type distribution: %s", _sanitize_log_value(e))
            return {}

    async def get_graph_visualization_data(
        self,
        limit: int = 100,
        min_connections: int = 1
    ) -> Dict[str, Any]:
        """Get graph data for visualization in D3.js-compatible format."""
        try:
            if not self.conn:
                logger.error("Database not initialized")
                return {"nodes": [], "edges": []}

            def _get_graph_nodes():
                cursor = self.conn.execute("""
                    SELECT
                        m.content_hash,
                        m.content,
                        m.memory_type,
                        m.created_at,
                        m.updated_at,
                        m.tags,
                        m.metadata,
                        COUNT(DISTINCT mg.target_hash) as connection_count
                    FROM memories m
                    INNER JOIN memory_graph mg ON m.content_hash = mg.source_hash
                    WHERE m.deleted_at IS NULL
                      -- has_entity rows point at an entity NAME, not a memory
                      -- hash, so the edge loop below always drops them. Counting
                      -- them here selected memories whose connections can never
                      -- render, filling the view with isolated dots (Issue #256).
                      -- COALESCE, not a bare !=: relationship_type has no NOT NULL
                      -- constraint, and `NULL != 'has_entity'` is NULL in SQL, which
                      -- would silently drop those rows instead of keeping them. The
                      -- edge loop already reads NULL as 'related'.
                      AND COALESCE(mg.relationship_type, 'related') != 'has_entity'
                    GROUP BY m.content_hash
                    HAVING connection_count >= ?
                    ORDER BY connection_count DESC
                    LIMIT ?
                """, (min_connections, limit))
                return cursor.fetchall()

            nodes = []
            node_hashes = set()

            for row in await self._execute_with_retry(_get_graph_nodes):
                content_hash, content, memory_type, created_at, updated_at, tags_str, metadata_str, connection_count = row

                tags_list = []
                if tags_str:
                    tags_list = [tag.strip() for tag in tags_str.split(",") if tag.strip()]

                metadata = self._safe_json_loads(metadata_str, "get_graph_visualization_data")

                nodes.append({
                    "id": content_hash,
                    "type": memory_type or "untyped",
                    "content": content[:100] if content else "",
                    "connections": connection_count,
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "quality_score": metadata.get("quality_score", 0.5),
                    "tags": tags_list
                })
                node_hashes.add(content_hash)

            if node_hashes:
                # Bind only the source side (one placeholder per node) and filter
                # the target in Python. This keeps the bound-variable count at N
                # instead of 2*N, so large node limits stay well under SQLite's
                # variable cap; the source_hash index still drives the lookup.
                placeholders = ",".join("?" * len(node_hashes))
                query = f"""
                    SELECT
                        source_hash,
                        target_hash,
                        relationship_type,
                        similarity,
                        connection_types
                    FROM memory_graph
                    WHERE source_hash IN ({placeholders})
                """

                def _get_graph_edges(q=query, nh=node_hashes):
                    cursor = self.conn.execute(q, list(nh))
                    return cursor.fetchall()

                edges = []
                for row in await self._execute_with_retry(_get_graph_edges):
                    source, target, rel_type, similarity, conn_types = row

                    # Keep only edges whose target is also in the rendered node set
                    if target not in node_hashes:
                        continue

                    edges.append({
                        "source": source,
                        "target": target,
                        "relationship_type": rel_type or "related",
                        "similarity": similarity if similarity is not None else 0.5,
                        "connection_types": conn_types or ""
                    })

            else:
                edges = []

            return {
                "nodes": nodes,
                "edges": edges,
                "meta": {
                    "total_nodes": len(nodes),
                    "total_edges": len(edges),
                    "min_connections": min_connections,
                    "limit": limit
                }
            }

        except sqlite3.Error as e:
            logger.error("Database error getting graph visualization data: %s", _sanitize_log_value(e))
            return {"nodes": [], "edges": []}
        except Exception as e:
            logger.error("Unexpected error getting graph visualization data: %s", _sanitize_log_value(e))
            return {"nodes": [], "edges": []}

    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            if not self.conn:
                return {"error": "Database not initialized"}

            def _get_stats():
                week_ago = time.time() - (7 * 24 * 60 * 60)
                total = self.conn.execute(
                    'SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL'
                ).fetchone()[0]
                tag_rows = self.conn.execute(
                    'SELECT tags FROM memories WHERE tags IS NOT NULL AND tags != "" AND deleted_at IS NULL'
                ).fetchall()
                this_week = self.conn.execute(
                    'SELECT COUNT(*) FROM memories WHERE created_at >= ? AND deleted_at IS NULL',
                    (week_ago,)
                ).fetchone()[0]
                return total, tag_rows, this_week

            total_memories, tag_rows, memories_this_week = await self._execute_with_retry(_get_stats)

            unique_tags = len(set(
                tag.strip()
                for (tag_string,) in tag_rows
                if tag_string
                for tag in tag_string.split(",")
                if tag.strip()
            ))

            file_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

            return {
                "backend": "sqlite-vec",
                "total_memories": total_memories,
                "unique_tags": unique_tags,
                "memories_this_week": memories_this_week,
                "database_size_bytes": file_size,
                "database_size_mb": round(file_size / (1024 * 1024), 2),
                "embedding_model": self.embedding_model_name,
                "embedding_dimension": self.embedding_dimension
            }

        except sqlite3.Error as e:
            logger.error("Database error getting stats: %s", _sanitize_log_value(e))
            return {"error": f"Database error: {str(e)}"}
        except OSError as e:
            logger.error("File system error getting stats: %s", _sanitize_log_value(e))
            return {"error": f"File system error: {str(e)}"}
        except Exception as e:
            logger.error("Unexpected error getting stats: %s", _sanitize_log_value(e))
            return {"error": f"Unexpected error: {str(e)}"}

    def sanitized(self, tags):
        """Sanitize and normalize tags to a JSON string."""
        if tags is None:
            return json.dumps([])

        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
        elif isinstance(tags, list):
            tags = [str(tag).strip() for tag in tags if str(tag).strip()]
        else:
            return json.dumps([])

        return json.dumps(tags)

    async def recall(self, query: Optional[str] = None, n_results: int = 5, start_timestamp: Optional[float] = None, end_timestamp: Optional[float] = None) -> List[MemoryQueryResult]:
        """Retrieve memories with combined time filtering and optional semantic search."""
        try:
            if not self.conn:
                logger.error("Database not initialized, cannot retrieve memories")
                return []

            time_conditions = []
            params = []

            if start_timestamp is not None:
                time_conditions.append("created_at >= ?")
                params.append(float(start_timestamp))

            if end_timestamp is not None:
                time_conditions.append("created_at <= ?")
                params.append(float(end_timestamp))

            time_where = " AND ".join(time_conditions) if time_conditions else ""

            logger.info("Time filtering conditions: %s, params: %s", _sanitize_log_value(time_where), _sanitize_log_value(params))

            if query and self.embedding_model:
                try:
                    query_embedding = self._generate_embedding(query)

                    # Filter eligible rows inside KNN so excluded neighbors do not consume k.
                    memory_filter = "deleted_at IS NULL"
                    if time_where:
                        memory_filter += f" AND {time_where}"

                    base_query = f'''
                        SELECT m.content_hash, m.content, m.tags, m.memory_type, m.metadata,
                               m.created_at, m.updated_at, m.created_at_iso, m.updated_at_iso,
                               e.distance
                        FROM memories m
                        JOIN (
                            SELECT rowid, distance
                            FROM memory_embeddings
                            WHERE content_embedding MATCH ? AND k = ?
                              AND rowid IN (SELECT id FROM memories WHERE {memory_filter})
                        ) e ON m.id = e.rowid
                    '''

                    base_query += " ORDER BY e.distance"

                    query_params = [serialize_float32(query_embedding), n_results] + params

                    def _recall_semantic(bq=base_query, qp=query_params):
                        cursor = self.conn.execute(bq, qp)
                        return cursor.fetchall()

                    rows = await self._execute_with_retry(_recall_semantic)
                    results = []
                    for row in rows:
                        try:
                            content_hash, content, tags_str, memory_type, metadata_str = row[:5]
                            created_at, updated_at, created_at_iso, updated_at_iso, distance = row[5:]

                            tags_list = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
                            metadata = self._safe_json_loads(metadata_str, "memory_metadata")

                            memory = Memory(
                                content=content,
                                content_hash=content_hash,
                                tags=tags_list,
                                memory_type=memory_type,
                                metadata=metadata,
                                created_at=created_at,
                                updated_at=updated_at,
                                created_at_iso=created_at_iso,
                                updated_at_iso=updated_at_iso
                            )

                            relevance_score = (
                                max(0.0, 1.0 - (float(distance) / 2.0))
                                if distance is not None
                                else 0.0
                            )

                            results.append(MemoryQueryResult(
                                memory=memory,
                                relevance_score=relevance_score,
                                debug_info={
                                    "distance": distance,
                                    "backend": "sqlite-vec",
                                    "time_filtered": bool(time_where),
                                }
                            ))

                        except Exception as parse_error:
                            logger.warning("Failed to parse memory result: %s", _sanitize_log_value(parse_error))
                            continue

                    logger.info("Retrieved %s memories for semantic query with time filter", len(results))
                    return results

                except Exception as query_error:
                    logger.error("Error in semantic search with time filter: %s", _sanitize_log_value(query_error))
                    logger.info("Falling back to time-based retrieval")

            base_query = '''
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, created_at_iso, updated_at_iso
                FROM memories
            '''

            if time_where:
                base_query += f" WHERE deleted_at IS NULL AND {time_where}"
            else:
                base_query += " WHERE deleted_at IS NULL"

            base_query += " ORDER BY created_at DESC LIMIT ?"
            params.append(n_results)

            def _recall_timebased(bq=base_query, p=params):
                cursor = self.conn.execute(bq, p)
                return cursor.fetchall()

            time_rows = await self._execute_with_retry(_recall_timebased)

            results = []
            for row in time_rows:
                try:
                    content_hash, content, tags_str, memory_type, metadata_str = row[:5]
                    created_at, updated_at, created_at_iso, updated_at_iso = row[5:]

                    tags_list = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
                    metadata = self._safe_json_loads(metadata_str, "memory_metadata")

                    memory = Memory(
                        content=content,
                        content_hash=content_hash,
                        tags=tags_list,
                        memory_type=memory_type,
                        metadata=metadata,
                        created_at=created_at,
                        updated_at=updated_at,
                        created_at_iso=created_at_iso,
                        updated_at_iso=updated_at_iso
                    )

                    results.append(MemoryQueryResult(
                        memory=memory,
                        relevance_score=None,
                        debug_info={"backend": "sqlite-vec", "time_filtered": bool(time_where), "query_type": "time_based"}
                    ))

                except Exception as parse_error:
                    logger.warning("Failed to parse memory result: %s", _sanitize_log_value(parse_error))
                    continue

            logger.info("Retrieved %s memories for time-based query", len(results))
            return results

        except Exception as e:
            logger.error("Error in recall: %s", _sanitize_log_value(e))
            logger.error(traceback.format_exc())
            return []

    async def get_memories_by_time_range(
        self,
        start_time: float,
        end_time: float,
        include_embeddings: bool = False,
    ) -> List[Memory]:
        """Get memories within a specific time range."""
        try:
            await self.initialize()

            if include_embeddings:
                sql = '''
                    SELECT m.content_hash, m.content, m.tags, m.memory_type, m.metadata,
                           m.created_at, m.updated_at, m.created_at_iso, m.updated_at_iso,
                           e.content_embedding
                    FROM memories m
                    LEFT JOIN memory_embeddings e ON m.id = e.rowid
                    WHERE m.created_at BETWEEN ? AND ? AND m.deleted_at IS NULL
                    ORDER BY m.created_at DESC
                '''
            else:
                sql = '''
                    SELECT content_hash, content, tags, memory_type, metadata,
                           created_at, updated_at, created_at_iso, updated_at_iso
                    FROM memories
                    WHERE created_at BETWEEN ? AND ? AND deleted_at IS NULL
                    ORDER BY created_at DESC
                '''

            def _get_by_time_range():
                cursor = self.conn.execute(sql, (start_time, end_time))
                return cursor.fetchall()

            results = []
            for row in await self._execute_with_retry(_get_by_time_range):
                memory = self._row_to_memory(row)
                if memory is not None:
                    results.append(memory)

            logger.info("Retrieved %s memories in time range %s-%s", len(results), _sanitize_log_value(start_time), _sanitize_log_value(end_time))
            return results

        except Exception as e:
            logger.error("Error getting memories by time range: %s", _sanitize_log_value(e))
            return []

    async def get_memory_connections(self) -> Dict[str, int]:
        """Get memory connection statistics."""
        try:
            await self.initialize()

            def _get_connections():
                cursor = self.conn.execute("""
                    SELECT tags, COUNT(*) as count
                    FROM memories
                    WHERE tags IS NOT NULL AND tags != '' AND deleted_at IS NULL
                    GROUP BY tags
                """)
                return cursor.fetchall()

            connections = {}
            for row in await self._execute_with_retry(_get_connections):
                tags_str, count = row
                if tags_str:
                    tags_list = [tag.strip() for tag in tags_str.split(",") if tag.strip()]
                    for tag in tags_list:
                        connections[f"tag:{tag}"] = connections.get(f"tag:{tag}", 0) + count

            return connections

        except Exception as e:
            logger.error("Error getting memory connections: %s", _sanitize_log_value(e))
            return {}

    async def get_access_patterns(self) -> Dict[str, datetime]:
        """Get memory access pattern statistics."""
        try:
            await self.initialize()

            def _get_access_patterns():
                cursor = self.conn.execute("""
                    SELECT content_hash, updated_at_iso
                    FROM memories
                    WHERE updated_at_iso IS NOT NULL AND deleted_at IS NULL
                    ORDER BY updated_at DESC
                    LIMIT 100
                """)
                return cursor.fetchall()

            patterns = {}
            for row in await self._execute_with_retry(_get_access_patterns):
                content_hash, updated_at_iso = row
                try:
                    patterns[content_hash] = datetime.fromisoformat(updated_at_iso.replace('Z', '+00:00'))
                except Exception:
                    patterns[content_hash] = datetime.now()

            return patterns

        except Exception as e:
            logger.error("Error getting access patterns: %s", _sanitize_log_value(e))
            return {}
