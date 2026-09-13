"""MetadataMixin: update_memory_metadata, batch updates, conflicts, versioning, staleness."""

import json
import logging
import os
import time
import traceback
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Set

try:
    from sqlite_vec import serialize_float32
except ImportError:
    pass

from ...models.memory import Memory, MemoryQueryResult
from ...utils.hashing import generate_content_hash

logger = logging.getLogger(__name__)


class MetadataMixin:
    """Mixin providing metadata updates, conflict detection, versioning, and staleness."""

    async def _persist_access_metadata(self, memory: Memory):
        """Persist access tracking metadata (access_count, last_accessed_at) to storage."""
        def update_metadata():
            self.conn.execute('''
                UPDATE memories
                SET metadata = ?
                WHERE content_hash = ?
            ''', (json.dumps(memory.metadata), memory.content_hash))
            self.conn.commit()

        await self._execute_with_retry(update_metadata)

    async def _persist_access_metadata_batch(self, memories: List[Memory]):
        """Batch-persist access metadata for multiple memories in one transaction."""
        if not memories:
            return

        def batch_update():
            # issue #1239: persist BOTH the metadata JSON and the dedicated
            # last_accessed column, so staleness/decay measure disuse (not age).
            # The column is what _apply_stale_days_filter reads.
            rows = []
            for m in memories:
                la = m.metadata.get("last_accessed_at")
                last_accessed = int(la) if la is not None else None
                rows.append((json.dumps(m.metadata), last_accessed, m.content_hash))
            self.conn.executemany(
                "UPDATE memories SET metadata = ?, last_accessed = ? "
                "WHERE content_hash = ? AND deleted_at IS NULL",
                rows,
            )
            self.conn.commit()

        await self._execute_with_retry(batch_update)

    async def update_memory_metadata(self, content_hash: str, updates: Dict[str, Any], preserve_timestamps: bool = True) -> Tuple[bool, str]:
        """Update memory metadata without recreating the entire memory entry."""
        try:
            if not self.conn:
                return False, "Database not initialized"

            def _read_current():
                cursor = self.conn.execute(
                    """
                    SELECT content, tags, memory_type, metadata, created_at, created_at_iso,
                           updated_at, updated_at_iso
                    FROM memories WHERE content_hash = ? AND deleted_at IS NULL
                """,
                    (content_hash,),
                )
                return cursor.fetchone()

            row = await self._execute_with_retry(_read_current)
            if not row:
                return False, f"Memory with hash {content_hash} not found"

            content, current_tags, current_type, current_metadata_str, created_at, created_at_iso, current_updated_at, current_updated_at_iso = row

            current_metadata = self._safe_json_loads(current_metadata_str, "update_memory_metadata")

            new_tags = current_tags
            new_type = current_type
            new_metadata = current_metadata.copy()

            if "tags" in updates:
                if isinstance(updates["tags"], list):
                    new_tags = ",".join(updates["tags"])
                else:
                    return False, "Tags must be provided as a list of strings"

            if "memory_type" in updates:
                new_type = updates["memory_type"]

            if "metadata" in updates:
                if isinstance(updates["metadata"], dict):
                    new_metadata.update(updates["metadata"])
                else:
                    return False, "Metadata must be provided as a dictionary"

            protected_fields = {
                "content", "content_hash", "tags", "memory_type", "metadata",
                "embedding", "created_at", "created_at_iso", "updated_at", "updated_at_iso"
            }

            for key, value in updates.items():
                if key not in protected_fields:
                    new_metadata[key] = value

            now = time.time()
            now_iso = datetime.utcfromtimestamp(now).isoformat() + "Z"

            structural_change = any(k in updates for k in ("tags", "memory_type", "content"))
            if preserve_timestamps and not structural_change:
                updated_at = current_updated_at if current_updated_at else now
                updated_at_iso = current_updated_at_iso if current_updated_at_iso else now_iso
            elif not preserve_timestamps:
                created_at = updates.get('created_at', created_at)
                created_at_iso = updates.get('created_at_iso', created_at_iso)
                updated_at = updates.get('updated_at', now)
                updated_at_iso = updates.get('updated_at_iso', now_iso)
            else:
                updated_at = now
                updated_at_iso = now_iso

            def _do_update():
                self.conn.execute(
                    """
                    UPDATE memories SET
                        tags = ?, memory_type = ?, metadata = ?,
                        updated_at = ?, updated_at_iso = ?,
                        created_at = ?, created_at_iso = ?
                    WHERE content_hash = ? AND deleted_at IS NULL
                """,
                    (
                        new_tags,
                        new_type,
                        json.dumps(new_metadata),
                        updated_at,
                        updated_at_iso,
                        created_at,
                        created_at_iso,
                        content_hash,
                    ),
                )
                self.conn.commit()

            await self._execute_with_retry(_do_update)

            updated_fields = []
            if "tags" in updates:
                updated_fields.append("tags")
            if "memory_type" in updates:
                updated_fields.append("memory_type")
            if "metadata" in updates:
                updated_fields.append("custom_metadata")

            for key in updates.keys():
                if key not in protected_fields and key not in ["tags", "memory_type", "metadata"]:
                    updated_fields.append(key)

            updated_fields.append("updated_at")

            summary = f"Updated fields: {', '.join(updated_fields)}"
            logger.info(f"Successfully updated metadata for memory {content_hash}")
            return True, summary

        except Exception as e:
            error_msg = f"Error updating memory metadata: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return False, error_msg

    async def update_memories_batch(self, memories: List[Memory], preserve_timestamps: bool = False) -> List[bool]:
        """Update multiple memories in a single database transaction."""
        if not memories:
            return []

        try:
            if not self.conn:
                return [False] * len(memories)

            results = [False] * len(memories)
            now = time.time()
            now_iso = datetime.utcfromtimestamp(now).isoformat() + "Z"

            def _batch_update():
                cursor = self.conn.cursor()
                for idx, memory in enumerate(memories):
                    try:
                        cursor.execute(
                            """
                            SELECT content, tags, memory_type, metadata, created_at, created_at_iso,
                                   updated_at, updated_at_iso
                            FROM memories WHERE content_hash = ? AND deleted_at IS NULL
                        """,
                            (memory.content_hash,),
                        )

                        row = cursor.fetchone()
                        if not row:
                            logger.warning(f"Memory {memory.content_hash} not found during batch update")
                            continue

                        (content, current_tags, current_type, current_metadata_str,
                         created_at, created_at_iso, current_updated_at, current_updated_at_iso) = row

                        current_metadata = self._safe_json_loads(current_metadata_str, "update_memories_batch")

                        if memory.metadata:
                            merged_metadata = current_metadata.copy()
                            merged_metadata.update(memory.metadata)
                        else:
                            merged_metadata = current_metadata

                        new_tags = ",".join(memory.tags) if memory.tags else current_tags
                        new_type = memory.memory_type if memory.memory_type else current_type

                        structural_change = (
                            new_tags != current_tags or
                            new_type != current_type
                        )
                        if preserve_timestamps and not structural_change:
                            mem_updated_at = current_updated_at if current_updated_at else now
                            mem_updated_at_iso = current_updated_at_iso if current_updated_at_iso else now_iso
                        else:
                            mem_updated_at = now
                            mem_updated_at_iso = now_iso

                        cursor.execute(
                            """
                            UPDATE memories SET
                                tags = ?, memory_type = ?, metadata = ?,
                                updated_at = ?, updated_at_iso = ?
                            WHERE content_hash = ? AND deleted_at IS NULL
                        """,
                            (
                                new_tags,
                                new_type,
                                json.dumps(merged_metadata),
                                mem_updated_at,
                                mem_updated_at_iso,
                                memory.content_hash,
                            ),
                        )

                        results[idx] = True

                    except Exception as e:
                        logger.warning(f"Failed to update memory {memory.content_hash} in batch: {e}")
                        continue

                self.conn.commit()

            await self._execute_with_retry(_batch_update)

            success_count = sum(results)
            logger.info(f"Batch update completed: {success_count}/{len(memories)} memories updated successfully")

            return results

        except Exception as e:
            if self.conn:
                self.conn.rollback()
            logger.error(f"Batch update failed: {e}")
            logger.error(traceback.format_exc())
            return [False] * len(memories)

    async def mark_superseded_batch(self, pairs: list[tuple[str, str]]) -> int:
        """Mark memories as superseded in a single transaction."""
        if not pairs or not self.conn:
            return 0

        def _batch_mark():
            self.conn.executemany(
                "UPDATE memories SET superseded_by = ? WHERE content_hash = ? AND deleted_at IS NULL",
                [(winner, loser) for winner, loser in pairs],
            )
            self.conn.commit()
            return len(pairs)

        try:
            return await self._execute_with_retry(_batch_mark)
        except Exception as e:
            logger.error(f"mark_superseded_batch failed: {e}")
            if self.conn:
                self.conn.rollback()
            return 0

    @staticmethod
    def _effective_confidence(
        confidence: Optional[float],
        last_accessed: Optional[float],
        created_at: Optional[float] = None,
        now: Optional[float] = None,
    ) -> float:
        """Compute time-decayed confidence score."""
        decay_window = float(os.environ.get("MEMORY_DECAY_WINDOW_DAYS", "0"))
        if decay_window <= 0:
            return confidence if confidence is not None else 1.0
        decay_rate = 0.5
        ts_now = now or time.time()
        reference = last_accessed or created_at or ts_now
        days_since = (ts_now - reference) / 86400.0
        staleness = days_since / decay_window
        decay = max(0.0, 1.0 - staleness * decay_rate)
        return round((confidence if confidence is not None else 1.0) * decay, 4)

    def _detect_conflicts(self, new_hash: str, new_content: str, embedding) -> list:
        """Detect conflicting active memories for a newly stored memory."""
        from difflib import SequenceMatcher

        SIMILARITY_THRESHOLD = 0.95
        DIVERGENCE_THRESHOLD = 0.20

        if not self.conn or embedding is None:
            return []

        try:
            cursor = self.conn.execute(
                """SELECT m.content_hash, m.content,
                          vec_distance_cosine(me.content_embedding, ?) as distance
                   FROM memories m
                   JOIN memory_embeddings me ON m.rowid = me.rowid
                   WHERE m.deleted_at IS NULL
                     AND (m.superseded_by IS NULL OR m.superseded_by = '')
                     AND m.content_hash != ?
                   ORDER BY distance ASC
                   LIMIT 5""",
                (serialize_float32(embedding), new_hash),
            )
            candidates = cursor.fetchall()
        except Exception as e:
            logger.warning(f"Conflict detection query failed: {e}")
            return []

        conflicts = []
        for cand_hash, cand_content, distance in candidates:
            similarity = max(0.0, 1.0 - float(distance))
            if similarity < SIMILARITY_THRESHOLD:
                continue

            ratio = SequenceMatcher(None, new_content.lower(), cand_content.lower()).ratio()
            divergence = 1.0 - ratio
            if divergence < DIVERGENCE_THRESHOLD:
                continue

            conflicts.append({
                "existing_hash": cand_hash,
                "existing_content": cand_content,
                "similarity": round(similarity, 4),
                "divergence": round(divergence, 4),
            })

        return conflicts

    async def _record_conflicts(self, new_hash: str, conflicts: list) -> None:
        """Tag conflicting memories and create graph edges."""
        import json as _json

        def _record_all_conflicts():
            for c in conflicts:
                existing_hash = c["existing_hash"]
                metadata = _json.dumps({
                    "similarity": c["similarity"],
                    "divergence": c["divergence"],
                    "detected_at": time.time(),
                })

                for h in (new_hash, existing_hash):
                    cursor = self.conn.execute(
                        "SELECT tags FROM memories WHERE content_hash = ?", (h,)
                    )
                    row = cursor.fetchone()
                    if row:
                        tags = row[0] or ""
                        if "conflict:unresolved" not in tags:
                            new_tags = f"{tags},conflict:unresolved" if tags else "conflict:unresolved"
                            self.conn.execute(
                                "UPDATE memories SET tags = ? WHERE content_hash = ? AND deleted_at IS NULL",
                                (new_tags, h),
                            )

                connection_types_json = _json.dumps(["semantic"])
                now = time.time()
                for src, tgt in ((new_hash, existing_hash), (existing_hash, new_hash)):
                    self.conn.execute(
                        """INSERT OR REPLACE INTO memory_graph
                           (source_hash, target_hash, similarity, connection_types,
                            metadata, created_at, relationship_type)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (src, tgt, c["similarity"], connection_types_json,
                         metadata, now, "contradicts"),
                    )

            self.conn.commit()

        await self._execute_with_retry(_record_all_conflicts)
        logger.info(f"Recorded {len(conflicts)} conflict(s) for {new_hash[:8]}")

    async def get_conflicts(self) -> list:
        """Return all unresolved conflict pairs."""
        if not self.conn:
            return []

        try:
            def _get_conflicts():
                cursor = self.conn.execute(
                    """SELECT g.source_hash, g.target_hash, g.similarity, g.metadata,
                              m1.content AS content_a, m2.content AS content_b
                       FROM memory_graph g
                       JOIN memories m1 ON m1.content_hash = g.source_hash
                       JOIN memories m2 ON m2.content_hash = g.target_hash
                       WHERE g.relationship_type = 'contradicts'
                       AND m1.deleted_at IS NULL AND (m1.superseded_by IS NULL OR m1.superseded_by = '')
                       AND m2.deleted_at IS NULL AND (m2.superseded_by IS NULL OR m2.superseded_by = '')
                       AND g.source_hash < g.target_hash"""
                )
                return cursor.fetchall()

            results = []
            for row in await self._execute_with_retry(_get_conflicts):
                meta = self._safe_json_loads(row[3], "get_conflicts") if row[3] else {}
                results.append({
                    "hash_a": row[0],
                    "hash_b": row[1],
                    "content_a": row[4],
                    "content_b": row[5],
                    "similarity": row[2],
                    "divergence": meta.get("divergence"),
                    "detected_at": meta.get("detected_at"),
                })
            return results
        except Exception as e:
            logger.error(f"get_conflicts error: {e}")
            return []

    async def resolve_conflict(self, winner_hash: str, loser_hash: str) -> Tuple[bool, str]:
        """Resolve a conflict: supersede loser, boost winner confidence."""
        try:
            if not self.conn:
                return False, "Database not initialized"

            def _check_both_exist():
                for h, label in ((winner_hash, "Winner"), (loser_hash, "Loser")):
                    cursor = self.conn.execute(
                        "SELECT content_hash FROM memories WHERE content_hash = ? AND deleted_at IS NULL",
                        (h,),
                    )
                    if not cursor.fetchone():
                        return label, h
                return None

            missing = await self._execute_with_retry(_check_both_exist)
            if missing:
                label, h = missing
                return False, f"{label} memory {h} not found or deleted"

            now = time.time()

            def _do_resolve():
                self.conn.execute(
                    "UPDATE memories SET superseded_by = ? WHERE content_hash = ? AND deleted_at IS NULL",
                    (winner_hash, loser_hash),
                )

                self.conn.execute(
                    "UPDATE memories SET confidence = 1.0, last_accessed = ? WHERE content_hash = ? AND deleted_at IS NULL",
                    (int(now), winner_hash),
                )

                for h in (winner_hash, loser_hash):
                    cursor = self.conn.execute(
                        "SELECT tags FROM memories WHERE content_hash = ?", (h,)
                    )
                    row = cursor.fetchone()
                    if row and row[0]:
                        tags = [t.strip() for t in row[0].split(",") if t.strip() != "conflict:unresolved"]
                        self.conn.execute(
                            "UPDATE memories SET tags = ? WHERE content_hash = ? AND deleted_at IS NULL",
                            (",".join(tags), h),
                        )

                self.conn.commit()

            await self._execute_with_retry(_do_resolve)
            logger.info(f"Conflict resolved: {winner_hash[:8]} wins over {loser_hash[:8]}")
            return True, f"Conflict resolved: {winner_hash[:8]} supersedes {loser_hash[:8]}"

        except Exception as e:
            logger.error(f"resolve_conflict error: {e}")
            return False, str(e)

    async def retrieve_with_staleness(
        self,
        query: str,
        n_results: int = 5,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.0,
    ) -> List[MemoryQueryResult]:
        """Semantic search with staleness-aware confidence scoring."""
        fetch_n = max(n_results * 3, 20) if min_confidence > 0.0 else n_results
        raw = await self.retrieve(query, fetch_n, tags)
        if not raw:
            return raw

        hashes = [r.memory.content_hash for r in raw]
        placeholders = ",".join("?" * len(hashes))

        def _fetch_staleness_meta(ph=placeholders, h=hashes):
            cursor = self.conn.execute(
                f"SELECT content_hash, confidence, last_accessed, created_at "
                f"FROM memories WHERE content_hash IN ({ph})",
                h,
            )
            return cursor.fetchall()

        meta = {row[0]: row[1:] for row in await self._execute_with_retry(_fetch_staleness_meta)}

        now = time.time()
        enriched: List[MemoryQueryResult] = []
        hashes_to_touch: List[str] = []

        for result in raw:
            ch = result.memory.content_hash
            confidence, last_accessed, created_at = meta.get(ch, (1.0, None, None))
            eff = self._effective_confidence(confidence, last_accessed, created_at, now)

            if eff < min_confidence:
                continue

            result.debug_info["effective_confidence"] = eff
            result.debug_info["confidence"] = confidence if confidence is not None else 1.0
            result.debug_info["last_accessed"] = last_accessed
            hashes_to_touch.append(ch)
            enriched.append(result)

            if len(enriched) >= n_results:
                break

        if hashes_to_touch:
            def _touch(hashes=hashes_to_touch, ts=now):
                self.conn.executemany(
                    "UPDATE memories SET last_accessed = ? WHERE content_hash = ? AND deleted_at IS NULL",
                    [(int(ts), h) for h in hashes],
                )
                self.conn.commit()
            try:
                await self._execute_with_retry(_touch)
            except Exception as e:
                logger.warning(f"Failed to update last_accessed (non-fatal): {e}")

        return enriched

    async def update_memory_versioned(
        self,
        content_hash: str,
        new_content: str,
        new_tags: Optional[List[str]] = None,
        new_memory_type: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """Non-destructive versioned update using existing store/update infrastructure."""
        try:
            if not self.conn:
                return False, "Database not initialized", None

            def _check_exists():
                cursor = self.conn.execute(
                    "SELECT content_hash, tags, memory_type FROM memories WHERE content_hash = ? AND deleted_at IS NULL",
                    (content_hash,),
                )
                return cursor.fetchone()

            row = await self._execute_with_retry(_check_exists)
            if not row:
                return False, f"Memory {content_hash} not found", None

            old_hash, old_tags_str, old_type = row
            resolved_tags = new_tags if new_tags is not None else (
                [t for t in old_tags_str.split(",") if t] if old_tags_str else []
            )
            resolved_type = new_memory_type if new_memory_type is not None else old_type

            new_hash = generate_content_hash(new_content)
            new_memory = Memory(
                content=new_content,
                content_hash=new_hash,
                tags=resolved_tags,
                memory_type=resolved_type,
            )
            store_ok, store_msg = await self.store(new_memory, skip_semantic_dedup=True)
            if not store_ok:
                return False, f"Failed to store new version: {store_msg}", None

            meta_updates: Dict[str, Any] = {"metadata": {"superseded_by": new_hash}}
            if reason:
                meta_updates["metadata"]["evolution_reason"] = reason
            await self.update_memory_metadata(content_hash, meta_updates, preserve_timestamps=True)

            logger.info(f"Memory evolved: {old_hash[:8]} → {new_hash[:8]}")
            return True, "Memory versioned successfully", new_hash

        except Exception as e:
            logger.error(f"update_memory_versioned error: {e}")
            return False, str(e), None

    async def get_memory_history(self, content_hash: str) -> List[Dict[str, Any]]:
        """Return full version lineage for a memory, oldest-first."""
        try:
            if not self.conn:
                return []

            current = content_hash
            visited: Set[str] = set()
            while True:
                if current in visited:
                    break
                visited.add(current)
                _current = current

                def _get_parent(c=_current):
                    cursor = self.conn.execute(
                        """SELECT m.parent_id FROM memories m
                           WHERE m.content_hash = ? AND m.parent_id IS NOT NULL
                           AND EXISTS (SELECT 1 FROM memories p WHERE p.content_hash = m.parent_id)""",
                        (c,),
                    )
                    return cursor.fetchone()

                row = await self._execute_with_retry(_get_parent)
                if not row:
                    break
                current = row[0]

            root = current

            def _get_lineage(r=root):
                cursor = self.conn.execute(
                    """
                    WITH RECURSIVE lineage(content_hash, content, version, parent_id, superseded_by, created_at) AS (
                        SELECT content_hash, content, version, parent_id, superseded_by, created_at
                        FROM memories WHERE content_hash = ? AND deleted_at IS NULL
                        UNION ALL
                        SELECT m.content_hash, m.content, m.version, m.parent_id, m.superseded_by, m.created_at
                        FROM memories m
                        INNER JOIN lineage l ON m.parent_id = l.content_hash
                        WHERE m.deleted_at IS NULL
                    )
                    SELECT content_hash, content, version, parent_id, superseded_by, created_at
                    FROM lineage
                    ORDER BY COALESCE(version, 1) ASC
                    """,
                    (r,),
                )
                return cursor.fetchall()

            lineage_rows = await self._execute_with_retry(_get_lineage)
            return [
                {
                    "content_hash": r[0],
                    "content": r[1],
                    "version": r[2] or 1,
                    "parent_id": r[3],
                    "superseded_by": r[4],
                    "created_at": r[5],
                    "active": r[4] is None,
                }
                for r in lineage_rows
            ]

        except Exception as e:
            logger.error(f"get_memory_history error: {e}")
            return []
