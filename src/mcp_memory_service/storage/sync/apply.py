"""
Delta-sync Phase 4a apply functionality.

Implements apply_remote_event and advance_sync_cursor functions for pulling and applying
events from remote peers. Integrates with the resolver from Phase 2 for conflict resolution.

ADR-0021: Apply materializes directly to SQLite (not via POST /api/memories)
ADR-0022: Apply preserves original authorship (agent_id, event_id, HLC)
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional

from ..base import MemoryStorage
from .resolver import EventView, reduce_events

logger = logging.getLogger(__name__)


@dataclass
class ApplyResult:
    """Result of applying a remote event."""
    applied: bool
    materialized: bool  
    reason: str


def apply_remote_event(storage: MemoryStorage, event: Dict[str, Any]) -> ApplyResult:
    """
    Apply a remote sync event to local storage.
    
    Implements the three-step process:
    1. Insert event into sync_events (idempotent)
    2. Resolve conflicts using Phase 2 resolver
    3. Materialize if remote event wins
    
    Args:
        storage: The local storage instance
        event: Event dict from remote feed
        
    Returns:
        ApplyResult with applied/materialized flags and reason
    """
    try:
        content_hash = event["content_hash"]
        agent_id = event["agent_id"]
        event_id = event["event_id"]
        op = event["op"]
        hlc_physical = event["hlc_physical"]
        hlc_logical = event["hlc_logical"]
        embedding_model = event.get("embedding_model")
        embedding_dim = event.get("embedding_dim")
        payload = event.get("payload", {})
        
        # Step 1: Insert event into sync_events (idempotent via UNIQUE constraint)
        try:
            storage.conn.execute("""
                INSERT OR IGNORE INTO sync_events 
                (agent_id, event_id, op, content_hash, hlc_physical, hlc_logical, 
                 embedding_model, embedding_dim, payload, created_at, created_at_iso)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                agent_id, event_id, op, content_hash, hlc_physical, hlc_logical,
                embedding_model, embedding_dim, json.dumps(payload),
                time.time(), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            ))
            
            # Check if event was actually inserted (not a duplicate)
            cursor = storage.conn.execute(
                "SELECT 1 FROM sync_events WHERE agent_id = ? AND event_id = ?",
                (agent_id, event_id)
            )
            event_exists = cursor.fetchone() is not None
            
            if not event_exists:
                return ApplyResult(applied=False, materialized=False, reason="Failed to insert event")
                
        except Exception as e:
            logger.error(f"Error inserting sync event: {e}")
            return ApplyResult(applied=False, materialized=False, reason=f"Insert failed: {e}")
        
        # Step 2: Resolve conflicts using the Phase 2 resolver
        try:
            # Create EventView for the incoming remote event
            remote_event_view = EventView(
                hlc_physical=hlc_physical,
                hlc_logical=hlc_logical,
                agent_id=agent_id,
                event_id=event_id,
                op=op,
                content_hash=content_hash,
            )
            
            # Get all competing events for this content_hash from local sync_events
            cursor = storage.conn.execute("""
                SELECT hlc_physical, hlc_logical, agent_id, event_id, op, content_hash
                FROM sync_events 
                WHERE content_hash = ?
                ORDER BY hlc_physical DESC, hlc_logical DESC
            """, (content_hash,))
            
            competing_events = []
            for row in cursor.fetchall():
                competing_events.append(EventView(
                    hlc_physical=row[0],
                    hlc_logical=row[1], 
                    agent_id=row[2],
                    event_id=row[3],
                    op=row[4],
                    content_hash=row[5],
                ))
            
            # If no competing events, remote wins by default
            if not competing_events:
                winner = remote_event_view
            else:
                # Use resolver to find winner among all events (including remote)
                all_events = competing_events
                if remote_event_view not in competing_events:
                    all_events.append(remote_event_view)
                winner = reduce_events(all_events)
            
            # Step 3: Materialize if remote event wins
            if winner and winner.event_id == event_id and winner.agent_id == agent_id:
                # Remote event wins, materialize it
                try:
                    materialized = _materialize_event(storage, event)
                    storage.conn.commit()  # durability: persist event + materialization (ADR-0019/§8.5)
                    if materialized:
                        return ApplyResult(applied=True, materialized=True, reason="Remote event won and materialized")
                    else:
                        return ApplyResult(applied=True, materialized=False, reason="Remote event won but materialization failed")
                except Exception as e:
                    logger.error(f"Materialization failed: {e}")
                    return ApplyResult(applied=True, materialized=False, reason=f"Materialization error: {e}")
            else:
                # Remote event lost or tied - event recorded but not materialized
                storage.conn.commit()  # durability: persist the recorded event (ADR-0019/§8.5)
                return ApplyResult(applied=True, materialized=False, reason="Remote event lost conflict resolution")
                
        except Exception as e:
            logger.error(f"Error in conflict resolution: {e}")
            return ApplyResult(applied=True, materialized=False, reason=f"Resolver failed: {e}")
            
    except Exception as e:
        logger.error(f"Error applying remote event: {e}")
        return ApplyResult(applied=False, materialized=False, reason=f"Apply failed: {e}")


def _materialize_event(storage: MemoryStorage, event: Dict[str, Any]) -> bool:
    """
    Materialize an event directly into the memories table.
    
    Handles create/update/delete operations according to ADR-0021.
    """
    try:
        op = event["op"]
        content_hash = event["content_hash"] 
        payload = event.get("payload", {})
        embedding_model = event.get("embedding_model")
        
        if op == "delete":
            # Soft delete - set deleted_at timestamp
            storage.conn.execute("""
                UPDATE memories 
                SET deleted_at = ?
                WHERE content_hash = ?
            """, (
                time.time(),
                content_hash
            ))
            return True
            
        elif op in ("create", "update", "update_metadata"):
            # Extract content and metadata from payload
            content = payload.get("content", "")
            tags = payload.get("tags", [])
            metadata = payload.get("metadata", {})
            memory_type = payload.get("memory_type", "general")
            
            if not content and op == "create":
                # Content is required for create operations
                logger.warning(f"Create event missing content for hash {content_hash}")
                return False
            
            # Determine if the local embedding can be (re)generated now. When the event's
            # model matches ours (or is unknown/legacy) and we have content, we re-embed
            # locally and mark the memory searchable (pending=0). When the model differs,
            # we cannot produce a comparable vector → mark pending and leave it out of
            # search until a re-embed (ADR-0014/0016, §8.3).
            model_mismatch = bool(embedding_model) and embedding_model != storage.embedding_model_name
            can_embed = bool(content) and not model_mismatch
            embedding_pending = 0 if can_embed else 1

            # UPSERT the memory
            storage.conn.execute("""
                INSERT OR REPLACE INTO memories
                (content_hash, content, tags, memory_type, metadata,
                 created_at, created_at_iso, updated_at, updated_at_iso,
                 deleted_at, embedding_pending, store)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, 'default')
            """, (
                content_hash,
                content,
                json.dumps(tags) if tags else "[]",
                memory_type,
                json.dumps(metadata) if metadata else "{}",
                time.time(),  # Will be overridden if memory exists
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                time.time(),
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                embedding_pending
            ))

            # Generate the embedding locally when we can, so the materialized memory is
            # actually searchable (a memory row without a memory_embeddings vector is
            # invisible to semantic search). Mirrors store() (mixins/store.py).
            if can_embed:
                try:
                    from sqlite_vec import serialize_float32
                    rowid = storage.conn.execute(
                        "SELECT id FROM memories WHERE content_hash = ?", (content_hash,)
                    ).fetchone()[0]
                    embedding = storage._generate_embedding(content)
                    storage.conn.execute(
                        "DELETE FROM memory_embeddings WHERE rowid = ?", (rowid,)
                    )
                    storage.conn.execute(
                        "INSERT INTO memory_embeddings (rowid, content_embedding, store) VALUES (?, ?, ?)",
                        (rowid, serialize_float32(embedding), "default"),
                    )
                except Exception as emb_err:
                    # If embedding generation fails, keep the row but mark it pending so
                    # it is not served with a missing/stale vector (fail-safe, §8.3).
                    logger.warning("apply: embedding generation failed for %s: %s", content_hash, emb_err)
                    storage.conn.execute(
                        "UPDATE memories SET embedding_pending = 1 WHERE content_hash = ?",
                        (content_hash,),
                    )
            return True
            
        else:
            logger.warning(f"Unknown operation type: {op}")
            return False
            
    except Exception as e:
        logger.error(f"Error materializing event: {e}")
        return False


def advance_sync_cursor(storage: MemoryStorage, peer_id: str, last_seq: int) -> None:
    """
    Advance the sync cursor for a peer to the given sequence number.
    
    Args:
        storage: The local storage instance
        peer_id: Identifier of the peer
        last_seq: Last sequence number successfully processed
    """
    try:
        storage.conn.execute("""
            INSERT OR REPLACE INTO sync_cursor 
            (peer_id, last_seq_seen, updated_at)
            VALUES (?, ?, ?)
        """, (peer_id, last_seq, time.time()))
        # Durability (§8.5 / ADR-0019): the cursor and the applied events of the batch
        # must survive a crash. Commit here closes the batch transaction (apply writes +
        # cursor advance) atomically — without this they stay in an open transaction and
        # a crash loses applied events AND the cursor.
        storage.conn.commit()

        logger.debug(f"Advanced sync cursor for peer {peer_id} to seq {last_seq}")

    except Exception as e:
        logger.error(f"Error advancing sync cursor: {e}")
        raise