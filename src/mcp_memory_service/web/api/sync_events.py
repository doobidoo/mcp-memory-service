"""
Sync Events API endpoint for delta-sync Phase 4a.

Provides GET /api/sync/events for pulling event feed with pagination.
Implements ADR-0020 (pagination by seq) and ADR-0021 (enriched create events).
"""

import json
import logging
import os
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Query, Depends, HTTPException
from pydantic import BaseModel

from ...storage.base import MemoryStorage
from ...storage.sync.apply import _sqlite
from ...storage.mixins.base import _sanitize_log_value
from ..dependencies import get_storage
from ..oauth.middleware import require_read_access, require_write_access

logger = logging.getLogger(__name__)

router = APIRouter()


class SyncEventData(BaseModel):
    """Individual sync event in the feed response."""
    seq: int
    schema_version: int = 1
    event_id: str
    op: str
    content_hash: str
    agent_id: Optional[str]
    hlc_physical: int
    hlc_logical: int
    embedding_model: Optional[str]
    embedding_dim: Optional[int]
    payload: Dict[str, Any]


class SyncEventsResponse(BaseModel):
    """Response for GET /api/sync/events endpoint."""
    events: List[SyncEventData]
    next_seq: int
    has_more: bool


@router.get("/sync/events", response_model=SyncEventsResponse)
async def get_sync_events(
    since_seq: int = Query(0, description="Return events with seq > since_seq"),
    limit: int = Query(100, description="Maximum number of events to return", le=1000),
    _auth: bool = Depends(require_read_access),
    storage: MemoryStorage = Depends(get_storage),
) -> SyncEventsResponse:
    """
    Get sync events feed for delta-sync pull.
    
    Returns events with seq > since_seq, ordered by seq.
    Create events are enriched with content from memories table if available.
    
    Args:
        since_seq: Return events with seq greater than this value
        limit: Maximum events to return (capped at 1000)
        
    Returns:
        SyncEventsResponse with events, next_seq, and has_more flag
    """
    try:
        # Validate parameters
        if limit <= 0:
            raise HTTPException(status_code=400, detail="limit must be positive")
        if limit > 1000:
            limit = 1000
            
        # Get events from sync_events table
        cursor = _sqlite(storage).conn.execute("""
            SELECT 
                seq, schema_version, event_id, op, content_hash, agent_id, hlc_physical, hlc_logical,
                embedding_model, embedding_dim, payload
            FROM sync_events 
            WHERE seq > ?
            ORDER BY seq 
            LIMIT ?
        """, (since_seq, limit))
        
        rows = cursor.fetchall()
        events = []
        
        for row in rows:
            seq, schema_version, event_id, op, content_hash, agent_id, hlc_physical, hlc_logical, embedding_model, embedding_dim, payload = row
            
            # Parse payload JSON
            payload_dict = json.loads(payload) if payload else {}
            
            # For create events, enrich with content from memories table if available
            if op == "create":
                memory_cursor = _sqlite(storage).conn.execute(
                    "SELECT content FROM memories WHERE content_hash = ? AND deleted_at IS NULL",
                    (content_hash,)
                )
                memory_row = memory_cursor.fetchone()
                # Only override the payload's content when the table actually has
                # non-empty content. An empty/zeroed table row (e.g. a memory whose
                # content was cleared) must NOT clobber the real content carried in the
                # original create payload — otherwise the puller receives a create with
                # no content and cannot materialize it.
                if memory_row and memory_row[0]:
                    payload_dict["content"] = memory_row[0]
            
            events.append(SyncEventData(
                seq=seq,
                schema_version=schema_version if schema_version is not None else 1,
                event_id=event_id,
                op=op,
                content_hash=content_hash,
                agent_id=agent_id,
                hlc_physical=hlc_physical,
                hlc_logical=hlc_logical,
                embedding_model=embedding_model,
                embedding_dim=embedding_dim,
                payload=payload_dict
            ))
        
        # Determine next_seq and has_more
        if events:
            next_seq = events[-1].seq
        else:
            next_seq = since_seq
            
        # Check if there are more events beyond what we returned
        has_more = False
        if events and len(events) == limit:
            # Check if there's at least one more event beyond our result
            check_cursor = _sqlite(storage).conn.execute(
                "SELECT 1 FROM sync_events WHERE seq > ? LIMIT 1",
                (next_seq,)
            )
            has_more = check_cursor.fetchone() is not None
        
        return SyncEventsResponse(
            events=events,
            next_seq=next_seq,
            has_more=has_more
        )
        
    except Exception as e:
        logger.error("Error retrieving sync events: %s", _sanitize_log_value(e))
        raise HTTPException(status_code=500, detail="Internal server error")


# --- Phase 4c: outbound push ingestion (ADR-0027) ---

class IngestEventsRequest(BaseModel):
    """Request body for POST /api/sync/events (push ingestion)."""
    events: List[Dict[str, Any]]


class IngestEventResult(BaseModel):
    """Per-event ingestion result."""
    event_id: str
    status: str  # applied | skipped_duplicate | failed


class IngestEventsResponse(BaseModel):
    """Response for POST /api/sync/events."""
    results: List[IngestEventResult]
    applied: int
    skipped: int
    failed: int


def _push_allowed_agents() -> Optional[set]:
    """Allow-list from MCP_SYNC_PUSH_ALLOWED_AGENTS (comma-separated). None = accept any (R4)."""
    raw = os.getenv("MCP_SYNC_PUSH_ALLOWED_AGENTS", "").strip()
    if not raw:
        return None
    return {a.strip() for a in raw.split(",") if a.strip()}


@router.post("/sync/events", response_model=IngestEventsResponse)
async def ingest_sync_events(
    body: IngestEventsRequest,
    _auth: bool = Depends(require_write_access),
    storage: MemoryStorage = Depends(get_storage),
) -> IngestEventsResponse:
    """
    Ingest pushed sync events from a spoke (delta-sync Phase 4c, ADR-0027).

    Applies each event via apply_remote_event (idempotent, preserves authorship).
    Authorship guard (R4): if MCP_SYNC_PUSH_ALLOWED_AGENTS is set, a batch containing an
    event whose agent_id is not in the allow-list is rejected whole (403) and NOTHING is
    applied. If unset, any non-empty agent_id is accepted.
    """
    from ...storage.sync.apply import apply_remote_event

    allowed = _push_allowed_agents()

    # R4: validate authorship for the WHOLE batch before applying anything.
    for ev in body.events:
        agent_id = (ev.get("agent_id") or "").strip()
        if not agent_id:
            raise HTTPException(status_code=403, detail="event missing agent_id")
        if allowed is not None and agent_id not in allowed:
            raise HTTPException(
                status_code=403,
                detail=f"agent_id not allowed to push: {agent_id}",
            )

    results: List[IngestEventResult] = []
    applied = skipped = failed = 0
    stop = False
    for ev in body.events:
        eid = ev.get("event_id", "")
        if stop:
            # Applied in order; after the first failure the remaining events are NOT
            # applied (the spoke re-pushes from the last acked seq, R6).
            results.append(IngestEventResult(event_id=eid, status="failed"))
            failed += 1
            continue
        try:
            res = apply_remote_event(storage, ev)
            # Classify by the real ApplyResult contract (apply.py uses INSERT OR IGNORE,
            # so a duplicate does not fail — it returns applied=True, materialized=False).
            if res.applied and res.materialized:
                status = "applied"
                applied += 1
            elif res.applied and not res.materialized:
                # event recorded but not materialized: duplicate or lost conflict resolution
                # — for push semantics the hub already has it / local winner kept.
                status = "skipped_duplicate"
                skipped += 1
            elif getattr(res, "skippable", False) or res.reason == "Replay with altered payload rejected":
                # Benignly non-ingestable: either a permanently empty-content create
                # (skippable) or a same-identity event whose payload diverges from the one
                # the hub already holds (anti-replay protection kept the hub's version).
                # Neither should fail-stop the push — the hub keeps its authoritative copy
                # and the spoke must be allowed to advance past it (otherwise a single
                # divergent event blocks the entire push backlog).
                status = "skipped_duplicate"
                skipped += 1
            else:
                status = "failed"
                failed += 1
                stop = True
        except Exception as e:
            logger.error("ingest apply failed for %s: %s", _sanitize_log_value(eid), _sanitize_log_value(e))
            status = "failed"
            failed += 1
            stop = True
        results.append(IngestEventResult(event_id=eid, status=status))

    return IngestEventsResponse(results=results, applied=applied, skipped=skipped, failed=failed)


class BaselineResponse(BaseModel):
    """Bootstrap baseline: the source's current state as replayable events + watermark."""
    events: List[Dict[str, Any]]
    watermark: int
    count: int


@router.get("/sync/baseline", response_model=BaselineResponse)
async def get_sync_baseline(
    storage: MemoryStorage = Depends(get_storage),
    _auth=Depends(require_read_access),
) -> BaselineResponse:
    """Delta-sync Phase 5: serve a bootstrap baseline of the current state (RFC §9.3).

    A new/reinstalled spoke calls this once to catch up on the corpus without reprocessing,
    then pulls live events from the returned watermark forward. Generated on demand; the
    watermark is read atomically with the snapshot (see generate_baseline).
    """
    from ...storage.sync.bootstrap import generate_baseline
    import asyncio
    try:
        # Greptile P2-1: generate_baseline is synchronous — it takes the snapshot lock, runs
        # BEGIN IMMEDIATE, reads the whole corpus and builds every event. Running it directly
        # on the event loop would stall unrelated HTTP requests. Offload to a worker thread
        # (its internal lock keeps the snapshot atomic regardless of thread).
        events, watermark = await asyncio.to_thread(generate_baseline, storage)
        return BaselineResponse(events=events, watermark=watermark, count=len(events))
    except Exception as e:
        logger.error("baseline generation failed: %s", _sanitize_log_value(e))
        raise HTTPException(status_code=500, detail="baseline generation failed")
