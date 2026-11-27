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
Memory CRUD endpoints for the HTTP interface.
"""

import asyncio
import heapq
import logging
import socket
from collections import Counter
from typing import List, Optional, Dict, Any, Literal, Set, Tuple

from fastapi import APIRouter, HTTPException, Depends, Query, Request
from pydantic import BaseModel, Field, field_validator

from ...storage.base import MemoryStorage
from ...storage.mixins.base import deserialize_embedding
from ...storage.resonance_db import ResonanceDB
from ...models.memory import Memory
from ...models.ontology import coerce_memory_type, get_all_types
from ...services.memory_service import MemoryService, _chunk_metadata
from ...utils.hashing import generate_content_hash
from ...utils.content_splitter import split_content
from ...compat import _sanitize_log_value
from ...config import (
    INCLUDE_HOSTNAME,
    ENABLE_AUTO_SPLIT,
    CONTENT_PRESERVE_BOUNDARIES,
    CONTENT_SPLIT_OVERLAP,
)
# OAuth config no longer needed - auth is always enabled
from ..dependencies import get_storage, get_memory_service
from .store_scope import resolve_store
from ..sse import sse_manager, create_memory_stored_event, create_memory_deleted_event

# OAuth authentication imports
from ..oauth.middleware import (
    require_read_access,
    require_store_read,
    require_store_write,
    get_current_user,
    AuthenticationResult,
)

# Markers a split edit writes on the chunks it stores (#2143): the hash the
# memory keeps across edits, and the chunks of its last edit.
SPLIT_OF_KEY = "split_of"
SPLIT_CHUNKS_KEY = "split_chunk_hashes"

router = APIRouter()
logger = logging.getLogger(__name__)


# Request/Response Models
class MemoryCreateRequest(BaseModel):
    """Request model for creating a new memory."""
    content: str = Field(..., description="The memory content to store")
    tags: List[str] = Field(default=[], description="Tags to categorize the memory")
    memory_type: Optional[str] = Field(None, description="Type of memory. Validated against the built-in ontology (base types: observation, decision, learning, error, pattern, planning, ceremony, milestone, stakeholder, meeting, research, communication). Unknown types are coerced to 'observation' and a warning is included in the response. Register additional types via MCP_CUSTOM_MEMORY_TYPES env var.")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata for the memory")
    client_hostname: Optional[str] = Field(None, description="Client machine hostname for source tracking")
    conversation_id: Optional[str] = Field(None, description="Optional conversation identifier. When provided, semantic deduplication is skipped, allowing multiple incremental memories from the same conversation to be stored even if their content is topically similar.")
    store: str = Field("default", description="Target store partition (default: 'default'). Use 'all' only for read scopes.")
    restore_deleted: bool = Field(False, description="Bring back content that has a delete on record. Default false: re-sending content this server deleted is refused with status 409 (the delete's event_id, HLC and date are in the detail) instead of silently reviving it. Set true to purge the tombstone and store it again as a deliberate re-create.")

    @field_validator("content")
    @classmethod
    def _content_must_not_be_blank(cls, value: str) -> str:
        """Reject blank content before it becomes a zero-vector memory (#1791)."""
        if not value.strip():
            raise ValueError("content must not be empty or whitespace-only")
        return value


class MemoryUpdateRequest(BaseModel):
    """Request model for updating memory content, tags, type, and metadata."""
    content: Optional[str] = Field(None, description="Updated memory content, stored as a new version")
    tags: Optional[List[str]] = Field(None, description="Updated tags to categorize the memory")
    memory_type: Optional[str] = Field(None, description="Updated memory type (e.g., 'note', 'reminder', 'fact')")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata for the memory")

    @field_validator("content")
    @classmethod
    def _content_must_not_be_blank(cls, value: Optional[str]) -> Optional[str]:
        """Reject blank content instead of creating a zero-vector version."""
        if value is not None and not value.strip():
            raise ValueError("content must not be empty or whitespace-only")
        return value


class MemoryResponse(BaseModel):
    """Response model for memory data."""
    content: str
    content_hash: str
    tags: List[str]
    memory_type: Optional[str]
    metadata: Dict[str, Any]
    created_at: Optional[float]
    created_at_iso: Optional[str]
    updated_at: Optional[float]  
    updated_at_iso: Optional[str]


class MemoryListResponse(BaseModel):
    """Response model for paginated memory list."""
    memories: List[MemoryResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class ContentHashListResponse(BaseModel):
    """Response model for paginated content hash list."""
    hashes: List[str]
    next_cursor: Optional[int] = None
    has_more: bool


class MemoryCreateResponse(BaseModel):
    """Response model for memory creation."""
    success: bool
    message: str
    content_hash: Optional[str] = None
    memory: Optional[MemoryResponse] = None
    dedup_outcome: Optional[str] = Field(
        None,
        description=(
            "Semantic-dedup decision, when one was made. "
            "'deduplicated': nothing new was stored; matched_hash is the existing memory. "
            "'superseded': the new memory was stored and matched_hash was marked superseded "
            "by it (MCP_SEMANTIC_DEDUP_MODE=supersede). Null when dedup did not decide."
        ),
    )
    matched_hash: Optional[str] = Field(
        None,
        description="Content hash of the memory semantic dedup matched. See dedup_outcome.",
    )
    dedup_decisions: Optional[List[Dict[str, Any]]] = Field(
        None,
        description=(
            "Per-chunk semantic-dedup decisions when the write was split. "
            "Each item has chunk_index, dedup_outcome, and matched_hash. "
            "A single decision is also copied to dedup_outcome and matched_hash."
        ),
    )


def _dedup_response_fields(result: dict) -> dict:
    """Semantic-dedup fields for MemoryCreateResponse. Empty when dedup did not decide."""
    fields = {}
    outcome = result.get("dedup_outcome")
    if outcome:
        fields["dedup_outcome"] = outcome
        fields["matched_hash"] = result.get("matched_hash")
    decisions = result.get("dedup_decisions")
    if decisions:
        fields["dedup_decisions"] = decisions
    return fields


def _superseded_hashes(result: dict) -> List[str]:
    """Hashes retired by this store, from the chunk list or the single decision."""
    decisions = result.get("dedup_decisions") or []
    hashes = [
        decision.get("matched_hash")
        for decision in decisions
        if decision.get("dedup_outcome") == "superseded" and decision.get("matched_hash")
    ]
    if hashes:
        return hashes
    matched = result.get("matched_hash")
    if result.get("dedup_outcome") == "superseded" and matched:
        return [matched]
    return []


def _with_supersede_note(result: dict, message: str) -> str:
    """Append the replaced hash when this store superseded an older memory."""
    hashes = _superseded_hashes(result)
    if not hashes:
        return message
    return f"{message} (superseded {', '.join(hashes)})"


class MemoryDeleteResponse(BaseModel):
    """Response model for memory deletion."""
    success: bool
    message: str
    content_hash: str


class MemoryUpdateResponse(BaseModel):
    """Response model for memory update."""
    success: bool
    message: str
    content_hash: str
    memory: Optional[MemoryResponse] = None


class TagResponse(BaseModel):
    """Response model for a single tag with its count."""
    tag: str
    count: int


class TagListResponse(BaseModel):
    """Response model for tags list."""
    tags: List[TagResponse]


def memory_to_response(memory: Memory) -> MemoryResponse:
    """Convert Memory model to response format."""
    return MemoryResponse(
        content=memory.content,
        content_hash=memory.content_hash,
        tags=memory.tags,
        memory_type=memory.memory_type,
        metadata=memory.metadata,
        created_at=memory.created_at,
        created_at_iso=memory.created_at_iso,
        updated_at=memory.updated_at,
        updated_at_iso=memory.updated_at_iso
    )


def _client_hostname(request: MemoryCreateRequest, http_request: Request) -> Optional[str]:
    """Hostname for consistent tagging (the tagging itself happens in the service).

    Client-provided hostname first, then the X-Client-Hostname header, then
    this server's hostname (the original behavior).
    """
    if not INCLUDE_HOSTNAME:
        return None
    return (
        request.client_hostname
        or http_request.headers.get('X-Client-Hostname')
        or socket.gethostname()
    )


def _tags_with_agent(request: MemoryCreateRequest, http_request: Request) -> List[str]:
    """The request's tags, plus agent:<id> when the X-Agent-ID header names one."""
    tags = list(request.tags)
    agent_id = http_request.headers.get('X-Agent-ID')
    if agent_id and f"agent:{agent_id}" not in tags:
        tags.append(f"agent:{agent_id}")
    return tags


async def _broadcast_stored(result: dict, store: str) -> None:
    """Send the memory_stored SSE event; a failed broadcast never fails the request."""
    try:
        # A chunked store is announced with its first chunk.
        stored = result["memory"] if "memory" in result else result["memories"][0]
        memory_data = {
            "content_hash": stored["content_hash"],
            "content": stored["content"],
            "tags": stored["tags"],
            "memory_type": stored["memory_type"],
        }
        await sse_manager.broadcast_event(create_memory_stored_event(memory_data, store=store))
    except Exception as e:
        logger.warning("Failed to broadcast memory_stored event: %s", _sanitize_log_value(e))


def _coercion_warning(requested: Optional[str], effective: Optional[str]) -> str:
    """Name the runtime registration when an unknown memory_type was coerced (#2127).

    Memory.__post_init__ rewrites a type the ontology does not know to
    "observation". Without an explicit warning the caller sees "success"
    while their type filter silently breaks. `is not None` (not truthy), so
    an explicit empty string, also coerced to 'observation', still warns;
    this mirrors the MCP handler's presence-based check.
    """
    if requested is None or not effective or requested == effective:
        return ""
    return (
        f" Warning: requested memory_type '{requested}' is not in "
        f"the ontology — stored as '{effective}'. Register it without a "
        f"restart with `memory ontology add {requested}`, or at startup via "
        f"MCP_CUSTOM_MEMORY_TYPES, e.g. '{{\"{requested}\": []}}'."
    )


def _refused_chunks_note(result: dict) -> str:
    """#1560: name the chunks a recorded delete kept out, and how to bring them back."""
    refused_chunks = result.get("refused_chunks") or []
    if not refused_chunks:
        return ""
    return (
        f"; {len(refused_chunks)} chunk(s) not stored: deleted on "
        f"{refused_chunks[0]['deleted_at']}, pass restore_deleted=true "
        f"to bring them back"
    )


def _created_response(result: dict, requested_type: Optional[str]) -> MemoryCreateResponse:
    """The success response for a single memory, or for a chunked store by its first chunk."""
    if "memory" in result:
        stored = result["memory"]
        message = _with_supersede_note(result, "Memory stored successfully")
    else:
        stored = result["memories"][0]
        message = _with_supersede_note(
            result, f"Memory stored as {result['total_chunks']} chunks"
        ) + _refused_chunks_note(result)
    return MemoryCreateResponse(
        success=True,
        message=message + _coercion_warning(requested_type, stored.get("memory_type")),
        content_hash=stored["content_hash"],
        memory=stored,
        **_dedup_response_fields(result),
    )


def _not_stored_response(result: dict) -> MemoryCreateResponse:
    """The answer to a store the MemoryService did not perform."""
    refused = result.get("refused_recreate")
    if refused:
        # #1560: re-sending content this server deleted conflicts with the
        # recorded delete — 409, naming the delete, until the client says
        # restore_deleted=true. Not a 200/success=false: bundled clients
        # would log and move on, and the memory would quietly come back
        # on the next sync.
        raise HTTPException(
            status_code=409,
            detail={"message": result.get("error", "Not stored"), **refused},
        )
    return MemoryCreateResponse(
        success=False,
        message=str(result.get("error", "Failed to store memory")),
        content_hash=None,
        **_dedup_response_fields(result),
    )


@router.post("/memories", response_model=MemoryCreateResponse, tags=["memories"])
async def store_memory(
    request: MemoryCreateRequest,
    http_request: Request,
    memory_service: MemoryService = Depends(get_memory_service),
    user: AuthenticationResult = Depends(get_current_user)
):
    """
    Store a new memory.

    Uses the MemoryService for consistent business logic including content processing,
    hostname tagging, and metadata enrichment.
    """
    try:
        client_hostname = _client_hostname(request, http_request)
        tags = _tags_with_agent(request, http_request)

        if request.store == "all":
            raise HTTPException(
                status_code=400,
                detail="store='all' is only valid for read scopes",
            )

        # Store write is gated per-store (store is in the body, so the check is
        # here rather than in a dependency).
        user.require_store_scope("write", request.store)

        # Use injected MemoryService for consistent business logic (hostname tagging handled internally)
        result = await memory_service.store_memory(
            content=request.content,
            tags=tags,
            memory_type=request.memory_type,
            metadata=request.metadata,
            client_hostname=client_hostname,
            conversation_id=request.conversation_id,
            store=request.store,
            restore_deleted=request.restore_deleted,
        )

        if not result["success"]:
            return _not_stored_response(result)
        await _broadcast_stored(result, request.store)
        return _created_response(result, request.memory_type)

    except HTTPException:
        # Validation errors (e.g. store='all' on a write) must reach the client
        # as their real status code instead of being flattened to 500.
        raise
    except Exception as e:
        logger.error("Failed to store memory: %s", _sanitize_log_value(e))
        raise HTTPException(status_code=500, detail="Failed to store memory. Please try again.")


@router.get("/memories", response_model=MemoryListResponse, tags=["memories"])
async def list_memories(
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(10, ge=1, le=100, description="Number of memories per page"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    memory_type: Optional[str] = Query(None, description="Filter by memory type"),
    tag_match: Optional[str] = Query("any", description="Tag matching mode: 'any' (OR) or 'all' (AND)"),
    store: str = Query("default", description="Target store partition (default: 'default'). Use 'all' for every store."),
    memory_service: MemoryService = Depends(get_memory_service),
    user: AuthenticationResult = Depends(require_store_read)
):
    """
    List memories with pagination and optional filtering.

    Uses the MemoryService for consistent business logic and optimal database-level filtering.
    """
    try:
        # Split comma-separated tags into list for proper multi-tag filtering
        tags_list = [t.strip() for t in tag.split(",") if t.strip()] if tag else None

        # Use the injected service for consistent, performant memory listing
        result = await memory_service.list_memories(
            page=page,
            page_size=page_size,
            tags=tags_list,
            memory_type=memory_type,
            tag_match=tag_match,
            store=resolve_store(store),
        )

        return MemoryListResponse(
            memories=result["memories"],
            total=result["total"],
            page=result["page"],
            page_size=result["page_size"],
            has_more=result["has_more"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list memories: {str(e)}")


@router.get('/memories/hashes', response_model=ContentHashListResponse, tags=['memories'])
async def list_content_hashes(
    cursor: int = Query(0, ge=0, description='id-based cursor; pass next_cursor from previous page'),
    limit: int = Query(1000, ge=1, le=5000, description='max hashes per page'),
    include_deleted: bool = Query(False),
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access),
):
    try:
        page = await storage.list_content_hashes_page(after_id=cursor, limit=limit, include_deleted=include_deleted)
        hashes = [h for _, h in page]
        has_more = len(page) == limit
        next_cursor = page[-1][0] if (page and has_more) else None
        return ContentHashListResponse(hashes=hashes, next_cursor=next_cursor, has_more=has_more)
    except Exception as e:
        # Log the real error for operators; return a generic message so a
        # read-access client never sees raw DB/internal details (Greptile P2).
        logger.error("Failed to list content hashes: %s", _sanitize_log_value(e))
        raise HTTPException(status_code=500, detail="Failed to list content hashes")


@router.get("/memories/{content_hash}", response_model=MemoryResponse, tags=["memories"])
async def get_memory(
    content_hash: str,
    store: str = Query("default", description="Target store partition (default: 'default'). Use 'all' for every store."),
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_store_read)
):
    """
    Get a specific memory by its content hash.

    Retrieves a single memory entry using its unique content hash identifier.
    The lookup honours the same ``store`` scope as the list endpoint, so a hash
    that belongs to another partition is reported as not found.
    """
    try:
        memory = await storage.get_by_hash(content_hash, store=resolve_store(store))
        
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        return memory_to_response(memory)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get memory: {str(e)}")


@router.delete("/memories/{content_hash}", response_model=MemoryDeleteResponse, tags=["memories"])
async def delete_memory(
    content_hash: str,
    store: str = Query("default", description="Target store partition (default: 'default'). Use 'all' for every store."),
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_store_write)
):
    """
    Delete a memory by its content hash.

    Permanently removes a memory entry from the storage. The hash is resolved
    within the requested ``store`` scope first, so a hash owned by another
    partition cannot be deleted through a different scope.
    """
    if store == "all":
        raise HTTPException(
            status_code=400,
            detail="store='all' is only valid for read scopes",
        )

    try:
        existing = await storage.get_by_hash(content_hash, store=resolve_store(store))
        if not existing:
            raise HTTPException(status_code=404, detail="Memory not found")

        # Confine the delete itself, not only the lookup before it.
        success, message = await storage.delete(content_hash, store=resolve_store(store))
        
        # Broadcast SSE event for memory deletion
        try:
            event = create_memory_deleted_event(content_hash, success, store=store)
            await sse_manager.broadcast_event(event)
        except Exception as e:
            # Don't fail the request if SSE broadcasting fails
            logger.warning("Failed to broadcast memory_deleted event: %s", _sanitize_log_value(e))
        
        return MemoryDeleteResponse(
            success=success,
            message=message,
            content_hash=content_hash
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete memory: %s", _sanitize_log_value(e))
        raise HTTPException(status_code=500, detail="Failed to delete memory. Please try again.")


def _build_memory_replacement(
    existing_memory: Memory, request: MemoryUpdateRequest
) -> Memory:
    """Build a replacement while retaining fields omitted by the caller."""
    inherited_metadata = dict(existing_memory.metadata or {})
    inherited_metadata.pop("superseded_by", None)
    inherited_metadata.pop("evolution_reason", None)
    final_metadata = inherited_metadata
    if request.metadata:
        final_metadata.update(request.metadata)
    for key in ("tags", "type"):
        final_metadata.pop(key, None)
    content_hash = generate_content_hash(request.content)
    if content_hash == existing_memory.content_hash:
        content_hash = generate_content_hash(
            f"{request.content}\0{existing_memory.content_hash}"
        )
    return Memory(
        content=request.content,
        content_hash=content_hash,
        tags=(
            request.tags
            if request.tags is not None
            else [tag for tag in existing_memory.tags if tag != "conflict:unresolved"]
        ),
        memory_type=(
            coerce_memory_type(request.memory_type)
            if request.memory_type is not None
            else existing_memory.memory_type
        ),
        metadata=final_metadata,
    )


async def _store_unversioned_replacement(
    storage: MemoryStorage,
    memory_service: MemoryService,
    content_hash: str,
    store: str,
    replacement: Memory,
) -> Tuple[bool, str, Optional[str], bool]:
    """Store a replacement and remove its original on unversioned backends."""
    max_length = getattr(storage, "max_content_length", None)
    if ENABLE_AUTO_SPLIT and max_length and len(replacement.content) > max_length:
        return await _store_unversioned_split_replacement(
            storage, memory_service, content_hash, store, replacement, max_length
        )

    success, message = await storage.store(
        replacement,
        skip_semantic_dedup=True,
        store=resolve_store(store),
    )
    if not success:
        if "Duplicate content detected (exact match)" not in message:
            return False, message, None, False
        lookup_store = resolve_store(store)
        existing = await storage.get_by_hash(replacement.content_hash, store=lookup_store)
        # A previous attempt may have stored the replacement before the
        # original deletion failed. Resume only when every editable field matches;
        # another memory that merely shares the text is still a conflict.
        if (
            existing is None
            or existing.content != replacement.content
            or set(existing.tags or []) != set(replacement.tags)
            or existing.memory_type != replacement.memory_type
            or (existing.metadata or {}) != (replacement.metadata or {})
        ):
            return False, message, None, False
        message = "Replacement already stored; retrying original deletion"

    delete_success, delete_message = await storage.delete(
        content_hash, store=resolve_store(store)
    )
    if not delete_success:
        return False, (
            f"Stored replacement {replacement.content_hash}, but failed to delete "
            f"original memory: {delete_message}"
        ), None, False

    await memory_service._run_post_store_steps(replacement)
    return True, message, replacement.content_hash, False


def _owns_chunk_row(
    existing: Optional[Memory],
    chunk_hash: str,
    plain_chunk_hash: str,
    content_hash: str,
    replacement_hash: str,
    split_of: Optional[str] = None,
) -> bool:
    """Whether a split edit of content_hash may update the row at chunk_hash in place."""
    if existing is None:
        return False
    return (
        chunk_hash == content_hash
        # A salted hash is derived from the edited memory's own hash and the
        # chunk position, so only an earlier split edit of this memory can have
        # written the row, whatever full text that edit carried.
        or chunk_hash != plain_chunk_hash
        or (existing.metadata or {}).get("original_hash") == replacement_hash
        # Chunks of an earlier split edit of this memory carry this marker, so
        # the next edit recognises them even under their plain hash (#2143).
        or (
            split_of is not None
            and (existing.metadata or {}).get(SPLIT_OF_KEY) == split_of
        )
    )


def _split_of(base_row: Optional[Memory], content_hash: str) -> str:
    """The hash a memory keeps across split edits, so its chunks stay recognisable."""
    if base_row is not None:
        marker = (base_row.metadata or {}).get(SPLIT_OF_KEY)
        if marker:
            return marker
    return content_hash


def _split_chunk_hashes(row: Optional[Memory]) -> Set[str]:
    """Hashes of the chunks the last split edit of this memory wrote."""
    values = (row.metadata or {}).get(SPLIT_CHUNKS_KEY) if row is not None else None
    return {value for value in values or [] if isinstance(value, str)}


def _salt_chunk_hash(chunk: str, content_hash: str, index: int) -> str:
    """Hash for a chunk that cannot keep its plain hash in this split edit."""
    return generate_content_hash(f"{chunk}\0{content_hash}\0{index}")


async def _owns_current_generation(
    storage: MemoryStorage, lookup_store: Optional[str], ledger: List[str]
) -> bool:
    """Whether the list on the store is still the one this edit wrote.

    Two edits of the same memory can start from the same list. The one that wrote
    last owns the generation; the other must not delete anything, or the saved list
    ends up naming a chunk that is no longer there.
    """
    if not ledger:
        return False
    owner = await storage.get_by_hash(ledger[0], store=lookup_store)
    if owner is None:
        return False
    current = (owner.metadata or {}).get(SPLIT_CHUNKS_KEY)
    return isinstance(current, list) and list(current) == list(ledger)


async def _plan_split_chunks(
    storage: MemoryStorage,
    lookup_store: Optional[str],
    chunks: List[str],
    content_hash: str,
    replacement_hash: str,
    split_of: str,
) -> List[Tuple[str, str, Optional[Memory]]]:
    """Pick the row each chunk is written under and whether that row is reused."""
    planned: List[Tuple[str, str, Optional[Memory]]] = []
    claimed: Set[str] = set()
    for index, chunk in enumerate(chunks):
        plain_chunk_hash = generate_content_hash(chunk)
        chunk_hash = plain_chunk_hash
        if chunk_hash in claimed:
            chunk_hash = _salt_chunk_hash(chunk, content_hash, index)
        existing = await storage.get_by_hash(chunk_hash, store=lookup_store)
        can_reuse = chunk_hash not in claimed and _owns_chunk_row(
            existing, chunk_hash, plain_chunk_hash,
            content_hash, replacement_hash, split_of,
        )
        if (
            existing is not None
            and not (existing.content == chunk and can_reuse)
            and chunk_hash == plain_chunk_hash
        ):
            chunk_hash = _salt_chunk_hash(chunk, content_hash, index)
            existing = await storage.get_by_hash(chunk_hash, store=lookup_store)
            can_reuse = chunk_hash not in claimed and _owns_chunk_row(
                existing, chunk_hash, plain_chunk_hash,
                content_hash, replacement_hash, split_of,
            )
        claimed.add(chunk_hash)
        if existing is not None and existing.content == chunk and can_reuse:
            planned.append((chunk, chunk_hash, existing))
        else:
            planned.append((chunk, chunk_hash, None))
    return planned


def _split_chunk_metadata(
    replacement: Memory,
    index: int,
    total: int,
    chunk_hash: str,
    split_of: str,
    ledger: List[str],
) -> Dict[str, Any]:
    """Metadata of one chunk of a split edit, with the markers a later edit needs."""
    extra = {SPLIT_OF_KEY: split_of}
    if index == 0:
        extra[SPLIT_CHUNKS_KEY] = ledger
    return _chunk_metadata(
        replacement.metadata, index, total,
        replacement.content_hash, chunk_hash, extra,
    )


async def _write_split_chunks(
    storage: MemoryStorage,
    planned: List[Tuple[str, str, Optional[Memory]]],
    replacement: Memory,
    content_hash: str,
    store: str,
    lookup_store: Optional[str],
    split_of: str,
    ledger: List[str],
) -> Tuple[bool, str, List[Memory], bool]:
    """Write the planned chunks, updating the reused rows in place."""
    stored_chunks: List[Memory] = []
    original_reused = False
    for index, (chunk, chunk_hash, existing) in enumerate(planned):
        chunk_metadata = _split_chunk_metadata(
            replacement, index, len(planned), chunk_hash, split_of, ledger
        )
        if existing is not None:
            # Only the original row or a chunk of this memory's split edits may
            # be updated in place; an independent memory keeps its own fields.
            original_reused = original_reused or chunk_hash == content_hash
            if (
                set(existing.tags or []) != set(replacement.tags)
                or existing.memory_type != replacement.memory_type
                or (existing.metadata or {}) != chunk_metadata
            ):
                fields_ok, fields_message = await storage.update_memory_metadata(
                    chunk_hash,
                    {
                        "tags": replacement.tags,
                        "memory_type": replacement.memory_type,
                        "metadata": chunk_metadata,
                    },
                    preserve_timestamps=True,
                    store=resolve_store(store),
                )
                if not fields_ok:
                    return False, (
                        f"Failed to apply edited fields to reused chunk: {fields_message}"
                    ), [], False
                existing = await storage.get_by_hash(chunk_hash, store=lookup_store)
            stored_chunks.append(existing)
            continue
        chunk_memory = Memory(
            content=chunk,
            content_hash=chunk_hash,
            tags=replacement.tags,
            memory_type=replacement.memory_type,
            metadata=chunk_metadata,
        )
        success, message = await storage.store(
            chunk_memory,
            skip_semantic_dedup=True,
            store=resolve_store(store),
        )
        if not success:
            return False, message, [], False
        stored_chunks.append(chunk_memory)
    return True, "", stored_chunks, original_reused


async def _store_unversioned_split_replacement(
    storage: MemoryStorage,
    memory_service: MemoryService,
    content_hash: str,
    store: str,
    replacement: Memory,
    max_length: int,
) -> Tuple[bool, str, Optional[str], bool]:
    """Split an oversized replacement the same way store_memory() does."""
    lookup_store = resolve_store(store)
    base_row = await storage.get_by_hash(content_hash, store=lookup_store)
    # A split edit marks the chunks it writes and records them on the row the
    # memory keeps, so the next edit replaces that generation instead of
    # leaving it in the store (#2143). Only this path writes those two keys.
    split_of = _split_of(base_row, content_hash)
    previous_split_hashes = _split_chunk_hashes(base_row)
    chunks = split_content(
        replacement.content,
        max_length=max_length,
        preserve_boundaries=CONTENT_PRESERVE_BOUNDARIES,
        overlap=CONTENT_SPLIT_OVERLAP,
    )
    planned = await _plan_split_chunks(
        storage, lookup_store, chunks,
        content_hash, replacement.content_hash, split_of,
    )
    ledger = [chunk_hash for _chunk, chunk_hash, _existing in planned]
    written, message, stored_chunks, original_reused = await _write_split_chunks(
        storage, planned, replacement, content_hash,
        store, lookup_store, split_of, ledger,
    )
    if not written:
        return False, message, None, False

    # Written first, deleted after: a failure here leaves the memory larger than
    # expected, never empty.
    #
    # Another edit of the same memory can start from this same list and write after
    # this one. Its ledger is the authority then, and removing a chunk it kept would
    # leave the saved list pointing at a row that is gone. So the generation is read
    # back first: only the edit whose ledger is still on the row cleans up.
    if previous_split_hashes and await _owns_current_generation(storage, lookup_store, ledger):
        for stale_hash in sorted(previous_split_hashes - set(ledger) - {content_hash}):
            stale = await storage.get_by_hash(stale_hash, store=lookup_store)
            if stale is None or (stale.metadata or {}).get(SPLIT_OF_KEY) != split_of:
                continue
            deleted, delete_message = await storage.delete(
                stale_hash, store=resolve_store(store)
            )
            if not deleted:
                return False, (
                    f"Stored replacement chunks, but failed to delete chunk "
                    f"{stale_hash} of the previous edit: {delete_message}"
                ), None, False

    if not original_reused:
        delete_success, delete_message = await storage.delete(
            content_hash, store=resolve_store(store)
        )
        if not delete_success:
            return False, (
                f"Stored replacement chunks, but failed to delete "
                f"original memory: {delete_message}"
            ), None, False

    for chunk_memory in stored_chunks:
        await memory_service._run_post_store_steps(chunk_memory)
    return True, f"Stored as {len(stored_chunks)} chunks", stored_chunks[0].content_hash, original_reused


async def _broadcast_memory_update(
    storage: MemoryStorage, content_hash: str, new_hash: str, store: str,
    original_deleted: bool = True,
) -> Optional[Memory]:
    """Read the stored replacement and notify other dashboard clients."""
    updated_memory = await storage.get_by_hash(new_hash, store=resolve_store(store))
    if updated_memory is None:
        return None

    try:
        if original_deleted:
            old_event = create_memory_deleted_event(content_hash, True, store=store)
            await sse_manager.broadcast_event(old_event)
        new_event = create_memory_stored_event(
            {
                "content_hash": updated_memory.content_hash,
                "content": updated_memory.content,
                "tags": updated_memory.tags,
                "memory_type": updated_memory.memory_type,
            },
            store=store,
        )
        await sse_manager.broadcast_event(new_event)
    except Exception as e:
        logger.warning(
            "Failed to broadcast memory update events: %s",
            _sanitize_log_value(e),
        )
    return updated_memory


async def _update_memory_content(
    storage: MemoryStorage,
    memory_service: MemoryService,
    content_hash: str,
    existing_memory: Memory,
    request: MemoryUpdateRequest,
    store: str,
) -> MemoryUpdateResponse:
    """Update content through versioning or the unversioned replacement path."""
    original_deleted = True
    # Length check before choosing the versioned path: evolve_memory() inserts
    # the full text as one row, which backends with a field cap (Milvus limits
    # content to 65,535 bytes) reject. The destructive split fallback is only
    # valid on backends without version history.
    max_length = getattr(storage, "max_content_length", None)
    if ENABLE_AUTO_SPLIT and max_length and len(request.content) > max_length:
        if hasattr(storage, "update_memory_versioned"):
            return MemoryUpdateResponse(
                success=False,
                message=(
                    f"Content exceeds the {max_length}-character storage limit; "
                    "oversized edits are not supported on versioned storage"
                ),
                content_hash=content_hash,
                memory=memory_to_response(existing_memory),
            )
        replacement = _build_memory_replacement(existing_memory, request)
        success, message, new_hash, original_kept = await _store_unversioned_split_replacement(
            storage, memory_service, content_hash, store, replacement, max_length
        )
        original_deleted = not original_kept
        if success and not new_hash:
            new_hash = content_hash
    elif hasattr(storage, "update_memory_versioned"):
        version_store = resolve_store(store)
        success, message, new_hash = await memory_service.evolve_memory(
            content_hash,
            request.content,
            tags=request.tags,
            memory_type=request.memory_type,
            metadata=request.metadata,
            store=version_store,
        )
    else:
        replacement = _build_memory_replacement(existing_memory, request)
        success, message, new_hash, original_kept = await _store_unversioned_replacement(
            storage, memory_service, content_hash, store, replacement
        )
        original_deleted = not original_kept
        if success and not new_hash:
            new_hash = content_hash

    if success and new_hash:
        updated_memory = await _broadcast_memory_update(
            storage, content_hash, new_hash, store, original_deleted=original_deleted
        )
        return MemoryUpdateResponse(
            success=True,
            message=message,
            content_hash=new_hash,
            memory=memory_to_response(updated_memory) if updated_memory else None,
        )
    return MemoryUpdateResponse(
        success=False,
        message=message,
        content_hash=content_hash,
    )


async def _update_memory_metadata_fields(
    storage: MemoryStorage,
    content_hash: str,
    existing_memory: Memory,
    request: MemoryUpdateRequest,
    store: str,
) -> MemoryUpdateResponse:
    """Apply non-content fields without changing the memory hash."""
    updates = {}
    if request.tags is not None:
        updates["tags"] = request.tags
    if request.memory_type is not None:
        updates["memory_type"] = request.memory_type
    if request.metadata is not None:
        updates["metadata"] = request.metadata
    if not updates:
        return MemoryUpdateResponse(
            success=True,
            message="No updates provided - memory unchanged",
            content_hash=content_hash,
            memory=memory_to_response(existing_memory),
        )

    success, message = await storage.update_memory_metadata(
        content_hash=content_hash,
        updates=updates,
        preserve_timestamps=True,
        store=resolve_store(store),
    )
    if not success:
        return MemoryUpdateResponse(
            success=False,
            message=message,
            content_hash=content_hash,
        )

    updated_memory = await storage.get_by_hash(content_hash, store=resolve_store(store))
    return MemoryUpdateResponse(
        success=True,
        message=message,
        content_hash=content_hash,
        memory=memory_to_response(updated_memory) if updated_memory else None,
    )


@router.put("/memories/{content_hash}", response_model=MemoryUpdateResponse, tags=["memories"])
async def update_memory(
    content_hash: str,
    request: MemoryUpdateRequest,
    store: str = Query("default", description="Target store partition (default: 'default'). Use 'all' for every store."),
    storage: MemoryStorage = Depends(get_storage),
    memory_service: MemoryService = Depends(get_memory_service),
    user: AuthenticationResult = Depends(require_store_write)
):
    """Update a memory in place, or store changed content as its next version.

    Tags, type, and metadata updates preserve the content and creation timestamp.
    Changed content uses versioned updates when supported. Other backends store a
    replacement and delete the original without semantic deduplication.
    """
    if store == "all":
        raise HTTPException(
            status_code=400,
            detail="store='all' is only valid for read scopes",
        )

    try:
        # First, check that the memory exists inside the requested store scope
        existing_memory = await storage.get_by_hash(content_hash, store=resolve_store(store))
        if not existing_memory:
            raise HTTPException(status_code=404, detail=f"Memory with hash {content_hash} not found")

        if request.content is not None and request.content != existing_memory.content:
            return await _update_memory_content(
                storage, memory_service, content_hash, existing_memory, request, store
            )

        return await _update_memory_metadata_fields(
            storage, content_hash, existing_memory, request, store
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update memory: {str(e)}")


@router.get("/tags", response_model=TagListResponse, tags=["tags"])
async def get_tags(
    store: str = Query("default", description="Target store partition (default: 'default'). Use 'all' for every store."),
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_store_read)
):
    """
    Get all tags with their usage counts.

    Returns a list of all unique tags along with how many memories use each tag,
    sorted by count in descending order.
    """
    try:
        # Get tags with counts from storage. The storage-wide tag query is
        # useful for the legacy all-store view; scoped reads must derive counts
        # from the same filtered memory set as list/search.
        scope = resolve_store(store)
        if scope is None:
            tag_data = await storage.get_all_tags_with_counts()
        else:
            memories = await storage.get_all_memories(store=scope)
            tag_counts = Counter(
                tag
                for memory in memories
                for tag in (memory.tags or [])
                if tag
            )
            tag_data = [
                {"tag": tag, "count": count}
                for tag, count in sorted(tag_counts.items(), key=lambda item: (-item[1], item[0]))
            ]

        # Convert to response format
        tags = [TagResponse(tag=item["tag"], count=item["count"]) for item in tag_data]

        return TagListResponse(tags=tags)

    except AttributeError as e:
        # Handle case where storage backend doesn't implement get_all_tags_with_counts
        raise HTTPException(status_code=501, detail=f"Tags endpoint not supported by current storage backend: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tags: {str(e)}")


# ── Session ingestion ──────────────────────────────────────────────────────────

class SessionTurn(BaseModel):
    role: str = Field(..., description="Speaker role, e.g. 'user' or 'assistant'")
    content: str = Field(..., description="Turn content")


class SessionCreateRequest(BaseModel):
    turns: List[SessionTurn] = Field(..., min_length=1, description="Ordered conversation turns (at least one required)")
    session_id: Optional[str] = Field(None, description="Stable session identifier; auto-generated UUID if omitted")
    tags: List[str] = Field(default=[], description="Additional tags. 'session:<id>' is always added automatically.")
    metadata: Dict[str, Any] = Field(default={}, description="Optional extra metadata")
    store: str = Field("default", description="Target store partition (default: 'default'). Use 'all' only for read scopes.")


class SessionCreateResponse(BaseModel):
    success: bool
    message: str
    session_id: str
    content_hash: Optional[str] = None
    turn_count: int


@router.post("/sessions", response_model=SessionCreateResponse, tags=["memories"])
async def store_session(
    request: SessionCreateRequest,
    memory_service: MemoryService = Depends(get_memory_service),
    user: AuthenticationResult = Depends(get_current_user),
):
    """Store a conversation session as a single memory unit.

    All turns are concatenated into '[role] content' lines and stored as
    memory_type='session' with a 'session:<id>' tag for reliable session-level retrieval.
    """
    import uuid as _uuid  # inline import

    session_id = request.session_id or str(_uuid.uuid4())
    lines = [f"[{t.role}] {t.content.strip()}" for t in request.turns if t.content.strip()]
    if not lines:
        raise HTTPException(status_code=422, detail="All turns have empty content")

    if request.store == "all":
        raise HTTPException(
            status_code=400,
            detail="store='all' is only valid for read scopes",
        )

    # Session write is gated per-store (store arrives in the body).
    user.require_store_scope("write", request.store)

    content = "\n".join(lines)
    tags = [f"session:{session_id}"] + request.tags

    try:
        result = await memory_service.store_memory(
            content=content,
            tags=tags,
            memory_type="session",
            metadata=request.metadata,
            store=request.store,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage error: {str(e)}")

    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Storage error"))

    raw_memory = result.get("memory") or {}
    return SessionCreateResponse(
        success=True,
        message="Session stored successfully",
        session_id=session_id,
        content_hash=raw_memory.get("content_hash"),
        turn_count=len(lines),
    )


@router.get("/types")
async def get_memory_types(
    _auth: AuthenticationResult = Depends(require_read_access),
) -> List[str]:
    """Return all valid memory types (built-in + custom from MCP_CUSTOM_MEMORY_TYPES)."""
    return sorted(get_all_types())


# ── ResonanceDB — Wave-Based Semantic Search ──────────────────────────────────

class ResonanceSearchRequest(BaseModel):
    """Request model for ResonanceDB wave-based semantic search."""
    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=10, ge=1, le=100, description="Max results to return")
    mode: Literal["resonance", "phase", "hybrid"] = Field("hybrid", description="Search mode: 'resonance' (pure constructive interference), 'phase' (phase alignment only), 'hybrid' (default)")
    store: str = Field(default="default", description="Target store partition (default: 'default'). Use 'all' for every store.")


class ResonanceSearchResult(BaseModel):
    """Single result from ResonanceDB search."""
    content_hash: str
    score: float
    details: Dict[str, float]


class ResonanceSearchResponse(BaseModel):
    """Response model for ResonanceDB wave-based search."""
    query: str
    mode: str
    results: List[ResonanceSearchResult]
    total_stored: int
    backend: str = "resonance"


def _fetch_resonance_candidates(conn, store: Optional[str]) -> list:
    """Read (content_hash, embedding blob) for every current, searchable memory."""
    sql = """
        SELECT m.content_hash, e.content_embedding
        FROM memories m
        JOIN memory_embeddings e ON m.id = e.rowid
        WHERE m.deleted_at IS NULL
          AND (m.embedding_pending IS NULL OR m.embedding_pending = 0)
          AND (m.superseded_by IS NULL OR m.superseded_by = '')
    """
    params: tuple = ()
    if store is not None:
        sql += " AND m.store = ?"
        params = (store,)
    return conn.execute(sql, params).fetchall()


def _score_resonance_candidates(rows: list, query_embedding: List[float], dimension: int, top_k: int, mode: str):
    """Score all rows in batches and keep the overall top_k. Returns (results, scored_count)."""
    best: list = []
    scored = 0
    batch_size = 1000
    for start in range(0, len(rows), batch_size):
        db = ResonanceDB(dimension=dimension)
        for content_hash, blob in rows[start:start + batch_size]:
            values = deserialize_embedding(blob)
            if values is None or len(values) != dimension:
                continue
            db.add(content_hash, values)
        scored += len(db)
        best = heapq.nlargest(top_k, best + db.search(query_embedding, top_k=top_k, mode=mode), key=lambda r: r["score"])
    return best, scored


@router.post("/memories/resonance-search", response_model=ResonanceSearchResponse, tags=["memories", "resonance"])
async def resonance_search(
    request: ResonanceSearchRequest,
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access),
):
    """
    Wave-based semantic search using ResonanceDB (experimental).

    Converts the stored real-valued embeddings to complex waveforms
    ψ(x) = A(x) · exp(i · φ(x)) at query time and ranks them by
    constructive interference and phase alignment with the query.

    Requires a SQLite-backed storage (sqlite_vec, or hybrid via its local
    primary). Scans every eligible memory in the selected store.

    See: arXiv:2509.09691
    """
    # Hybrid keeps its SQLite backend in `primary`; Cloudflare and Milvus have no local table to scan.
    backend = getattr(storage, "primary", storage)
    if getattr(backend, "conn", None) is None or not hasattr(backend, "_execute_with_retry"):
        raise HTTPException(status_code=501, detail="Resonance search requires the sqlite_vec or hybrid storage backend.")
    if not getattr(backend, "embedding_model", None):
        raise HTTPException(status_code=503, detail="Resonance search requires an embedding model.")

    try:
        store = resolve_store(request.store)
        # ponytail: full scan in pure Python per request, O(n·dim); cache waveforms if stores grow past ~50k.
        rows = await backend._execute_with_retry(lambda: _fetch_resonance_candidates(backend.conn, store))
        query_embedding = await asyncio.to_thread(backend._generate_embedding, request.query)
        results, scored = await asyncio.to_thread(
            _score_resonance_candidates, rows, query_embedding, backend.embedding_dimension, request.top_k, request.mode
        )
    except Exception as e:
        logger.error("Resonance search failed: %s", _sanitize_log_value(e))
        raise HTTPException(status_code=500, detail="Resonance search failed")

    return ResonanceSearchResponse(
        query=request.query,
        mode=request.mode,
        results=[ResonanceSearchResult(**r) for r in results],
        total_stored=scored,
    )
