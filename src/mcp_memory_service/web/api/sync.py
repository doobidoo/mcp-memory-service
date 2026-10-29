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
Sync management endpoints for hybrid and event-log sync.

Provides delta-sync monitoring for any backend and hybrid sync controls.
"""

import time
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...storage.base import MemoryStorage
from ...storage.sync.apply import KNOWN_SCHEMA_VERSION
from ...storage.sync.protocol import (
    SYNC_PROTOCOL_MIN_VERSION,
    SYNC_PROTOCOL_VERSION,
    resolve_embedding_identity,
)
from ..dependencies import get_storage

# OAuth config no longer needed - auth is always enabled
# OAuth authentication imports
from ..oauth.middleware import (
    AuthenticationResult,
    require_read_access,
    require_write_access,
)

router = APIRouter()

try:
    from ... import __version__
except (ImportError, AttributeError):
    __version__ = "0.0.0.dev0"


class SyncCapabilitiesResponse(BaseModel):
    """Versions and storage capabilities required before remote sync operations."""

    sync_protocol: int = Field(description="Highest delta-sync wire protocol version supported")
    sync_protocol_min: int = Field(description="Lowest delta-sync wire protocol version supported")
    event_schema_version: int = Field(
        description="Event-envelope schema understood; independent of protocol versions"
    )
    embedding_model: str = Field(description="Embedding model used by the vector storage")
    embedding_dimension: int = Field(description="Vector dimension used by the storage")
    version: str = Field(description="Installed MCP Memory Service package version")
    store_scoped_hash_ops: bool = Field(
        default=False,
        description="Whether get, update, and delete by hash enforce the store parameter",
    )
    store_scoped_access: bool = Field(
        default=False,
        description="Whether all supported store-aware reads and writes are partitioned",
    )


@router.get("/sync/capabilities", response_model=SyncCapabilitiesResponse)
async def get_sync_capabilities(
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access),
):
    """Advertise delta-sync compatibility; this is not a general health endpoint."""
    embedding_model, embedding_dimension = resolve_embedding_identity(storage)
    return SyncCapabilitiesResponse(
        sync_protocol=SYNC_PROTOCOL_VERSION,
        sync_protocol_min=SYNC_PROTOCOL_MIN_VERSION,
        event_schema_version=KNOWN_SCHEMA_VERSION,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        version=__version__,
        store_scoped_hash_ops=bool(
            getattr(storage, "get_by_hash_is_store_scoped", False)
        ),
        store_scoped_access=bool(
            getattr(storage, "supports_store_scoped_access", False)
        ),
    )


class SyncStatusResponse(BaseModel):
    """Sync status response model."""
    is_hybrid: bool
    is_running: bool
    is_paused: bool
    last_sync_time: float
    operations_pending: int
    operations_processed: int  # lifetime success count
    operations_failed: int  # lifetime failure count (does not indicate current health)
    consecutive_failures: int = 0  # resets on any success — use this for current health
    sync_interval_seconds: int
    time_since_last_sync_seconds: float
    next_sync_eta_seconds: float
    status: str  # 'synced', 'syncing', 'pending', 'error' — reflects current health only
    delta_sync_peers: list[dict[str, Any]] = Field(default_factory=list)
    delta_sync: dict[str, Any] = Field(default_factory=dict)


def _get_delta_sync_status() -> dict[str, Any]:
    from ...api.client import get_scheduler
    from ...consolidation.scheduler import ConsolidationScheduler

    scheduler = get_scheduler()
    return (scheduler.get_delta_sync_status() if scheduler is not None
            else ConsolidationScheduler.delta_sync_configuration())


def _get_delta_sync_peer_status() -> list[dict[str, Any]]:
    """Return scheduler-owned peer status without coupling API startup to it."""
    from ...api.client import get_scheduler

    scheduler = get_scheduler()
    return scheduler.get_sync_peer_status() if scheduler is not None else []


class SyncForceResponse(BaseModel):
    """Force sync response model."""
    success: bool
    message: str
    operations_synced: int
    memories_pulled: int
    time_taken_seconds: float
    timestamp: str
    failed_operations: int = 0
    skipped_remote_tombstones: int = 0
    tombstones_applied: int = 0
    tombstone_conflicts: int = 0
    tombstone_failures: int = 0
    tombstone_conflict_hashes: list[str] = Field(default_factory=list)


class TombstoneConflictResolutionRequest(BaseModel):
    """Explicit resolution choice for one current sync conflict."""

    action: Literal["keep_local", "accept_delete"]


class TombstoneConflictResolutionResponse(BaseModel):
    """Outcome of a requested tombstone conflict resolution."""

    success: bool
    status: str
    action: Literal["keep_local", "accept_delete"]
    message: str


@router.get("/sync/status", response_model=SyncStatusResponse)
async def get_sync_status(
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access)
):
    """
    Get hybrid sync status and scheduled delta-sync peer results.

    Returns sync state, pending operations, last sync time, and health metrics.
    Delta-sync information is available for any storage backend. Hybrid control
    fields are inactive when the storage does not support hybrid sync.
    """
    # Check if storage supports sync (hybrid mode only)
    if not hasattr(storage, 'get_sync_status'):
        return SyncStatusResponse(
            is_hybrid=False,
            is_running=False,
            is_paused=False,
            last_sync_time=0,
            operations_pending=0,
            operations_processed=0,
            operations_failed=0,
            sync_interval_seconds=0,
            time_since_last_sync_seconds=0,
            next_sync_eta_seconds=0,
            status='not_hybrid',
            delta_sync_peers=_get_delta_sync_peer_status(),
            delta_sync=_get_delta_sync_status(),
        )

    try:
        # Get sync status from hybrid backend
        sync_status = await storage.get_sync_status()

        # Calculate time since last sync
        current_time = time.time()
        last_sync = sync_status.get('last_sync_time', 0)
        time_since_sync = current_time - last_sync if last_sync > 0 else 0

        # Calculate ETA for next sync
        sync_interval = sync_status.get('sync_interval', 300)
        next_sync_eta = max(0, sync_interval - time_since_sync)

        # Determine status — reflects CURRENT health only. Lifetime `operations_failed`
        # is not used here because a single historical failure would then pin `status`
        # to 'error' forever, masking real-time sync health (see issue #750).
        is_running = sync_status.get('is_running', False)
        pending_ops = sync_status.get('pending_operations', 0)
        actively_syncing = sync_status.get('actively_syncing', False)  # True only during active sync
        consecutive_failures = sync_status.get('consecutive_failures', 0)

        if actively_syncing:
            status = 'syncing'
        elif pending_ops > 0:
            status = 'pending'
        elif consecutive_failures > 0:
            status = 'error'
        else:
            status = 'synced'

        return SyncStatusResponse(
            is_hybrid=True,
            is_running=is_running,
            is_paused=sync_status.get('is_paused', not is_running),
            last_sync_time=last_sync,
            operations_pending=pending_ops,
            operations_processed=sync_status.get('operations_processed', 0),
            operations_failed=sync_status.get('operations_failed', 0),
            consecutive_failures=consecutive_failures,
            sync_interval_seconds=sync_interval,
            time_since_last_sync_seconds=time_since_sync,
            next_sync_eta_seconds=next_sync_eta,
            status=status,
            delta_sync_peers=_get_delta_sync_peer_status(),
            delta_sync=_get_delta_sync_status(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sync status: {str(e)}")


@router.post("/sync/force", response_model=SyncForceResponse)
async def force_sync(
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_write_access)
):
    """
    Manually trigger immediate bi-directional sync with Cloudflare.

    Performs BOTH directions:
    1. PULL: Download new memories FROM Cloudflare TO local SQLite
    2. PUSH: Upload pending operations FROM local TO Cloudflare

    This ensures complete synchronization between both backends.
    Only available when using hybrid storage backend.
    """
    # Check if storage supports force sync (hybrid mode only)
    if not hasattr(storage, 'force_sync'):
        raise HTTPException(
            status_code=404,
            detail="Manual sync only available in hybrid mode"
        )

    try:
        start_time = time.time()
        pull_result = None
        if hasattr(storage, 'force_pull_sync'):
            pull_result = await storage.force_pull_sync()
        push_result = await storage.force_sync()
        return _build_force_sync_response(
            pull_result,
            push_result,
            time.time() - start_time,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to force sync: {str(e)}"
        )


def _force_sync_metrics(
    pull_result: dict[str, Any] | None, push_result: dict[str, Any]
) -> dict[str, Any]:
    """Combine pull/push outcomes without double-counting shared conflicts."""
    pull_result = pull_result or {}
    failed_operations = push_result.get('failed_operations', 0)
    push_success = (
        push_result.get('status') == 'completed' and failed_operations == 0
        if 'status' in push_result
        else push_result.get('success', False) and failed_operations == 0
    )
    tombstone_conflicts = max(
        pull_result.get('tombstone_conflicts', 0),
        push_result.get('tombstone_conflicts', 0),
    )
    tombstone_failures = max(
        pull_result.get('tombstone_failures', 0),
        push_result.get('tombstone_failures', 0),
    )
    return {
        'memories_pulled': pull_result.get('memories_pulled', 0),
        'operations_synced': push_result.get(
            'synced_to_secondary', push_result.get('operations_synced', 0)
        ),
        'skipped_already_present': push_result.get('skipped_already_present', 0),
        'failed_operations': failed_operations,
        'skipped_remote_tombstones': push_result.get(
            'skipped_remote_tombstones', 0
        ),
        'tombstones_applied': pull_result.get('tombstones_applied', 0)
        + push_result.get('tombstones_applied', 0),
        'tombstone_conflicts': tombstone_conflicts,
        'tombstone_failures': tombstone_failures,
        'tombstone_conflict_hashes': list(dict.fromkeys(
            pull_result.get('tombstone_conflict_hashes', [])
            + push_result.get('tombstone_conflict_hashes', [])
        ))[:100],
        'pull_success': pull_result.get('success', True),
        'push_success': push_success,
    }


def _force_sync_transfer_message(pulled: int, pushed: int) -> str | None:
    """Describe successful transfers, if either direction moved memories."""
    if pulled > 0 and pushed > 0:
        return f"Pulled {pulled} from Cloudflare, pushed {pushed} to Cloudflare"
    if pulled > 0:
        return f"Pulled {pulled} from Cloudflare"
    if pushed > 0:
        return f"Pushed {pushed} to Cloudflare"
    return None


def _force_sync_base_message(metrics: dict[str, Any]) -> str:
    """Choose the primary force-sync message from its aggregate metrics."""
    transfer_message = _force_sync_transfer_message(
        metrics['memories_pulled'], metrics['operations_synced']
    )
    if transfer_message:
        return transfer_message
    if metrics['tombstones_applied'] > 0:
        return f"Applied {metrics['tombstones_applied']} remote deletion(s) locally"
    if metrics['skipped_remote_tombstones'] > 0:
        return (
            "No memories uploaded ("
            f"{metrics['skipped_remote_tombstones']} remote tombstone(s) "
            "protected from resurrection)"
        )
    if metrics['tombstone_conflicts'] or metrics['tombstone_failures']:
        return "Sync incomplete; tombstone reconciliation needs attention"
    if metrics['skipped_already_present'] > 0:
        return f"No changes to sync ({metrics['skipped_already_present']} already synchronized)"
    return "No changes to sync (already synchronized)"


def _force_sync_message_details(
    metrics: dict[str, Any],
    pull_result: dict[str, Any] | None,
    push_result: dict[str, Any],
) -> list[str]:
    """Collect failure and conflict details from pull and push outcomes."""
    pull_result = pull_result or {}
    operation_details = _force_sync_operation_details(
        metrics, pull_result, push_result
    )
    conflict_details = _force_sync_tombstone_details(
        metrics, pull_result.get('message', '').lower()
    )
    return operation_details + conflict_details


def _force_sync_operation_details(
    metrics: dict[str, Any],
    pull_result: dict[str, Any],
    push_result: dict[str, Any],
) -> list[str]:
    """Collect pull/push errors and failed-operation details."""
    details = []
    if not metrics['pull_success']:
        pull_message = pull_result.get('message')
        if pull_message:
            details.append(pull_message)
    if not metrics['push_success']:
        push_error = push_result.get('error')
        if push_error:
            details.append(push_error)
    if metrics['failed_operations']:
        details.append(
            f"{metrics['failed_operations']} push operation(s) failed"
        )

    return details


def _force_sync_tombstone_details(
    metrics: dict[str, Any], pull_message: str
) -> list[str]:
    """Describe reconciliation conflicts not already explained by the pull."""
    details = []
    if metrics['tombstone_conflicts'] and 'conflict' not in pull_message:
        details.append(
            f"preserved {metrics['tombstone_conflicts']} newer local edit(s) "
            "with tombstone conflicts"
        )
    if (
        metrics['tombstone_failures']
        and 'tombstone reconciliation incomplete' not in pull_message
    ):
        details.append(
            "tombstone reconciliation incomplete "
            f"({metrics['tombstone_failures']} failure(s))"
        )
    return details


def _force_sync_message(
    base_message: str,
    metrics: dict[str, Any],
    pull_result: dict[str, Any] | None,
    push_result: dict[str, Any],
) -> str:
    """Add actionable failure/conflict details to the primary sync message."""
    details = _force_sync_message_details(metrics, pull_result, push_result)
    if details:
        return f"{base_message}; {'; '.join(details)}"
    return base_message


def _build_force_sync_response(
    pull_result: dict[str, Any] | None,
    push_result: dict[str, Any],
    time_taken: float,
) -> SyncForceResponse:
    """Build the API response from pull and push results."""
    metrics = _force_sync_metrics(pull_result, push_result)
    overall_success = (
        metrics['pull_success']
        and metrics['push_success']
        and metrics['tombstone_conflicts'] == 0
        and metrics['tombstone_failures'] == 0
    )
    message = _force_sync_message(
        _force_sync_base_message(metrics), metrics, pull_result, push_result
    )
    return SyncForceResponse(
        success=overall_success,
        message=message,
        operations_synced=metrics['operations_synced'],
        memories_pulled=metrics['memories_pulled'],
        time_taken_seconds=round(time_taken, 3),
        timestamp=datetime.now(timezone.utc).isoformat(),
        failed_operations=metrics['failed_operations'],
        skipped_remote_tombstones=metrics['skipped_remote_tombstones'],
        tombstones_applied=metrics['tombstones_applied'],
        tombstone_conflicts=metrics['tombstone_conflicts'],
        tombstone_failures=metrics['tombstone_failures'],
        tombstone_conflict_hashes=metrics['tombstone_conflict_hashes'],
    )


@router.post(
    "/sync/tombstone-conflicts/{content_hash}",
    response_model=TombstoneConflictResolutionResponse,
)
async def resolve_tombstone_conflict(
    content_hash: str,
    payload: TombstoneConflictResolutionRequest,
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_write_access),
) -> TombstoneConflictResolutionResponse:
    """Resolve one currently conflicting remote deletion explicitly."""
    resolver = getattr(storage, "resolve_tombstone_conflict", None)
    if (
        getattr(storage, "secondary_backend_type", None) != "cloudflare"
        or resolver is None
    ):
        raise HTTPException(
            status_code=404,
            detail="Tombstone conflict resolution is only available with a Cloudflare secondary",
        )

    try:
        result = await resolver(content_hash, payload.action)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to resolve tombstone conflict safely",
        ) from exc
    return TombstoneConflictResolutionResponse(**result)


class SyncPauseResponse(BaseModel):
    """Pause/resume sync response model."""
    success: bool
    message: str
    is_paused: bool
    timestamp: str


@router.post("/sync/pause", response_model=SyncPauseResponse)
async def pause_sync(
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_write_access)
):
    """
    Pause background sync operations.

    Pauses the background sync service to allow safe database operations.
    Sync will resume when resume_sync is called.
    Only available when using hybrid storage backend.
    """
    # Check if storage supports pause/resume (hybrid mode only)
    if not hasattr(storage, 'pause_sync'):
        raise HTTPException(
            status_code=404,
            detail="Pause sync only available in hybrid mode"
        )

    try:
        result = await storage.pause_sync()

        return SyncPauseResponse(
            success=result.get('success', True),
            message=result.get('message', 'Sync paused'),
            is_paused=True,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to pause sync: {str(e)}"
        )


@router.post("/sync/resume", response_model=SyncPauseResponse)
async def resume_sync(
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_write_access)
):
    """
    Resume background sync operations.

    Resumes the background sync service after it was paused.
    Only available when using hybrid storage backend.
    """
    # Check if storage supports pause/resume (hybrid mode only)
    if not hasattr(storage, 'resume_sync'):
        raise HTTPException(
            status_code=404,
            detail="Resume sync only available in hybrid mode"
        )

    try:
        result = await storage.resume_sync()

        return SyncPauseResponse(
            success=result.get('success', True),
            message=result.get('message', 'Sync resumed'),
            is_paused=False,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to resume sync: {str(e)}"
        )
