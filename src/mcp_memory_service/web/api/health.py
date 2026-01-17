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
Health check endpoints for the HTTP interface.
"""

import time
import platform
import psutil
from datetime import datetime, timezone
from typing import Dict, Any, TYPE_CHECKING

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ...storage.base import MemoryStorage
from ..dependencies import get_storage

try:
    from ... import __version__
except (ImportError, AttributeError):
    __version__ = "0.0.0.dev0"

from ...config import OAUTH_ENABLED

# OAuth authentication imports (conditional)
if OAUTH_ENABLED or TYPE_CHECKING:
    from ..oauth.middleware import require_read_access, AuthenticationResult
else:
    # Provide type stubs when OAuth is disabled
    AuthenticationResult = None
    require_read_access = None

router = APIRouter()


class HealthResponse(BaseModel):
    """Basic health check response."""
    status: str
    version: str
    timestamp: str
    uptime_seconds: float


class DetailedHealthResponse(BaseModel):
    """Detailed health check response."""
    status: str
    version: str
    timestamp: str
    uptime_seconds: float
    storage: Dict[str, Any]
    system: Dict[str, Any]
    performance: Dict[str, Any]
    statistics: Dict[str, Any] = None


# Track startup time for uptime calculation
_startup_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.now(timezone.utc).isoformat(),
        uptime_seconds=time.time() - _startup_time
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check(
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None
):
    """Detailed health check with system and storage information."""
    
    # Get system information
    memory_info = psutil.virtual_memory()
    disk_info = psutil.disk_usage('/')
    
    system_info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": round(memory_info.total / (1024**3), 2),
        "memory_available_gb": round(memory_info.available / (1024**3), 2),
        "memory_percent": memory_info.percent,
        "disk_total_gb": round(disk_info.total / (1024**3), 2),
        "disk_free_gb": round(disk_info.free / (1024**3), 2),
        "disk_percent": round((disk_info.used / disk_info.total) * 100, 2)
    }
    
    # Get storage information (support all storage backends)
    try:
        # Get statistics from storage using universal get_stats() method
        if hasattr(storage, 'get_stats') and callable(getattr(storage, 'get_stats')):
            # All storage backends now have async get_stats()
            stats = await storage.get_stats()
        else:
            stats = {"error": "Storage backend doesn't support statistics"}

        if "error" not in stats:
            # Detect backend type from storage class or stats
            backend_name = stats.get("storage_backend", storage.__class__.__name__)
            if "sqlite" in backend_name.lower():
                backend_type = "sqlite-vec"
            elif "cloudflare" in backend_name.lower():
                backend_type = "cloudflare"
            elif "hybrid" in backend_name.lower():
                backend_type = "hybrid"
            else:
                backend_type = backend_name

            storage_info = {
                "backend": backend_type,
                "status": "connected",
                "accessible": True
            }

            # Add backend-specific information if available
            if hasattr(storage, 'db_path'):
                storage_info["database_path"] = storage.db_path
            if hasattr(storage, 'embedding_model_name'):
                storage_info["embedding_model"] = storage.embedding_model_name

            # Add sync status for hybrid backend
            if backend_type == "hybrid" and hasattr(storage, 'get_sync_status'):
                try:
                    sync_status = await storage.get_sync_status()
                    storage_info["sync_status"] = {
                        "is_running": sync_status.get('is_running', False),
                        "last_sync_time": sync_status.get('last_sync_time', 0),
                        "pending_operations": sync_status.get('pending_operations', 0),
                        "operations_processed": sync_status.get('operations_processed', 0),
                        "operations_failed": sync_status.get('operations_failed', 0),
                        "time_since_last_sync": time.time() - sync_status.get('last_sync_time', 0) if sync_status.get('last_sync_time', 0) > 0 else 0
                    }
                except Exception as sync_err:
                    storage_info["sync_status"] = {"error": str(sync_err)}

            # Merge all stats
            storage_info.update(stats)
        else:
            storage_info = {
                "backend": storage.__class__.__name__,
                "status": "error",
                "accessible": False,
                "error": stats["error"]
            }

    except Exception as e:
        storage_info = {
            "backend": storage.__class__.__name__ if hasattr(storage, '__class__') else "unknown",
            "status": "error",
            "error": str(e)
        }
    
    # Performance metrics (basic for now)
    performance_info = {
        "uptime_seconds": time.time() - _startup_time,
        "uptime_formatted": format_uptime(time.time() - _startup_time)
    }
    
    # Extract statistics for separate field if available
    statistics = {
        "total_memories": storage_info.get("total_memories", 0),
        "unique_tags": storage_info.get("unique_tags", 0),
        "memories_this_week": storage_info.get("memories_this_week", 0),
        "database_size_mb": storage_info.get("database_size_mb", 0),
        "backend": storage_info.get("backend", "sqlite-vec")
    }
    
    return DetailedHealthResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.now(timezone.utc).isoformat(),
        uptime_seconds=time.time() - _startup_time,
        storage=storage_info,
        system=system_info,
        performance=performance_info,
        statistics=statistics
    )


@router.get("/health/sync-status")
async def sync_status(
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None
):
    """Get current initial sync status for hybrid storage."""

    # Check if this is a hybrid storage that supports sync status
    if hasattr(storage, 'get_initial_sync_status'):
        sync_status = storage.get_initial_sync_status()
        return {
            "sync_supported": True,
            "status": sync_status
        }
    else:
        return {
            "sync_supported": False,
            "status": {
                "in_progress": False,
                "total": 0,
                "completed": 0,
                "finished": True,
                "progress_percentage": 100
            }
        }


def format_uptime(seconds: float) -> str:
    """Format uptime in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    elif seconds < 86400:
        return f"{seconds/3600:.1f} hours"
    else:
        return f"{seconds/86400:.1f} days"


# =============================================================================
# MEMORY MANAGEMENT ENDPOINTS (v8.71.0 - Discussion #331)
# =============================================================================

class MemoryStatsResponse(BaseModel):
    """Response model for memory statistics."""
    process_memory_mb: float
    process_memory_virtual_mb: float
    system_memory_total_gb: float
    system_memory_available_gb: float
    system_memory_percent: float
    cached_storage_count: int
    cached_service_count: int
    model_cache_count: int
    embedding_cache_count: int
    cache_stats: Dict[str, Any]


class ClearCachesResponse(BaseModel):
    """Response model for cache clearing operation."""
    success: bool
    storage_instances_cleared: int
    service_instances_cleared: int
    models_cleared: int
    embeddings_cleared: int
    gc_collected: int
    memory_freed_estimate_mb: float


@router.get("/memory-stats", response_model=MemoryStatsResponse)
async def get_memory_stats(
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None
):
    """
    Get detailed memory usage statistics for the process.

    This endpoint provides insights into:
    - Process memory usage (RSS and virtual)
    - System memory status
    - Cache counts (storage, service, model, embedding)
    - Cache hit/miss statistics

    Useful for monitoring memory pressure and diagnosing memory issues.
    """
    from ...server.cache_manager import get_memory_usage, get_cache_stats
    from ...storage.sqlite_vec import get_model_cache_stats

    # Get process memory info
    process_memory = get_memory_usage()

    # Get system memory info
    memory_info = psutil.virtual_memory()

    # Get model cache stats
    model_stats = get_model_cache_stats()

    # Get cache stats
    cache_stats = get_cache_stats()

    return MemoryStatsResponse(
        process_memory_mb=process_memory["rss_mb"],
        process_memory_virtual_mb=process_memory["vms_mb"],
        system_memory_total_gb=round(memory_info.total / (1024**3), 2),
        system_memory_available_gb=round(memory_info.available / (1024**3), 2),
        system_memory_percent=memory_info.percent,
        cached_storage_count=process_memory["cached_storage_count"],
        cached_service_count=process_memory["cached_service_count"],
        model_cache_count=model_stats["model_count"],
        embedding_cache_count=model_stats["embedding_count"],
        cache_stats=cache_stats
    )


@router.post("/clear-caches", response_model=ClearCachesResponse)
async def clear_caches(
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None
):
    """
    Clear all caches to free memory.

    WARNING: This is a destructive operation that will:
    - Clear all storage backend caches
    - Clear all MemoryService caches
    - Clear all embedding model caches
    - Clear all computed embedding caches
    - Run garbage collection

    After clearing:
    - Next requests will be slower (cold start)
    - Models will be reloaded on demand
    - Storage connections will be re-established

    Use this endpoint when:
    - Memory pressure is high
    - Debugging memory issues
    - After long sessions to reclaim memory
    """
    from ...server.cache_manager import clear_all_caches, get_memory_usage
    from ...storage.sqlite_vec import clear_model_caches

    # Get memory usage before clearing
    memory_before = get_memory_usage()
    rss_before = memory_before["rss_mb"]

    # Clear service and storage caches
    cache_stats = clear_all_caches()

    # Clear model caches
    model_stats = clear_model_caches()

    # Get memory usage after clearing
    memory_after = get_memory_usage()
    rss_after = memory_after["rss_mb"]

    # Calculate estimated freed memory
    memory_freed = max(0, rss_before - rss_after)

    return ClearCachesResponse(
        success=True,
        storage_instances_cleared=cache_stats["storage_instances_cleared"],
        service_instances_cleared=cache_stats["service_instances_cleared"],
        models_cleared=model_stats["models_cleared"],
        embeddings_cleared=model_stats["embeddings_cleared"],
        gc_collected=model_stats["gc_collected"],
        memory_freed_estimate_mb=round(memory_freed, 2)
    )