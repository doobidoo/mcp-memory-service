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

import platform
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import psutil
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ... import __version__
from ...config import OAUTH_ENABLED
from ...storage.base import MemoryStorage
from ..dependencies import get_storage
from ..write_queue import write_queue

# Try importing QdrantStorage for type checking
try:
    from ...storage.qdrant_storage import QdrantStorage
except ImportError:
    QdrantStorage = None

# OAuth authentication imports (conditional)
if OAUTH_ENABLED or TYPE_CHECKING:
    from ..oauth.middleware import AuthenticationResult, require_read_access
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
    write_queue: dict[str, Any] = None


class DetailedHealthResponse(BaseModel):
    """Detailed health check response."""

    status: str
    version: str
    timestamp: str
    uptime_seconds: float
    storage: dict[str, Any]
    system: dict[str, Any]
    performance: dict[str, Any]
    statistics: dict[str, Any] = None


# Track startup time for uptime calculation
_startup_time = time.time()


@router.get("/health")
async def health_check(storage: MemoryStorage = Depends(get_storage)):
    """
    Basic health check endpoint with write queue statistics and Qdrant-specific checks.

    Returns 200 OK if healthy, 503 Service Unavailable if unhealthy.
    """
    import logging

    from fastapi.responses import JSONResponse

    logger = logging.getLogger(__name__)

    # Check if storage is Qdrant backend
    is_qdrant = QdrantStorage is not None and isinstance(storage, QdrantStorage)

    if is_qdrant:
        try:
            # Test actual connectivity
            collections_response = storage.client.get_collections()

            # Check circuit breaker status
            circuit_status = "closed"
            if hasattr(storage, "_circuit_open_until") and storage._circuit_open_until:
                circuit_status = f"open_until_{storage._circuit_open_until.isoformat()}"

            # Get failure count
            failure_count = getattr(storage, "_failure_count", 0)

            # If circuit breaker is open, service is unhealthy
            if circuit_status != "closed":
                logger.error(f"Health check failed: Qdrant circuit breaker is {circuit_status}")
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "unhealthy",
                        "version": __version__,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "uptime_seconds": time.time() - _startup_time,
                        "backend": "qdrant",
                        "circuit_breaker": circuit_status,
                        "failure_count": failure_count,
                        "error": "Qdrant circuit breaker is open - service unavailable",
                    },
                )

            # Healthy response with Qdrant details
            return {
                "status": "healthy",
                "version": __version__,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptime_seconds": time.time() - _startup_time,
                "write_queue": write_queue.get_stats(),
                "backend": "qdrant",
                "circuit_breaker": circuit_status,
                "failure_count": failure_count,
                "qdrant_collections": [col.name for col in collections_response.collections],
            }

        except Exception as e:
            logger.error(f"Health check failed: Qdrant connectivity error: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "version": __version__,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "uptime_seconds": time.time() - _startup_time,
                    "backend": "qdrant",
                    "error": str(e),
                    "message": "Qdrant is unavailable - service cannot function. Check logs and fix configuration.",
                },
            )

    # Non-Qdrant backends: standard response
    return HealthResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.now(timezone.utc).isoformat(),
        uptime_seconds=time.time() - _startup_time,
        write_queue=write_queue.get_stats(),
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check(
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None,
):
    """Detailed health check with system and storage information."""

    # Get system information
    memory_info = psutil.virtual_memory()
    disk_info = psutil.disk_usage("/")

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
        "disk_percent": round((disk_info.used / disk_info.total) * 100, 2),
    }

    # Get storage information (support all storage backends)
    try:
        # Get statistics from storage using universal get_stats() method
        if hasattr(storage, "get_stats") and callable(storage.get_stats):
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

            storage_info = {"backend": backend_type, "status": "connected", "accessible": True}

            # Add backend-specific information if available
            if hasattr(storage, "db_path"):
                storage_info["database_path"] = storage.db_path
            if hasattr(storage, "embedding_model_name"):
                storage_info["embedding_model"] = storage.embedding_model_name

            # Add sync status for hybrid backend
            if backend_type == "hybrid" and hasattr(storage, "get_sync_status"):
                try:
                    sync_status = await storage.get_sync_status()
                    storage_info["sync_status"] = {
                        "is_running": sync_status.get("is_running", False),
                        "last_sync_time": sync_status.get("last_sync_time", 0),
                        "pending_operations": sync_status.get("pending_operations", 0),
                        "operations_processed": sync_status.get("operations_processed", 0),
                        "operations_failed": sync_status.get("operations_failed", 0),
                        "time_since_last_sync": time.time() - sync_status.get("last_sync_time", 0)
                        if sync_status.get("last_sync_time", 0) > 0
                        else 0,
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
                "error": stats["error"],
            }

    except Exception as e:
        storage_info = {
            "backend": storage.__class__.__name__ if hasattr(storage, "__class__") else "unknown",
            "status": "error",
            "error": str(e),
        }

    # Performance metrics (basic for now)
    performance_info = {
        "uptime_seconds": time.time() - _startup_time,
        "uptime_formatted": format_uptime(time.time() - _startup_time),
    }

    # Extract statistics for separate field if available
    statistics = {
        "total_memories": storage_info.get("total_memories", 0),
        "unique_tags": storage_info.get("unique_tags", 0),
        "memories_this_week": storage_info.get("memories_this_week", 0),
        "database_size_mb": storage_info.get("database_size_mb", 0),
        "backend": storage_info.get("backend", "sqlite-vec"),
    }

    return DetailedHealthResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.now(timezone.utc).isoformat(),
        uptime_seconds=time.time() - _startup_time,
        storage=storage_info,
        system=system_info,
        performance=performance_info,
        statistics=statistics,
    )


@router.get("/health/sync-status")
async def sync_status(
    storage: MemoryStorage = Depends(get_storage),
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None,
):
    """Get current initial sync status for hybrid storage."""

    # Check if this is a hybrid storage that supports sync status
    if hasattr(storage, "get_initial_sync_status"):
        sync_status = storage.get_initial_sync_status()
        return {"sync_supported": True, "status": sync_status}
    else:
        return {
            "sync_supported": False,
            "status": {"in_progress": False, "total": 0, "completed": 0, "finished": True, "progress_percentage": 100},
        }


def format_uptime(seconds: float) -> str:
    """Format uptime in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} minutes"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f} hours"
    else:
        return f"{seconds / 86400:.1f} days"
