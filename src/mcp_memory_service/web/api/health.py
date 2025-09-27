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
from typing import Dict, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ...storage.sqlite_vec import SqliteVecMemoryStorage
from ..dependencies import get_storage
from ... import __version__
from ...config import OAUTH_ENABLED

# Import OAuth dependencies only when needed
if OAUTH_ENABLED:
    from ..oauth.middleware import require_read_access, AuthenticationResult

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
    storage: SqliteVecMemoryStorage = Depends(get_storage),
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
    
    # Get storage information
    try:
        # Get basic storage stats
        storage_info = {
            "backend": "sqlite-vec",
            "database_path": storage.db_path,
            "embedding_model": storage.embedding_model_name,
            "status": "connected"
        }
        
        # Try to get detailed statistics from storage
        try:
            stats = storage.get_stats()
            if "error" not in stats:
                storage_info.update(stats)
                storage_info["accessible"] = True
            else:
                storage_info["accessible"] = False
                storage_info["stats_error"] = stats["error"]
        except Exception as e:
            storage_info["accessible"] = False
            storage_info["error"] = str(e)
            
    except Exception as e:
        storage_info = {
            "backend": "sqlite-vec",
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