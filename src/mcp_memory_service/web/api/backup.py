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
Backup management endpoints for MCP Memory Service.

Provides status monitoring, manual backup triggering, and backup listing.
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from ...config import OAUTH_ENABLED
from ...backup.scheduler import get_backup_service, get_backup_scheduler

# OAuth authentication imports (conditional)
if OAUTH_ENABLED or TYPE_CHECKING:
    from ..oauth.middleware import require_read_access, require_write_access, AuthenticationResult
else:
    # Provide type stubs when OAuth is disabled
    AuthenticationResult = None
    require_read_access = None
    require_write_access = None

router = APIRouter()


class BackupStatusResponse(BaseModel):
    """Backup status response model."""
    enabled: bool
    interval: str
    retention_days: int
    max_count: int
    backup_count: int
    total_size_bytes: int
    last_backup_time: Optional[float]
    time_since_last_seconds: Optional[float]
    next_backup_at: Optional[str]
    scheduler_running: bool


class BackupCreateResponse(BaseModel):
    """Backup creation response model."""
    success: bool
    filename: Optional[str] = None
    size_bytes: Optional[int] = None
    created_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None


class BackupInfo(BaseModel):
    """Backup information model."""
    filename: str
    size_bytes: int
    created_at: str
    age_days: int


class BackupListResponse(BaseModel):
    """Backup list response model."""
    backups: List[BackupInfo]
    total_count: int
    total_size_bytes: int


@router.get("/backup/status", response_model=BackupStatusResponse)
async def get_backup_status(
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None
):
    """
    Get current backup service status.

    Returns backup configuration, last backup time, and next scheduled backup.
    """
    try:
        scheduler = get_backup_scheduler()
        status = scheduler.get_status()

        return BackupStatusResponse(
            enabled=status.get('enabled', False),
            interval=status.get('interval', 'daily'),
            retention_days=status.get('retention_days', 7),
            max_count=status.get('max_count', 10),
            backup_count=status.get('backup_count', 0),
            total_size_bytes=status.get('total_size_bytes', 0),
            last_backup_time=status.get('last_backup_time'),
            time_since_last_seconds=status.get('time_since_last_seconds'),
            next_backup_at=status.get('next_backup_at'),
            scheduler_running=status.get('scheduler_running', False)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get backup status: {str(e)}")


@router.post("/backup/now", response_model=BackupCreateResponse)
async def trigger_backup(
    user: AuthenticationResult = Depends(require_write_access) if OAUTH_ENABLED else None
):
    """
    Manually trigger an immediate backup.

    Creates a new backup of the database regardless of the schedule.
    """
    try:
        backup_service = get_backup_service()
        result = await backup_service.create_backup(description="Manual backup from dashboard")

        if result.get('success'):
            return BackupCreateResponse(
                success=True,
                filename=result.get('filename'),
                size_bytes=result.get('size_bytes'),
                created_at=result.get('created_at'),
                duration_seconds=result.get('duration_seconds')
            )
        else:
            return BackupCreateResponse(
                success=False,
                error=result.get('error', 'Unknown error')
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create backup: {str(e)}")


@router.get("/backup/list", response_model=BackupListResponse)
async def list_backups(
    user: AuthenticationResult = Depends(require_read_access) if OAUTH_ENABLED else None
):
    """
    List all available backups.

    Returns list of backups sorted by date (newest first).
    """
    try:
        backup_service = get_backup_service()
        backups = backup_service.list_backups()

        backup_infos = [
            BackupInfo(
                filename=b['filename'],
                size_bytes=b['size_bytes'],
                created_at=b['created_at'],
                age_days=b['age_days']
            )
            for b in backups
        ]

        total_size = sum(b['size_bytes'] for b in backups)

        return BackupListResponse(
            backups=backup_infos,
            total_count=len(backup_infos),
            total_size_bytes=total_size
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list backups: {str(e)}")
