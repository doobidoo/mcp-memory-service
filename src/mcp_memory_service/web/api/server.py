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
Server management endpoints for the HTTP interface.

Provides server status, version checking, updates, and restart capabilities.
"""

import os
import sys
import time
import asyncio
import logging
import platform
import subprocess
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

try:
    from ... import __version__
except (ImportError, AttributeError):
    __version__ = "0.0.0.dev0"

# OAuth authentication imports
from ..oauth.middleware import require_read_access, require_admin_access, AuthenticationResult

router = APIRouter()
logger = logging.getLogger(__name__)

# Constants
RESTART_DELAY_SECONDS = 3
GIT_TIMEOUT_SECONDS = 30
PIP_TIMEOUT_SECONDS = 300

# Track startup time for uptime calculation
_startup_time = time.time()


# Request/Response Models
class ServerStatusResponse(BaseModel):
    """Response model for server status."""
    version: str
    uptime_seconds: float
    uptime_formatted: str
    pid: int
    platform: str
    platform_version: str
    python_version: str
    timestamp: str


class VersionCheckResponse(BaseModel):
    """Response model for version check."""
    current_version: str
    commits_behind: int
    update_available: bool
    latest_commits: List[str]
    check_timestamp: str
    git_available: bool
    error: Optional[str] = None


class RestartRequest(BaseModel):
    """Request model for server restart."""
    confirm: bool = Field(..., description="Must be true to confirm restart")


class RestartResponse(BaseModel):
    """Response model for server restart."""
    status: str
    message: str
    restart_in_seconds: int


class UpdateRequest(BaseModel):
    """Request model for server update."""
    confirm: bool = Field(..., description="Must be true to confirm update")


class UpdateResponse(BaseModel):
    """Response model for server update."""
    status: str
    message: str
    git_output: Optional[str] = None
    pip_output: Optional[str] = None
    restart_scheduled: bool


# Helper functions
def _get_project_root() -> str:
    """Get the project root directory by searching for the .git folder."""
    from pathlib import Path

    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / ".git").is_dir():
            return str(parent)
    raise ValueError("Not a valid git repository: project root with .git folder not found.")


def _run_command(command: List[str], timeout: int, command_name: str) -> Tuple[str, bool]:
    """
    Run a command and return output.

    Returns:
        Tuple of (output_string, success_boolean)
    """
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=_get_project_root()
        )
        return result.stdout.strip(), result.returncode == 0
    except FileNotFoundError:
        return f"{command_name} command not found. Please install {command_name.lower()}.", False
    except subprocess.TimeoutExpired:
        return f"{command_name} command timed out after {timeout} seconds", False
    except Exception as e:
        return f"{command_name} command failed: {str(e)}", False


def _run_git_command(args: List[str], timeout: int = GIT_TIMEOUT_SECONDS) -> Tuple[str, bool]:
    """Run a git command and return output."""
    return _run_command(['git'] + args, timeout, "Git")


def _run_pip_command(args: List[str], timeout: int = PIP_TIMEOUT_SECONDS) -> Tuple[str, bool]:
    """Run a pip command and return output."""
    return _run_command([sys.executable, '-m', 'pip'] + args, timeout, "Pip")


def _format_uptime(seconds: float) -> str:
    """Format uptime in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    elif seconds < 86400:
        return f"{seconds/3600:.1f} hours"
    else:
        return f"{seconds/86400:.1f} days"


async def _restart_server_delayed():
    """Restart the server after a delay to allow HTTP response to complete."""
    await asyncio.sleep(RESTART_DELAY_SECONDS)

    logger.warning("AUDIT: Server process restart initiated")

    try:
        if sys.platform == 'win32':
            # Windows: Start a new process, then exit
            subprocess.Popen(
                [sys.executable] + sys.argv,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            # Linux/Mac: Replace process with execv
            os.execv(sys.executable, [sys.executable] + sys.argv)

        # Exit current process (Windows only reaches here)
        sys.exit(0)
    except PermissionError:
        logger.error("Failed to restart server: insufficient permissions")
    except FileNotFoundError:
        logger.error(f"Failed to restart server: executable not found: {sys.executable}")
    except Exception as e:
        logger.error(f"Failed to restart server: {type(e).__name__} - {e}")
        sys.exit(1)


# Endpoints
@router.get("/status", response_model=ServerStatusResponse)
async def get_server_status(
    user: AuthenticationResult = Depends(require_read_access)
):
    """
    Get server status information.

    Returns current version, uptime, process ID, platform details, and Python version.
    Requires read access.
    """
    uptime = time.time() - _startup_time

    return ServerStatusResponse(
        version=__version__,
        uptime_seconds=uptime,
        uptime_formatted=_format_uptime(uptime),
        pid=os.getpid(),
        platform=platform.system(),
        platform_version=platform.version(),
        python_version=platform.python_version(),
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@router.get("/version/check", response_model=VersionCheckResponse)
async def check_for_updates(
    user: AuthenticationResult = Depends(require_admin_access)
):
    """
    Check for available updates from git repository.

    Fetches latest changes and compares with current version.
    Returns number of commits behind and list of commit messages.
    Requires admin access.
    """
    # Try to fetch latest changes
    fetch_output, fetch_success = _run_git_command(['fetch', 'origin'])

    if not fetch_success:
        return VersionCheckResponse(
            current_version=__version__,
            commits_behind=0,
            update_available=False,
            latest_commits=[],
            check_timestamp=datetime.now(timezone.utc).isoformat(),
            git_available=False,
            error=fetch_output
        )

    # Count commits behind origin/main
    count_output, count_success = _run_git_command(['rev-list', '--count', 'HEAD..origin/main'])

    if not count_success:
        return VersionCheckResponse(
            current_version=__version__,
            commits_behind=0,
            update_available=False,
            latest_commits=[],
            check_timestamp=datetime.now(timezone.utc).isoformat(),
            git_available=True,
            error=count_output
        )

    try:
        commits_behind = int(count_output)
    except ValueError:
        commits_behind = 0

    # Get latest commit messages if behind
    latest_commits = []
    if commits_behind > 0:
        log_output, log_success = _run_git_command([
            'log', '--oneline', '--no-decorate', f'HEAD..origin/main', '-n', '5'
        ])
        if log_success and log_output:
            latest_commits = [line.strip() for line in log_output.split('\n') if line.strip()]

    return VersionCheckResponse(
        current_version=__version__,
        commits_behind=commits_behind,
        update_available=commits_behind > 0,
        latest_commits=latest_commits,
        check_timestamp=datetime.now(timezone.utc).isoformat(),
        git_available=True,
        error=None
    )


@router.post("/restart", response_model=RestartResponse)
async def restart_server(
    request: RestartRequest,
    background_tasks: BackgroundTasks,
    user: AuthenticationResult = Depends(require_admin_access)
):
    """
    Restart the server process.

    Schedules a delayed restart to allow HTTP response to complete.
    Returns 202 Accepted immediately.
    Requires admin access and explicit confirmation.
    """
    if not request.confirm:
        raise HTTPException(
            status_code=400,
            detail="Restart confirmation required. Set 'confirm' to true."
        )

    logger.warning(f"AUDIT: Server restart requested by user: {user.client_id} (auth: {user.auth_method})")

    # Schedule restart in background
    background_tasks.add_task(_restart_server_delayed)

    return RestartResponse(
        status="accepted",
        message=f"Server restarting in {RESTART_DELAY_SECONDS} seconds",
        restart_in_seconds=RESTART_DELAY_SECONDS
    )


@router.post("/update", response_model=UpdateResponse)
async def update_server(
    request: UpdateRequest,
    background_tasks: BackgroundTasks,
    user: AuthenticationResult = Depends(require_admin_access)
):
    """
    Update server from git and restart.

    Performs git pull, pip install, and schedules restart.
    Returns 202 Accepted with output from each step.
    Requires admin access and explicit confirmation.
    """
    if not request.confirm:
        raise HTTPException(
            status_code=400,
            detail="Update confirmation required. Set 'confirm' to true."
        )

    logger.warning(f"AUDIT: Server update requested by user: {user.client_id} (auth: {user.auth_method})")

    # Step 1: Git pull
    git_output, git_success = _run_git_command(['pull', 'origin', 'main'])

    if not git_success:
        return UpdateResponse(
            status="error",
            message="Git pull failed",
            git_output=git_output,
            pip_output=None,
            restart_scheduled=False
        )

    # Step 2: Pip install
    pip_output, pip_success = _run_pip_command(['install', '-e', '.'])

    if not pip_success:
        return UpdateResponse(
            status="error",
            message="Pip install failed",
            git_output=git_output,
            pip_output=pip_output,
            restart_scheduled=False
        )

    # Step 3: Schedule restart
    background_tasks.add_task(_restart_server_delayed)

    logger.warning(f"AUDIT: Server update completed by {user.client_id}, restart scheduled")

    return UpdateResponse(
        status="success",
        message=f"Update completed successfully, server restarting in {RESTART_DELAY_SECONDS} seconds",
        git_output=git_output,
        pip_output=pip_output,
        restart_scheduled=True
    )
