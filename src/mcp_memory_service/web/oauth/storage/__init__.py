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
OAuth 2.1 storage backends for MCP Memory Service.

This package provides storage backends for OAuth clients, authorization codes,
and access tokens. Multiple backends are supported:

- MemoryOAuthStorage: In-memory storage (development, single-instance)
- SQLiteOAuthStorage: SQLite storage (production, persistent)

Configure via environment variables:
    export MCP_OAUTH_STORAGE_BACKEND=sqlite
    export MCP_OAUTH_SQLITE_PATH=./data/oauth.db

Usage:
    from mcp_memory_service.web.oauth.storage import create_oauth_storage

    # Create in-memory storage
    storage = create_oauth_storage("memory")

    # Create SQLite storage
    storage = create_oauth_storage("sqlite", db_path="/path/to/oauth.db")
"""

import logging
from typing import Optional

from .base import OAuthStorage
from .factory import create_oauth_storage
from .memory import MemoryOAuthStorage
from .sqlite import SQLiteOAuthStorage

logger = logging.getLogger(__name__)

# Global OAuth storage instance (initialized on first use)
_oauth_storage: Optional[OAuthStorage] = None


def get_oauth_storage() -> OAuthStorage:
    """
    Get or create global OAuth storage instance.

    Uses configuration from environment variables:
    - MCP_OAUTH_STORAGE_BACKEND: "memory" or "sqlite"
    - MCP_OAUTH_SQLITE_PATH: Path to SQLite database (if backend=sqlite)

    Returns:
        Configured OAuth storage backend
    """
    global _oauth_storage

    if _oauth_storage is None:
        from ....config import OAUTH_STORAGE_BACKEND, OAUTH_SQLITE_PATH

        logger.info(f"Initializing OAuth storage backend: {OAUTH_STORAGE_BACKEND}")

        if OAUTH_STORAGE_BACKEND == "sqlite":
            logger.info(f"Using SQLite OAuth storage at: {OAUTH_SQLITE_PATH}")
            _oauth_storage = create_oauth_storage("sqlite", db_path=OAUTH_SQLITE_PATH)
        else:
            logger.info("Using in-memory OAuth storage (not persistent)")
            _oauth_storage = create_oauth_storage("memory")

    return _oauth_storage


# Lazy global instance - initialized on first access to avoid import-time issues
def __getattr__(name):
    """Lazy attribute access for backward compatibility."""
    if name == "oauth_storage":
        return get_oauth_storage()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "OAuthStorage",
    "create_oauth_storage",
    "MemoryOAuthStorage",
    "SQLiteOAuthStorage",
    "get_oauth_storage",
    "oauth_storage",
]
