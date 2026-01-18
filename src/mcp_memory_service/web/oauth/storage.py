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
OAuth 2.1 storage for MCP Memory Service.

Supports multiple storage backends:
- memory: In-memory storage (default, dev/testing)
- sqlite: SQLite persistent storage (production)

Configure via environment variables:
    export MCP_OAUTH_STORAGE_BACKEND=sqlite
    export MCP_OAUTH_SQLITE_PATH=./data/oauth.db
"""

import logging
from typing import Optional
from .storage import OAuthStorage, create_oauth_storage

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
        from ...config import OAUTH_STORAGE_BACKEND, OAUTH_SQLITE_PATH

        logger.info(f"Initializing OAuth storage backend: {OAUTH_STORAGE_BACKEND}")

        if OAUTH_STORAGE_BACKEND == "sqlite":
            logger.info(f"Using SQLite OAuth storage at: {OAUTH_SQLITE_PATH}")
            _oauth_storage = create_oauth_storage("sqlite", db_path=OAUTH_SQLITE_PATH)
        else:
            logger.info("Using in-memory OAuth storage (not persistent)")
            _oauth_storage = create_oauth_storage("memory")

    return _oauth_storage


# Backward compatibility: maintain global instance
oauth_storage = get_oauth_storage()

__all__ = ["OAuthStorage", "create_oauth_storage", "oauth_storage", "get_oauth_storage"]
