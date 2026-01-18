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
- SqliteOAuthStorage: SQLite storage (production, persistent) [Phase 2]

Usage:
    from mcp_memory_service.web.oauth.storage import create_oauth_storage

    # Create in-memory storage
    storage = create_oauth_storage("memory")

    # Future: Create SQLite storage
    # storage = create_oauth_storage("sqlite", db_path="/path/to/oauth.db")
"""

from .base import OAuthStorage
from .factory import create_oauth_storage
from .memory import MemoryOAuthStorage
from .sqlite import SQLiteOAuthStorage

__all__ = [
    "OAuthStorage",
    "create_oauth_storage",
    "MemoryOAuthStorage",
    "SQLiteOAuthStorage",
]
