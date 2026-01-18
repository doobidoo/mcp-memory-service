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
Factory for creating OAuth storage backend instances.

Provides a simple factory function to create storage backends based on
configuration. This allows easy switching between storage implementations
without changing application code.
"""

import logging
import os
from typing import Optional
from .base import OAuthStorage
from .memory import MemoryOAuthStorage
from .sqlite import SQLiteOAuthStorage

logger = logging.getLogger(__name__)


def create_oauth_storage(
    backend_type: str = "memory",
    **kwargs
) -> OAuthStorage:
    """
    Create an OAuth storage backend instance.

    Args:
        backend_type: Type of storage backend to create.
                     Supported values: "memory" (default), "sqlite"
        **kwargs: Additional backend-specific configuration parameters
                 - db_path: SQLite database path (for "sqlite" backend)

    Returns:
        OAuthStorage instance of the requested type

    Raises:
        ValueError: If backend_type is unsupported

    Examples:
        # Create in-memory storage (development)
        storage = create_oauth_storage("memory")

        # Create SQLite storage (production)
        storage = create_oauth_storage("sqlite", db_path="./data/oauth.db")
    """
    backend = backend_type.lower()

    if backend == "memory":
        logger.info("Creating in-memory OAuth storage backend")
        return MemoryOAuthStorage()

    elif backend == "sqlite":
        db_path = kwargs.get("db_path", "./data/oauth.db")

        # Ensure directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Created OAuth database directory: {db_dir}")

        logger.info(f"Creating SQLite OAuth storage backend at {db_path}")
        return SQLiteOAuthStorage(db_path=db_path)

    else:
        raise ValueError(
            f"Unsupported OAuth storage backend: {backend_type}. "
            f"Supported backends: memory, sqlite"
        )
