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
FastAPI dependencies for the HTTP interface.
"""

import logging
from typing import Optional
from fastapi import HTTPException

from ..storage.base import MemoryStorage

logger = logging.getLogger(__name__)

# Global storage instance
_storage: Optional[MemoryStorage] = None


def set_storage(storage: MemoryStorage) -> None:
    """Set the global storage instance."""
    global _storage
    _storage = storage


def get_storage() -> MemoryStorage:
    """Get the global storage instance."""
    if _storage is None:
        raise HTTPException(status_code=503, detail="Storage not initialized")
    return _storage




async def create_storage_backend() -> MemoryStorage:
    """
    Create and initialize storage backend for web interface based on configuration.

    Returns:
        Initialized storage backend
    """
    from ..config import DATABASE_PATH
    from ..storage.factory import create_storage_instance

    logger.info("Creating storage backend for web interface...")

    # Use shared factory with DATABASE_PATH for web interface
    return await create_storage_instance(DATABASE_PATH)