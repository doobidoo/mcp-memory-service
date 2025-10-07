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
CLI utilities for MCP Memory Service.
"""

import os
from typing import Optional

from ..storage.base import MemoryStorage


async def get_storage(backend: Optional[str] = None) -> MemoryStorage:
    """
    Get storage backend for CLI operations.

    Args:
        backend: Storage backend name ('sqlite_vec', 'cloudflare', or 'hybrid')

    Returns:
        Initialized storage backend
    """
    # Determine backend
    if backend is None:
        backend = os.getenv('MCP_MEMORY_STORAGE_BACKEND', 'sqlite_vec').lower()

    backend = backend.lower()

    if backend in ('sqlite_vec', 'sqlite-vec'):
        from ..storage.sqlite_vec import SqliteVecMemoryStorage
        from ..config import SQLITE_VEC_PATH
        storage = SqliteVecMemoryStorage(SQLITE_VEC_PATH)
        await storage.initialize()
        return storage
    elif backend == 'cloudflare':
        from ..storage.cloudflare import CloudflareStorage
        from ..config import (
            CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID,
            CLOUDFLARE_VECTORIZE_INDEX, CLOUDFLARE_D1_DATABASE_ID,
            CLOUDFLARE_R2_BUCKET, CLOUDFLARE_EMBEDDING_MODEL,
            CLOUDFLARE_LARGE_CONTENT_THRESHOLD, CLOUDFLARE_MAX_RETRIES,
            CLOUDFLARE_BASE_DELAY
        )
        storage = CloudflareStorage(
            api_token=CLOUDFLARE_API_TOKEN,
            account_id=CLOUDFLARE_ACCOUNT_ID,
            vectorize_index=CLOUDFLARE_VECTORIZE_INDEX,
            d1_database_id=CLOUDFLARE_D1_DATABASE_ID,
            r2_bucket=CLOUDFLARE_R2_BUCKET,
            embedding_model=CLOUDFLARE_EMBEDDING_MODEL,
            large_content_threshold=CLOUDFLARE_LARGE_CONTENT_THRESHOLD,
            max_retries=CLOUDFLARE_MAX_RETRIES,
            base_delay=CLOUDFLARE_BASE_DELAY
        )
        await storage.initialize()
        return storage
    else:
        raise ValueError(f"Unsupported storage backend: {backend}")