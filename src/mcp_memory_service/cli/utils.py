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
        backend: Storage backend name ('sqlite_vec' or 'chromadb')
        
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
    elif backend == 'chromadb':
        from ..storage.chroma import ChromaMemoryStorage
        from ..config import CHROMA_PATH
        storage = ChromaMemoryStorage(CHROMA_PATH)
        await storage.initialize()
        return storage
    else:
        raise ValueError(f"Unsupported storage backend: {backend}")