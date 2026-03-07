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
Health check strategies for different storage backends.

Implements Strategy Pattern to isolate backend-specific health check logic.
Extracted from server/handlers/utility.py Phase 3.1 refactoring.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

from ..config import SQLITE_VEC_PATH

logger = logging.getLogger(__name__)


class HealthCheckStrategy(ABC):
    """Abstract base class for storage health check strategies."""

    @abstractmethod
    async def check_health(self, storage: Any) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Check the health of a storage backend.

        Args:
            storage: Storage instance to check

        Returns:
            Tuple of (is_valid, message, stats_dict)
        """
        pass


class SqliteHealthChecker(HealthCheckStrategy):
    """Health check strategy for SQLite-vec storage."""

    async def check_health(self, storage: Any) -> Tuple[bool, str, Dict[str, Any]]:
        """Check SQLite-vec storage health."""
        # Check if connection exists
        if not hasattr(storage, 'conn') or storage.conn is None:
            return False, "SQLite database connection is not initialized", {}

        try:
            # Check for required tables
            cursor = storage.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
            )
            if not cursor.fetchone():
                return False, "SQLite database is missing required tables", {}

            # Count memories
            cursor = storage.conn.execute('SELECT COUNT(*) FROM memories')
            memory_count = cursor.fetchone()[0]

            # Check if embedding tables exist
            cursor = storage.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_embeddings'"
            )
            has_embeddings = cursor.fetchone() is not None

            # Check embedding model
            has_model = hasattr(storage, 'embedding_model') and storage.embedding_model is not None

            # Collect stats
            stats = {
                "status": "healthy",
                "backend": "sqlite-vec",
                "total_memories": memory_count,
                "has_embedding_tables": has_embeddings,
                "has_embedding_model": has_model,
                "embedding_model": storage.embedding_model_name if hasattr(storage, 'embedding_model_name') else "none"
            }

            # Get database file size
            db_path = storage.db_path if hasattr(storage, 'db_path') else None
            if db_path and os.path.exists(db_path):
                file_size = os.path.getsize(db_path)
                stats["database_size_bytes"] = file_size
                stats["database_size_mb"] = round(file_size / (1024 * 1024), 2)

            return True, "SQLite-vec database validation successful", stats

        except Exception as e:
            logger.error(f"SQLite health check error: {e}")
            return False, f"SQLite database validation error: {str(e)}", {
                "status": "error",
                "error": str(e),
                "backend": "sqlite-vec"
            }


class CloudflareHealthChecker(HealthCheckStrategy):
    """Health check strategy for Cloudflare storage."""

    async def check_health(self, storage: Any) -> Tuple[bool, str, Dict[str, Any]]:
        """Check Cloudflare storage health."""
        try:
            # Check if storage is properly initialized
            if not hasattr(storage, 'client') or storage.client is None:
                return False, "Cloudflare storage client is not initialized", {
                    "status": "error",
                    "error": "Cloudflare storage client is not initialized",
                    "backend": "cloudflare"
                }

            # Get storage stats
            storage_stats = await storage.get_stats()

            # Collect basic health info
            stats = {
                "status": "healthy",
                "backend": "cloudflare",
                "total_memories": storage_stats.get("total_memories", 0),
                "vectorize_index": storage.vectorize_index,
                "d1_database_id": storage.d1_database_id,
                "r2_bucket": storage.r2_bucket,
                "embedding_model": storage.embedding_model
            }

            # Add additional stats if available
            stats.update(storage_stats)

            return True, "Cloudflare storage validation successful", stats

        except Exception as e:
            logger.error(f"Cloudflare health check error: {e}")
            return False, f"Cloudflare storage validation error: {str(e)}", {
                "status": "error",
                "error": str(e),
                "backend": "cloudflare"
            }


class HybridHealthChecker(HealthCheckStrategy):
    """Health check strategy for Hybrid (SQLite-vec + Cloudflare) storage."""

    async def check_health(self, storage: Any) -> Tuple[bool, str, Dict[str, Any]]:
        """Check Hybrid storage health."""
        try:
            # Check primary storage exists
            if not hasattr(storage, 'primary') or storage.primary is None:
                return False, "Hybrid storage primary backend is not initialized", {
                    "status": "error",
                    "error": "Hybrid storage primary backend is not initialized",
                    "backend": "hybrid"
                }

            primary_storage = storage.primary

            # Validate primary storage (SQLite-vec)
            if not hasattr(primary_storage, 'conn') or primary_storage.conn is None:
                return False, "Hybrid storage: SQLite connection is not initialized", {
                    "status": "error",
                    "error": "SQLite connection is not initialized",
                    "backend": "hybrid"
                }

            # Check for required tables
            cursor = primary_storage.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
            )
            if not cursor.fetchone():
                return False, "Hybrid storage: SQLite database is missing required tables", {
                    "status": "error",
                    "error": "SQLite database is missing required tables",
                    "backend": "hybrid"
                }

            # Count memories
            cursor = primary_storage.conn.execute('SELECT COUNT(*) FROM memories')
            memory_count = cursor.fetchone()[0]

            # Check if embedding tables exist
            cursor = primary_storage.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_embeddings'"
            )
            has_embeddings = cursor.fetchone() is not None

            # Check secondary (Cloudflare) status
            cloudflare_status = "not_configured"
            if hasattr(storage, 'secondary') and storage.secondary:
                sync_service = getattr(storage, 'sync_service', None)
                if sync_service and getattr(sync_service, 'is_running', False):
                    cloudflare_status = "syncing"
                else:
                    cloudflare_status = "configured"

            # Collect stats
            stats = {
                "status": "healthy",
                "backend": "hybrid",
                "total_memories": memory_count,
                "has_embeddings": has_embeddings,
                "database_path": getattr(primary_storage, 'db_path', SQLITE_VEC_PATH),
                "cloudflare_sync": cloudflare_status
            }

            message = f"Hybrid storage validation successful ({memory_count} memories, Cloudflare: {cloudflare_status})"
            return True, message, stats

        except Exception as e:
            logger.error(f"Hybrid health check error: {e}")
            return False, f"Hybrid storage validation error: {str(e)}", {
                "status": "error",
                "error": str(e),
                "backend": "hybrid"
            }


class HealthCheckFactory:
    """Factory for creating appropriate health check strategies."""

    @staticmethod
    def _is_hybrid_like_storage(storage: Any) -> bool:
        """Detect hybrid storage instances, including delegated/wrapped variants."""
        if storage.__class__.__name__ == "HybridMemoryStorage":
            return True

        # Hybrid storage exposes a primary backend and either secondary backend
        # or sync service metadata. Use structural detection to avoid relying
        # solely on class names when storage objects are wrapped/delegated.
        has_primary = hasattr(storage, 'primary')
        has_secondary = hasattr(storage, 'secondary')
        has_sync_service = hasattr(storage, 'sync_service')

        return has_primary and (has_secondary or has_sync_service)

    @staticmethod
    def create(storage: Any) -> HealthCheckStrategy:
        """
        Create a health check strategy based on storage type.

        Args:
            storage: Storage instance

        Returns:
            Appropriate HealthCheckStrategy instance
        """
        storage_type = storage.__class__.__name__

        if HealthCheckFactory._is_hybrid_like_storage(storage):
            return HybridHealthChecker()
        elif storage_type == "SqliteVecMemoryStorage":
            return SqliteHealthChecker()
        elif storage_type == "CloudflareStorage":
            return CloudflareHealthChecker()
        else:
            # Default to a simple unknown strategy
            return UnknownStorageChecker()


class UnknownStorageChecker(HealthCheckStrategy):
    """Health check strategy for unknown storage types."""

    async def check_health(self, storage: Any) -> Tuple[bool, str, Dict[str, Any]]:
        """Handle unknown storage types."""
        storage_type = storage.__class__.__name__
        return False, f"Unknown storage type: {storage_type}", {
            "status": "error",
            "error": f"Unknown storage type: {storage_type}",
            "backend": "unknown"
        }
