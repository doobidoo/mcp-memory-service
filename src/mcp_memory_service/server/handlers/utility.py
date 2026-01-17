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
Utility handler functions for MCP server.

Database health checks, cache statistics, and monitoring utilities.
Extracted from server_impl.py Phase 2.3 refactoring.
"""

import os
import json
import logging
import traceback
from typing import List

from mcp import types
from ...server.cache_manager import _CACHE_STATS, _STORAGE_CACHE, _MEMORY_SERVICE_CACHE
from ...config import STORAGE_BACKEND, SQLITE_VEC_PATH, EMBEDDING_MODEL_NAME

try:
    from ... import __version__
except (ImportError, AttributeError):
    __version__ = "0.0.0.dev0"

logger = logging.getLogger(__name__)


async def handle_check_database_health(server, arguments: dict) -> List[types.TextContent]:
    """Handle database health check requests with performance metrics."""
    from ...utils.health_check import HealthCheckFactory

    logger.info("=== EXECUTING CHECK_DATABASE_HEALTH ===")
    try:
        # Initialize storage lazily when needed
        try:
            storage = await server._ensure_storage_initialized()
        except Exception as init_error:
            # Storage initialization failed
            result = {
                "validation": {
                    "status": "unhealthy",
                    "message": f"Storage initialization failed: {str(init_error)}"
                },
                "statistics": {
                    "status": "error",
                    "error": "Cannot get statistics - storage not initialized"
                },
                "performance": {
                    "storage": {},
                    "server": {
                        "average_query_time_ms": server.get_average_query_time(),
                        "total_queries": len(server.query_times)
                    }
                }
            }

            logger.error(f"Storage initialization failed during health check: {str(init_error)}")
            return [types.TextContent(
                type="text",
                text=f"Database Health Check Results:\n{json.dumps(result, indent=2)}"
            )]

        # Use Strategy pattern to check backend-specific health
        checker = HealthCheckFactory.create(storage)
        is_valid, message, stats = await checker.check_health(storage)

        # Get performance stats from optimized storage
        performance_stats = {}
        if hasattr(storage, 'get_performance_stats') and callable(storage.get_performance_stats):
            try:
                performance_stats = storage.get_performance_stats()
            except Exception as perf_error:
                logger.warning(f"Could not get performance stats: {str(perf_error)}")
                performance_stats = {"error": str(perf_error)}

        # Get server-level performance stats
        server_stats = {
            "average_query_time_ms": server.get_average_query_time(),
            "total_queries": len(server.query_times),
            "storage_type": storage.__class__.__name__
        }

        # Add storage initialization status for debugging
        if hasattr(storage, 'get_initialization_status') and callable(storage.get_initialization_status):
            try:
                server_stats["storage_initialization"] = storage.get_initialization_status()
            except Exception:
                pass

        # Combine results with performance data
        result = {
            "version": __version__,
            "validation": {
                "status": "healthy" if is_valid else "unhealthy",
                "message": message
            },
            "statistics": stats,
            "performance": {
                "storage": performance_stats,
                "server": server_stats
            }
        }

        logger.info(f"Database health result with performance data: {result}")
        return [types.TextContent(
            type="text",
            text=f"Database Health Check Results:\n{json.dumps(result, indent=2)}"
        )]
    except Exception as e:
        logger.error(f"Error in check_database_health: {str(e)}")
        logger.error(traceback.format_exc())
        return [types.TextContent(
            type="text",
            text=f"Error checking database health: {str(e)}"
        )]


async def handle_get_cache_stats(server, arguments: dict) -> List[types.TextContent]:
    """
    Get MCP server global cache statistics for performance monitoring.

    Returns detailed metrics about storage and memory service caching,
    including hit rates, initialization times, and cache sizes.
    """
    global _CACHE_STATS, _STORAGE_CACHE, _MEMORY_SERVICE_CACHE

    try:
        # Import shared stats calculation utility
        from ...utils.cache_manager import CacheStats, calculate_cache_stats_dict

        # Convert global dict to CacheStats dataclass
        stats = CacheStats(
            total_calls=_CACHE_STATS["total_calls"],
            storage_hits=_CACHE_STATS["storage_hits"],
            storage_misses=_CACHE_STATS["storage_misses"],
            service_hits=_CACHE_STATS["service_hits"],
            service_misses=_CACHE_STATS["service_misses"],
            initialization_times=_CACHE_STATS["initialization_times"]
        )

        # Calculate statistics using shared utility
        cache_sizes = (len(_STORAGE_CACHE), len(_MEMORY_SERVICE_CACHE))
        result = calculate_cache_stats_dict(stats, cache_sizes)

        # Add server-specific details
        result["storage_cache"]["keys"] = list(_STORAGE_CACHE.keys())
        result["backend_info"] = {
            "storage_backend": STORAGE_BACKEND,
            "sqlite_path": SQLITE_VEC_PATH,
            "embedding_model": EMBEDDING_MODEL_NAME
        }

        logger.info(f"Cache stats retrieved: {result['message']}")

        # Return JSON string for easy parsing by clients
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    except Exception as e:
        logger.error(f"Error in get_cache_stats: {str(e)}")
        logger.error(traceback.format_exc())
        return [types.TextContent(
            type="text",
            text=f"Error getting cache stats: {str(e)}"
        )]
