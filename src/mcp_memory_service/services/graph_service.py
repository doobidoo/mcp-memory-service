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
Graph Service - Shared business logic for knowledge graph operations.

Provides a unified interface for graph traversal operations (find connected,
shortest path, subgraph extraction) that can be used by both the stdio
MCP server (server_impl.py) and the streamable-http FastMCP server
(mcp_server.py).
"""

import logging
from typing import Any, Dict, Optional

from ..storage.graph import GraphStorage

logger = logging.getLogger(__name__)


class GraphService:
    """
    Shared service for knowledge graph operations.

    Wraps GraphStorage with consistent error handling and response
    formatting. All methods return Dict with a ``success`` field
    and operation-specific data, matching the response format used
    by the stdio mode graph handlers.

    Graph operations are only available for SQLite-based storage backends
    (sqlite_vec and hybrid). For other backends (milvus, cloudflare),
    ``graph_storage`` will be None and all operations return a structured
    error. Callers should check :meth:`is_available` before assuming
    graph features are present.
    """

    def __init__(self, graph_storage: Optional[GraphStorage] = None):
        """
        Initialize GraphService.

        Args:
            graph_storage: GraphStorage instance, or None if graph
                operations are not available for the current backend.
        """
        self._graph = graph_storage

    def is_available(self) -> bool:
        """Check if graph operations are available."""
        return self._graph is not None

    async def find_connected(
        self, content_hash: str, max_hops: int = 2
    ) -> Dict[str, Any]:
        """
        Find memories connected to a given memory via associations.

        Performs breadth-first traversal of the association graph up to
        max_hops distance.

        Args:
            content_hash: Content hash of the starting memory.
            max_hops: Maximum number of hops to traverse (default 2).

        Returns:
            Dict with success status, connected memories list, and count.
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Graph operations not available for current storage backend",
                "connected": [],
                "count": 0,
            }

        try:
            connected = await self._graph.find_connected(
                content_hash, max_hops=max_hops
            )
            result = {
                "success": True,
                "connected": [
                    {"hash": h, "distance": d} for h, d in connected
                ],
                "count": len(connected),
            }
            logger.info(
                "Found %d connected memories within %d hops of %s",
                len(connected), max_hops, content_hash[:12],
            )
            return result

        except Exception as e:
            logger.error("Error finding connected memories: %s", e)
            return {
                "success": False,
                "error": str(e),
                "connected": [],
                "count": 0,
            }

    async def find_shortest_path(
        self, hash1: str, hash2: str, max_depth: int = 5
    ) -> Dict[str, Any]:
        """
        Find shortest path between two memories in the association graph.

        Args:
            hash1: Starting memory content hash.
            hash2: Target memory content hash.
            max_depth: Maximum path length (default 5).

        Returns:
            Dict with success status, path list, and path length.
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Graph operations not available for current storage backend",
                "path": None,
                "length": 0,
            }

        try:
            path = await self._graph.shortest_path(
                hash1, hash2, max_depth=max_depth
            )
            if path is not None:
                logger.info(
                    "Found path of length %d between %s and %s",
                    len(path), hash1[:12], hash2[:12],
                )
                return {"success": True, "path": path, "length": len(path)}

            logger.info(
                "No path found between %s and %s within depth %d",
                hash1[:12], hash2[:12], max_depth,
            )
            return {
                "success": True,
                "path": None,
                "length": 0,
                "message": "No path found within depth limit",
            }

        except Exception as e:
            logger.error("Error finding shortest path: %s", e)
            return {
                "success": False,
                "error": str(e),
                "path": None,
                "length": 0,
            }

    async def get_subgraph(
        self, content_hash: str, radius: int = 2
    ) -> Dict[str, Any]:
        """
        Extract subgraph around a memory for visualization.

        Args:
            content_hash: Center memory content hash.
            radius: Number of hops to include (default 2).

        Returns:
            Dict with success status, nodes, edges, and counts.
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Graph operations not available for current storage backend",
                "nodes": [],
                "edges": [],
                "node_count": 0,
                "edge_count": 0,
            }

        try:
            subgraph = await self._graph.get_subgraph(
                content_hash, radius=radius
            )
            result = {
                "success": True,
                "nodes": subgraph["nodes"],
                "edges": subgraph["edges"],
                "node_count": len(subgraph["nodes"]),
                "edge_count": len(subgraph["edges"]),
            }
            logger.info(
                "Extracted subgraph: %d nodes, %d edges around %s",
                result["node_count"], result["edge_count"], content_hash[:12],
            )
            return result

        except Exception as e:
            logger.error("Error extracting subgraph: %s", e)
            return {
                "success": False,
                "error": str(e),
                "nodes": [],
                "edges": [],
                "node_count": 0,
                "edge_count": 0,
            }
