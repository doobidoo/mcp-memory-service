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
Graph traversal handler functions for MCP server.

Provides graph database operations including:
- find_connected_memories: Multi-hop connection discovery
- find_shortest_path: Path finding between memories
- get_memory_subgraph: Subgraph extraction for visualization
"""

import json
import logging
from typing import List, Optional

from mcp import types
from ...storage.graph import GraphStorage
from ...config import SQLITE_VEC_PATH, STORAGE_BACKEND

logger = logging.getLogger(__name__)


def get_graph_storage() -> Optional[GraphStorage]:
    """
    Get graph storage instance if available.

    Graph operations are only available for SQLite-based backends
    (sqlite_vec and hybrid). Cloudflare backend does not support
    graph traversal.

    Returns:
        GraphStorage instance if backend supports it, None otherwise
    """
    if STORAGE_BACKEND in ['sqlite_vec', 'hybrid']:
        try:
            return GraphStorage(SQLITE_VEC_PATH)
        except Exception as e:
            logger.error(f"Failed to initialize GraphStorage: {e}")
            return None
    return None


async def handle_memory_graph(server, arguments: dict) -> List[types.TextContent]:
    """Unified handler for graph operations."""
    action = arguments.get("action")

    if not action:
        return [types.TextContent(type="text", text="Error: action parameter is required")]

    # Validate action
    valid_actions = ["connected", "path", "subgraph"]
    if action not in valid_actions:
        return [types.TextContent(
            type="text",
            text=f"Error: Invalid action '{action}'. Must be one of: {', '.join(valid_actions)}"
        )]

    try:
        # Route to appropriate handler based on action
        if action == "connected":
            # Find connected memories
            hash_val = arguments.get("hash")
            if not hash_val:
                return [types.TextContent(type="text", text="Error: hash is required for 'connected' action")]

            return await handle_find_connected_memories(server, {
                "hash": hash_val,
                "max_hops": arguments.get("max_hops", 2)
            })

        elif action == "path":
            # Find shortest path
            hash1 = arguments.get("hash1")
            hash2 = arguments.get("hash2")
            if not hash1 or not hash2:
                return [types.TextContent(type="text", text="Error: hash1 and hash2 are required for 'path' action")]

            return await handle_find_shortest_path(server, {
                "hash1": hash1,
                "hash2": hash2,
                "max_depth": arguments.get("max_depth", 5)
            })

        elif action == "subgraph":
            # Get memory subgraph
            hash_val = arguments.get("hash")
            if not hash_val:
                return [types.TextContent(type="text", text="Error: hash is required for 'subgraph' action")]

            return await handle_get_memory_subgraph(server, {
                "hash": hash_val,
                "radius": arguments.get("radius", 2)
            })

        else:
            # Should never reach here due to validation above
            return [types.TextContent(type="text", text=f"Error: Unknown action '{action}'")]

    except Exception as e:
        import traceback
        error_msg = f"Error in memory_graph action '{action}': {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return [types.TextContent(type="text", text=error_msg)]


async def handle_find_connected_memories(
    server,
    arguments: dict
) -> List[types.TextContent]:
    """
    Find memories connected to a given memory via associations.

    Performs breadth-first traversal of the association graph up to
    max_hops distance, returning all connected memories with their
    distance from the source.

    Args:
        server: MCP server instance (unused, for handler pattern consistency)
        arguments: Dict with:
            - hash: Content hash of the starting memory
            - max_hops: Maximum number of hops to traverse (default: 2)

    Returns:
        List with single TextContent containing JSON result:
        {
            "success": true,
            "connected": [
                {"hash": "abc123", "distance": 1},
                {"hash": "def456", "distance": 2}
            ],
            "count": 2
        }
        Or error result if graph unavailable or operation fails.
    """
    logger.info("=== EXECUTING FIND_CONNECTED_MEMORIES ===")

    # Get graph storage
    graph = get_graph_storage()
    if graph is None:
        result = {
            "success": False,
            "error": f"Graph operations not available for backend: {STORAGE_BACKEND}",
            "connected": [],
            "count": 0
        }
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    # Validate required parameters
    memory_hash = arguments.get("hash")
    if not memory_hash:
        result = {
            "success": False,
            "error": "Missing required parameter: hash",
            "connected": [],
            "count": 0
        }
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    # Get optional parameters
    max_hops = arguments.get("max_hops", 2)

    try:
        # Perform graph traversal
        connected = await graph.find_connected(memory_hash, max_hops=max_hops)

        # Format results
        result = {
            "success": True,
            "connected": [
                {"hash": hash_val, "distance": distance}
                for hash_val, distance in connected
            ],
            "count": len(connected)
        }

        logger.info(f"Found {len(connected)} connected memories within {max_hops} hops")

        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    except Exception as e:
        logger.error(f"Error finding connected memories: {e}")
        result = {
            "success": False,
            "error": str(e),
            "connected": [],
            "count": 0
        }
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]


async def handle_find_shortest_path(
    server,
    arguments: dict
) -> List[types.TextContent]:
    """
    Find shortest path between two memories in the association graph.

    Uses breadth-first search to find the shortest sequence of associations
    connecting two memories. Returns null if no path exists.

    Args:
        server: MCP server instance (unused, for handler pattern consistency)
        arguments: Dict with:
            - hash1: Starting memory hash
            - hash2: Target memory hash
            - max_depth: Maximum path length (default: 5)

    Returns:
        List with single TextContent containing JSON result:
        {
            "success": true,
            "path": ["hash1", "intermediate_hash", "hash2"],
            "length": 3
        }
        Or null path if no connection exists.
    """
    logger.info("=== EXECUTING FIND_SHORTEST_PATH ===")

    # Get graph storage
    graph = get_graph_storage()
    if graph is None:
        result = {
            "success": False,
            "error": f"Graph operations not available for backend: {STORAGE_BACKEND}",
            "path": None,
            "length": 0
        }
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    # Validate required parameters
    hash1 = arguments.get("hash1")
    hash2 = arguments.get("hash2")

    if not hash1 or not hash2:
        result = {
            "success": False,
            "error": "Missing required parameters: hash1 and hash2",
            "path": None,
            "length": 0
        }
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    # Get optional parameters
    max_depth = arguments.get("max_depth", 5)

    try:
        # Find shortest path
        path = await graph.shortest_path(hash1, hash2, max_depth=max_depth)

        # Format results
        if path is not None:
            result = {
                "success": True,
                "path": path,
                "length": len(path)
            }
            logger.info(f"Found path of length {len(path)} between memories")
        else:
            result = {
                "success": True,
                "path": None,
                "length": 0,
                "message": "No path found within depth limit"
            }
            logger.info(f"No path found between {hash1} and {hash2}")

        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    except Exception as e:
        logger.error(f"Error finding shortest path: {e}")
        result = {
            "success": False,
            "error": str(e),
            "path": None,
            "length": 0
        }
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]


async def handle_get_memory_subgraph(
    server,
    arguments: dict
) -> List[types.TextContent]:
    """
    Get subgraph around a memory for visualization.

    Extracts all nodes and edges within the specified radius for
    graph visualization. Returns nodes (memory hashes) and edges
    (associations with metadata).

    Args:
        server: MCP server instance (unused, for handler pattern consistency)
        arguments: Dict with:
            - hash: Center memory hash
            - radius: Number of hops to include (default: 2)

    Returns:
        List with single TextContent containing JSON result:
        {
            "success": true,
            "nodes": ["hash1", "hash2", "hash3"],
            "edges": [
                {
                    "source": "hash1",
                    "target": "hash2",
                    "similarity": 0.65,
                    "connection_types": ["semantic", "temporal"],
                    "metadata": {}
                }
            ],
            "node_count": 3,
            "edge_count": 2
        }
    """
    logger.info("=== EXECUTING GET_MEMORY_SUBGRAPH ===")

    # Get graph storage
    graph = get_graph_storage()
    if graph is None:
        result = {
            "success": False,
            "error": f"Graph operations not available for backend: {STORAGE_BACKEND}",
            "nodes": [],
            "edges": [],
            "node_count": 0,
            "edge_count": 0
        }
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    # Validate required parameters
    memory_hash = arguments.get("hash")
    if not memory_hash:
        result = {
            "success": False,
            "error": "Missing required parameter: hash",
            "nodes": [],
            "edges": [],
            "node_count": 0,
            "edge_count": 0
        }
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    # Get optional parameters
    radius = arguments.get("radius", 2)

    try:
        # Extract subgraph
        subgraph = await graph.get_subgraph(memory_hash, radius=radius)

        # Format results
        result = {
            "success": True,
            "nodes": subgraph["nodes"],
            "edges": subgraph["edges"],
            "node_count": len(subgraph["nodes"]),
            "edge_count": len(subgraph["edges"])
        }

        logger.info(
            f"Extracted subgraph: {len(subgraph['nodes'])} nodes, "
            f"{len(subgraph['edges'])} edges"
        )

        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    except Exception as e:
        logger.error(f"Error extracting subgraph: {e}")
        result = {
            "success": False,
            "error": str(e),
            "nodes": [],
            "edges": [],
            "node_count": 0,
            "edge_count": 0
        }
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
