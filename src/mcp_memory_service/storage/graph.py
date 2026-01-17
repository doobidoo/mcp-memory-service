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
Graph storage layer for memory associations.

Provides recursive CTE-based graph traversal operations including:
- Multi-hop connection discovery (find all connected memories)
- Shortest path finding between memories
- Subgraph extraction for visualization
- Association CRUD operations

Uses SQLite recursive CTEs for efficient graph queries with cycle prevention.
"""

import sqlite3
import json
import logging
import asyncio
from typing import List, Dict, Any, Tuple, Optional, Set
from datetime import datetime, timezone
import os

logger = logging.getLogger(__name__)

# SQL query templates for graph traversal
# These templates use .format() for safe variable substitution (not user input)
_QUERY_TEMPLATE_FIND_CONNECTED = """
WITH RECURSIVE connected_memories(hash, distance, path) AS (
    -- Base case: Start with the given memory
    SELECT ?, 0, ?

    UNION ALL

    -- Recursive case: Find neighbors of connected memories
    SELECT
        {selected_hash},
        cm.distance + 1,
        cm.path || {selected_hash} || ','
    FROM connected_memories cm
    {join_condition}
    WHERE
        cm.distance < ?  -- Limit recursion depth
        AND instr(cm.path, ',' || {selected_hash} || ',') = 0  -- Prevent cycles (exact match)
        {relationship_filter}
)
SELECT DISTINCT hash, distance
FROM connected_memories
WHERE distance > 0  -- Exclude starting node
ORDER BY distance, hash
"""

_QUERY_TEMPLATE_SHORTEST_PATH = """
WITH RECURSIVE path_finder(current_hash, path, depth) AS (
    -- Base case: Start from hash1
    SELECT ?, ?, 1

    UNION ALL

    -- Recursive case: Expand path
    SELECT
        mg.target_hash,
        pf.path || mg.target_hash || ',',
        pf.depth + 1
    FROM path_finder pf
    JOIN memory_graph mg ON pf.current_hash = mg.source_hash
    WHERE
        pf.depth < ?  -- Limit search depth
        AND instr(pf.path, ',' || mg.target_hash || ',') = 0  -- Prevent cycles (exact match)
        AND pf.current_hash != ?  -- Stop if target found (handled in outer query)
        {relationship_filter}
)
SELECT path
FROM path_finder
WHERE current_hash = ?  -- Found target
ORDER BY depth
LIMIT 1  -- Return shortest path only
"""

_QUERY_TEMPLATE_SUBGRAPH = """
SELECT
    source_hash,
    target_hash,
    similarity,
    connection_types,
    metadata,
    relationship_type
FROM memory_graph
WHERE source_hash IN ({placeholders})
  AND target_hash IN ({placeholders})
  {relationship_filter}
"""


class GraphStorage:
    """
    Graph-based storage for memory associations with recursive CTE queries.

    Supports bidirectional traversal, multi-hop discovery, and subgraph extraction
    for knowledge graph operations on memory associations.
    """

    def __init__(self, db_path: str):
        """
        Initialize graph storage with SQLite database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._connection = None
        self._lock = asyncio.Lock()  # Instance-level lock for thread safety
        logger.info(f"Initialized GraphStorage with database: {db_path}")

    async def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection with optimizations."""
        if self._connection is None:
            # Run connection setup in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self._connection = await loop.run_in_executor(
                None, self._create_connection
            )
        return self._connection

    def _create_connection(self) -> sqlite3.Connection:
        """Create database connection with performance optimizations."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Enable column access by name

        # Performance optimizations
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        conn.execute("PRAGMA temp_store=MEMORY")

        return conn

    async def store_association(
        self,
        source_hash: str,
        target_hash: str,
        similarity: float,
        connection_types: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[float] = None,
        relationship_type: str = "related"
    ) -> bool:
        """
        Store a memory association with bidirectional edges.

        KNOWN LIMITATION: Bidirectional storage with same relationship_type is incorrect
        for asymmetric relationships (causes, fixes, contradicts, supports). This will be
        fixed in PR #348 to store only single directed edges. Current behavior is safe
        because asymmetric relationships are not yet exposed in production APIs.

        Args:
            source_hash: Source memory content hash
            target_hash: Target memory content hash
            similarity: Similarity score (0.0-1.0)
            connection_types: List of connection types (e.g., ["semantic", "temporal"])
            metadata: Optional metadata dict with discovery context
            created_at: Optional timestamp (Unix epoch). If None, uses current time.
            relationship_type: Type of relationship (default: "related")
                NOTE: Asymmetric types (causes, fixes, contradicts, supports) should not
                be used in production until PR #348 is merged.

        Returns:
            True if stored successfully, False otherwise
        """
        if not source_hash or not target_hash:
            logger.error("Invalid hash provided (empty string)")
            return False

        if source_hash == target_hash:
            logger.warning(f"Cannot create self-loop: {source_hash}")
            return False

        # Validate similarity score bounds
        if not (0.0 <= similarity <= 1.0):
            logger.error(f"Invalid similarity score {similarity}, must be in range [0.0, 1.0]")
            return False

        try:
            conn = await self._get_connection()
            if created_at is None:
                created_at = datetime.now(timezone.utc).timestamp()
            metadata_json = json.dumps(metadata or {})
            connection_types_json = json.dumps(connection_types)

            # Store bidirectional edges (both A→B and B→A)
            # This simplifies queries - no need to check both directions
            # TODO(PR #348): This is incorrect for asymmetric relationships.
            # Should store only single directed edge for causes/fixes/contradicts/supports.
            # See: https://github.com/doobidoo/mcp-memory-service/issues/348
            async with self._lock:  # Use instance lock for thread safety
                cursor = conn.cursor()
                try:
                    # Insert or replace source → target
                    cursor.execute("""
                        INSERT OR REPLACE INTO memory_graph
                        (source_hash, target_hash, similarity, connection_types, metadata, created_at, relationship_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (source_hash, target_hash, similarity, connection_types_json,
                          metadata_json, created_at, relationship_type))

                    # Insert or replace target → source (bidirectional)
                    # FIXME(PR #348): Remove this for asymmetric relationships
                    cursor.execute("""
                        INSERT OR REPLACE INTO memory_graph
                        (source_hash, target_hash, similarity, connection_types, metadata, created_at, relationship_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (target_hash, source_hash, similarity, connection_types_json,
                          metadata_json, created_at, relationship_type))

                    conn.commit()
                finally:
                    cursor.close()

            logger.debug(f"Stored bidirectional association: {source_hash} ↔ {target_hash}")
            return True

        except sqlite3.Error as e:
            logger.error(f"Failed to store association: {e}")
            return False

    async def find_connected(
        self,
        memory_hash: str,
        max_hops: int = 2,
        relationship_type: Optional[str] = None,
        direction: str = "both"
    ) -> List[Tuple[str, int]]:
        """
        Find all memories connected within N hops using recursive CTE.

        This performs a breadth-first traversal of the association graph,
        tracking distance (hop count) and preventing cycles.

        Args:
            memory_hash: Starting memory content hash
            max_hops: Maximum number of hops to traverse (default: 2)
            relationship_type: Optional filter for specific relationship type
            direction: Direction to traverse ("outgoing", "incoming", "both")

        Returns:
            List of (memory_hash, distance) tuples, sorted by distance

        Example:
            [(hash1, 1), (hash2, 1), (hash3, 2), ...]
        """
        if not memory_hash:
            logger.error("Invalid memory hash (empty string)")
            return []

        # Validate direction parameter
        if direction not in ("outgoing", "incoming", "both"):
            logger.error(f"Invalid direction '{direction}', must be 'outgoing', 'incoming', or 'both'")
            return []

        try:
            conn = await self._get_connection()

            # Recursive CTE query for multi-hop traversal
            # Base case: Direct neighbors (distance = 1)
            # Recursive case: Neighbors of neighbors (distance + 1)
            # Cycle prevention: Wrap hashes with delimiters to avoid substring matches

            # Build WHERE clause for relationship_type filter
            relationship_filter = ""
            params = [memory_hash, f',{memory_hash},', max_hops]

            if relationship_type is not None:
                relationship_filter = "AND mg.relationship_type = ?"
                params.append(relationship_type)

            # Build join condition and selected hash based on direction
            if direction == "outgoing":
                join_condition = "JOIN memory_graph mg ON cm.hash = mg.source_hash"
                selected_hash = "mg.target_hash"
            elif direction == "incoming":
                join_condition = "JOIN memory_graph mg ON cm.hash = mg.target_hash"
                selected_hash = "mg.source_hash"
            else:  # both
                join_condition = "JOIN memory_graph mg ON cm.hash = mg.source_hash"
                selected_hash = "mg.target_hash"

            query = _QUERY_TEMPLATE_FIND_CONNECTED.format(
                selected_hash=selected_hash,
                join_condition=join_condition,
                relationship_filter=relationship_filter
            )

            cursor = conn.cursor()
            try:
                cursor.execute(query, params)
                results = cursor.fetchall()

                connected = [(row['hash'], row['distance']) for row in results]
                logger.debug(f"Found {len(connected)} connected memories within {max_hops} hops")

                return connected
            finally:
                cursor.close()

        except sqlite3.Error as e:
            logger.error(f"Failed to find connected memories: {e}")
            return []

    async def get_relationship_types(self, memory_hash: str) -> Dict[str, int]:
        """
        Get count of each relationship type for a given memory.

        Args:
            memory_hash: Memory content hash

        Returns:
            Dict mapping relationship types to counts
            Example: {"causes": 3, "fixes": 1, "related": 5}
        """
        if not memory_hash:
            logger.error("Invalid memory hash (empty string)")
            return {}

        try:
            conn = await self._get_connection()

            query = """
            SELECT relationship_type, COUNT(*) as count
            FROM memory_graph
            WHERE source_hash = ?
            GROUP BY relationship_type
            """

            cursor = conn.cursor()
            try:
                cursor.execute(query, (memory_hash,))
                results = cursor.fetchall()

                relationship_counts = {row['relationship_type']: row['count'] for row in results}
                logger.debug(f"Found {len(relationship_counts)} relationship types for {memory_hash}")

                return relationship_counts
            finally:
                cursor.close()

        except sqlite3.Error as e:
            logger.error(f"Failed to get relationship types: {e}")
            return {}

    async def shortest_path(
        self,
        hash1: str,
        hash2: str,
        max_depth: int = 5,
        relationship_types: Optional[List[str]] = None
    ) -> Optional[List[str]]:
        """
        Find shortest path between two memories using recursive CTE (BFS).

        Uses unidirectional breadth-first search from source to target.
        BFS guarantees the shortest path is found first due to level-order traversal.

        Note: Bidirectional BFS could improve performance for deep searches but adds
        complexity. Current performance (15ms typical) is excellent for our use case.

        Args:
            hash1: Source memory content hash
            hash2: Target memory content hash
            max_depth: Maximum path length to search (default: 5)
            relationship_types: Optional list of relationship types to filter by

        Returns:
            Ordered list of memory hashes representing path, or None if no path exists
            Example: [hash1, intermediate_hash, hash2]
        """
        if not hash1 or not hash2:
            logger.error("Invalid hash provided (empty string)")
            return None

        if hash1 == hash2:
            return [hash1]  # Trivial path

        try:
            conn = await self._get_connection()

            # Recursive CTE for BFS pathfinding
            # Stops at first path found (BFS guarantees shortest)
            # Cycle prevention: Wrap hashes with delimiters to avoid substring matches

            # Build WHERE clause for relationship_types filter
            relationship_filter = ""
            params = [hash1, f',{hash1},', max_depth, hash2, hash2]

            if relationship_types is not None and len(relationship_types) > 0:
                # Create IN clause for relationship types
                placeholders = ','.join('?' * len(relationship_types))
                relationship_filter = f"AND mg.relationship_type IN ({placeholders})"
                params.extend(relationship_types)

            query = _QUERY_TEMPLATE_SHORTEST_PATH.format(
                relationship_filter=relationship_filter
            )

            cursor = conn.cursor()
            try:
                cursor.execute(query, params)
                result = cursor.fetchone()

                if result:
                    # Path format: ",hash1,,hash2,,hash3," - filter empty strings
                    path = [h for h in result['path'].split(',') if h]
                    logger.debug(f"Found path of length {len(path)}: {hash1} → {hash2}")
                    return path
                else:
                    logger.debug(f"No path found between {hash1} and {hash2}")
                    return None
            finally:
                cursor.close()

        except sqlite3.Error as e:
            logger.error(f"Failed to find shortest path: {e}")
            return None

    async def get_subgraph(
        self,
        memory_hash: str,
        radius: int = 2,
        relationship_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract subgraph centered on given memory for visualization.

        Returns all nodes (memories) and edges (associations) within
        the specified radius, formatted for graph visualization libraries.

        Args:
            memory_hash: Center node memory hash
            radius: Number of hops to include (default: 2)
            relationship_type: Optional filter for specific relationship type

        Returns:
            Dict with "nodes" (list of hashes) and "edges" (list of edge objects)
            Example:
            {
                "nodes": ["hash1", "hash2", "hash3"],
                "edges": [
                    {
                        "source": "hash1",
                        "target": "hash2",
                        "similarity": 0.65,
                        "connection_types": ["semantic", "temporal"],
                        "metadata": {...},
                        "relationship_type": "causes"
                    },
                    ...
                ]
            }
        """
        if not memory_hash:
            logger.error("Invalid memory hash (empty string)")
            return {"nodes": [], "edges": []}

        try:
            # Get all connected nodes within radius (with optional relationship_type filter)
            connected = await self.find_connected(
                memory_hash,
                max_hops=radius,
                relationship_type=relationship_type
            )

            # Build node set (include center node)
            nodes = {memory_hash}
            nodes.update(hash for hash, _ in connected)

            # Check SQLite parameter limit (999 max, we use 2*len(nodes) + maybe 1 for relationship_type)
            if len(nodes) > 499:
                logger.warning(
                    f"Subgraph too large ({len(nodes)} nodes > 499 limit). "
                    f"Truncating to prevent SQLite parameter overflow."
                )
                # Keep center node + first 498 connected nodes
                nodes = {memory_hash}
                nodes.update(h for h, _ in list(connected)[:498])

            # Fetch all edges between nodes in subgraph
            conn = await self._get_connection()

            # Use parameterized query with IN clause
            # Safety: placeholders are constructed from validated node set, not user input
            placeholders = ','.join('?' * len(nodes))

            # Add relationship_type filter if specified
            relationship_filter = ""
            params = list(nodes) + list(nodes)

            if relationship_type is not None:
                relationship_filter = "AND relationship_type = ?"
                params.append(relationship_type)

            query = _QUERY_TEMPLATE_SUBGRAPH.format(
                placeholders=placeholders,
                relationship_filter=relationship_filter
            )

            cursor = conn.cursor()
            try:
                cursor.execute(query, params)
                results = cursor.fetchall()

                # Format edges for visualization
                edges = []
                seen_edges = set()  # Avoid duplicates from bidirectional storage

                for row in results:
                    source = row['source_hash']
                    target = row['target_hash']

                    # Use canonical edge representation (sorted tuple) to deduplicate
                    edge_key = tuple(sorted([source, target]))
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)

                    edges.append({
                        "source": source,
                        "target": target,
                        "similarity": row['similarity'],
                        "connection_types": json.loads(row['connection_types']),
                        "metadata": json.loads(row['metadata']) if row['metadata'] else {},
                        "relationship_type": row['relationship_type']
                    })

                subgraph = {
                    "nodes": list(nodes),
                    "edges": edges
                }

                logger.debug(f"Extracted subgraph: {len(nodes)} nodes, {len(edges)} edges")
                return subgraph
            finally:
                cursor.close()

        except sqlite3.Error as e:
            logger.error(f"Failed to extract subgraph: {e}")
            return {"nodes": [], "edges": []}

    async def get_association(
        self,
        source_hash: str,
        target_hash: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve specific association between two memories.

        Args:
            source_hash: Source memory content hash
            target_hash: Target memory content hash

        Returns:
            Dict with association data, or None if not found
            Example:
            {
                "source_hash": "abc123",
                "target_hash": "def456",
                "similarity": 0.65,
                "connection_types": ["semantic"],
                "metadata": {...},
                "created_at": 1234567890.0
            }
        """
        if not source_hash or not target_hash:
            logger.error("Invalid hash provided (empty string)")
            return None

        try:
            conn = await self._get_connection()

            # Query for either direction (bidirectional storage)
            query = """
            SELECT
                source_hash,
                target_hash,
                similarity,
                connection_types,
                metadata,
                created_at
            FROM memory_graph
            WHERE (source_hash = ? AND target_hash = ?)
               OR (source_hash = ? AND target_hash = ?)
            LIMIT 1
            """

            cursor = conn.cursor()
            cursor.execute(query, (source_hash, target_hash, target_hash, source_hash))
            result = cursor.fetchone()

            if result:
                return {
                    "source_hash": result['source_hash'],
                    "target_hash": result['target_hash'],
                    "similarity": result['similarity'],
                    "connection_types": json.loads(result['connection_types']),
                    "metadata": json.loads(result['metadata']) if result['metadata'] else {},
                    "created_at": result['created_at']
                }
            else:
                logger.debug(f"No association found: {source_hash} ↔ {target_hash}")
                return None

        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve association: {e}")
            return None

    async def delete_association(
        self,
        source_hash: str,
        target_hash: str
    ) -> bool:
        """
        Delete association between two memories (both directions).

        Args:
            source_hash: Source memory content hash
            target_hash: Target memory content hash

        Returns:
            True if deleted successfully, False otherwise
        """
        if not source_hash or not target_hash:
            logger.error("Invalid hash provided (empty string)")
            return False

        try:
            conn = await self._get_connection()

            async with self._lock:  # Ensure thread safety
                cursor = conn.cursor()

                # Delete both directions (bidirectional storage)
                cursor.execute("""
                    DELETE FROM memory_graph
                    WHERE (source_hash = ? AND target_hash = ?)
                       OR (source_hash = ? AND target_hash = ?)
                """, (source_hash, target_hash, target_hash, source_hash))

                conn.commit()

                deleted_count = cursor.rowcount
                if deleted_count > 0:
                    logger.debug(f"Deleted association: {source_hash} ↔ {target_hash}")
                    return True
                else:
                    logger.warning(f"No association found to delete: {source_hash} ↔ {target_hash}")
                    return False

        except sqlite3.Error as e:
            logger.error(f"Failed to delete association: {e}")
            return False

    async def get_association_count(self, memory_hash: str) -> int:
        """
        Get count of direct associations for a memory.

        Args:
            memory_hash: Memory content hash

        Returns:
            Number of direct associations (1-hop connections)
        """
        if not memory_hash:
            return 0

        try:
            conn = await self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*) as count
                FROM memory_graph
                WHERE source_hash = ?
            """, (memory_hash,))

            result = cursor.fetchone()
            return result['count'] if result else 0

        except sqlite3.Error as e:
            logger.error(f"Failed to count associations: {e}")
            return 0

    async def close(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Closed GraphStorage connection")
