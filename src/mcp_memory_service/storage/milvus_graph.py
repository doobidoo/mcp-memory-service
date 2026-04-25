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
Milvus-backed graph storage for memory associations.

Provides the same public async interface as ``GraphStorage`` (storage/graph.py)
but stores edges in a dedicated Milvus scalar collection and uses
application-layer BFS instead of SQLite recursive CTEs.

Design notes:
  * The association collection uses a deterministic primary key
    ``f"{source_hash}:{target_hash}"`` so that re-storing the same edge
    pair overwrites the previous record (upsert semantics).
  * Symmetric relationships (related, contradicts) store two records
    (A:B and B:A); asymmetric relationships store only the forward edge.
  * Graph traversal (find_connected, shortest_path, get_subgraph) is
    implemented as application-layer BFS with per-level Milvus queries.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from pymilvus import MilvusClient, DataType
    PYMILVUS_AVAILABLE = True
except ImportError:
    PYMILVUS_AVAILABLE = False
    MilvusClient = None  # type: ignore
    DataType = None  # type: ignore

from ..models.ontology import is_symmetric_relationship, validate_relationship

logger = logging.getLogger(__name__)


class MilvusGraphStorage:
    """Graph storage backed by a Milvus scalar collection for edges.

    Implements the same duck-typed interface as ``GraphStorage`` so that
    ``GraphService`` can accept either implementation without type-checking.
    """

    def __init__(
        self,
        uri: str = "./milvus.db",
        token: Optional[str] = None,
        collection_name: str = "mcp_memory",
    ) -> None:
        if not PYMILVUS_AVAILABLE:
            raise ImportError(
                "pymilvus is required for MilvusGraphStorage. "
                "Install with: pip install 'mcp-memory-service[milvus]'"
            )

        self.uri = uri
        self.token = token
        self.collection_name = f"{collection_name}_graph"
        self.client: Optional[MilvusClient] = None
        self._lock = asyncio.Lock()

        logger.info(
            "Initialized MilvusGraphStorage (uri=%s, collection=%s)",
            self.uri, self.collection_name,
        )

    # -- Initialization ------------------------------------------------------

    async def initialize(self) -> None:
        """Connect to Milvus and ensure the association collection exists."""
        await asyncio.to_thread(self._connect_client)
        await asyncio.to_thread(self._ensure_collection)

    def _connect_client(self) -> None:
        kwargs: Dict[str, Any] = {"uri": self.uri}
        if self.token:
            kwargs["token"] = self.token
        self.client = MilvusClient(**kwargs)

    def _ensure_collection(self) -> None:
        """Create the association collection if it does not already exist."""
        assert self.client is not None

        if self.client.has_collection(collection_name=self.collection_name):
            logger.info(
                "Reusing existing association collection '%s'",
                self.collection_name,
            )
            return

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=128,
        )
        schema.add_field(
            field_name="source_hash",
            datatype=DataType.VARCHAR,
            max_length=64,
        )
        schema.add_field(
            field_name="target_hash",
            datatype=DataType.VARCHAR,
            max_length=64,
        )
        schema.add_field(
            field_name="similarity",
            datatype=DataType.FLOAT,
        )
        schema.add_field(
            field_name="connection_types",
            datatype=DataType.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name="metadata",
            datatype=DataType.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name="relationship_type",
            datatype=DataType.VARCHAR,
            max_length=32,
        )
        schema.add_field(
            field_name="created_at",
            datatype=DataType.DOUBLE,
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="source_hash", index_type="Trie")
        index_params.add_index(field_name="target_hash", index_type="Trie")

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(
            "Created association collection '%s' with scalar indexes",
            self.collection_name,
        )

    # -- Helper --------------------------------------------------------------

    def _ensure_ready(self) -> bool:
        """Return True if the client is connected, False otherwise."""
        if self.client is None:
            logger.error("MilvusGraphStorage used before initialize()")
            return False
        return True

    async def _call_client(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Thread-safe Milvus client call."""
        async with self._lock:
            fn = getattr(self.client, method_name)
            return await asyncio.to_thread(fn, *args, **kwargs)

    # -- CRUD ----------------------------------------------------------------

    async def store_association(
        self,
        source_hash: str,
        target_hash: str,
        similarity: float,
        connection_types: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[float] = None,
        relationship_type: str = "related",
    ) -> bool:
        """Store a memory association edge (or bidirectional pair for symmetric types)."""
        if not self._ensure_ready():
            return False

        if not source_hash or not target_hash:
            logger.error("Invalid hash provided (empty string)")
            return False

        if source_hash == target_hash:
            logger.warning("Cannot create self-loop: %s", source_hash)
            return False

        if not (0.0 <= similarity <= 1.0):
            logger.error(
                "Invalid similarity score %s, must be in range [0.0, 1.0]",
                similarity,
            )
            return False

        if not validate_relationship(relationship_type):
            logger.error("Invalid relationship type: %s", relationship_type)
            return False

        try:
            if created_at is None:
                created_at = datetime.now(timezone.utc).timestamp()
            ct_json = json.dumps(connection_types)
            meta_json = json.dumps(metadata or {})

            rows = [
                {
                    "id": f"{source_hash}:{target_hash}",
                    "source_hash": source_hash,
                    "target_hash": target_hash,
                    "similarity": float(similarity),
                    "connection_types": ct_json,
                    "metadata": meta_json,
                    "relationship_type": relationship_type,
                    "created_at": float(created_at),
                }
            ]

            if is_symmetric_relationship(relationship_type):
                rows.append(
                    {
                        "id": f"{target_hash}:{source_hash}",
                        "source_hash": target_hash,
                        "target_hash": source_hash,
                        "similarity": float(similarity),
                        "connection_types": ct_json,
                        "metadata": meta_json,
                        "relationship_type": relationship_type,
                        "created_at": float(created_at),
                    }
                )
                logger.debug(
                    "Storing bidirectional association: %s ↔ %s (type: %s)",
                    source_hash, target_hash, relationship_type,
                )
            else:
                logger.debug(
                    "Storing directed association: %s → %s (type: %s)",
                    source_hash, target_hash, relationship_type,
                )

            await self._call_client(
                "upsert",
                collection_name=self.collection_name,
                data=rows,
            )
            return True

        except Exception as exc:
            logger.error("Failed to store association: %s", exc)
            return False

    # -- BFS traversal -------------------------------------------------------

    async def _query_edges(
        self,
        field: str,
        hashes: Set[str],
        relationship_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query edges where ``field`` IN ``hashes``, with optional relationship filter."""
        if not hashes:
            return []

        escaped = [h.replace('"', '\\"') for h in hashes]
        in_clause = ", ".join(f'"{h}"' for h in escaped)
        expr = f'{field} in [{in_clause}]'
        if relationship_type is not None:
            safe_rt = relationship_type.replace('"', '\\"')
            expr += f' and relationship_type == "{safe_rt}"'

        try:
            results = await self._call_client(
                "query",
                collection_name=self.collection_name,
                filter=expr,
                output_fields=[
                    "source_hash", "target_hash", "similarity",
                    "connection_types", "metadata", "relationship_type",
                    "created_at",
                ],
                limit=16384,
            )
            return results or []
        except Exception as exc:
            logger.error("Edge query failed (%s IN ...): %s", field, exc)
            return []

    async def _bfs(
        self,
        start_hash: str,
        max_hops: int,
        direction: str = "both",
        relationship_type: Optional[str] = None,
        stop_at: Optional[str] = None,
    ) -> Tuple[List[Tuple[str, int]], Optional[Dict[str, str]]]:
        """Application-layer BFS over the association collection.

        Returns:
            (reachable, parents) where reachable is a list of (hash, distance)
            and parents is a dict mapping each hash to its predecessor (used
            for path reconstruction when stop_at is set). If stop_at is None,
            parents is None.
        """
        visited: Set[str] = {start_hash}
        frontier: Set[str] = {start_hash}
        result: List[Tuple[str, int]] = []
        parents: Optional[Dict[str, str]] = {} if stop_at else None

        for depth in range(1, max_hops + 1):
            next_frontier: Set[str] = set()

            if direction in ("outgoing", "both"):
                rows = await self._query_edges("source_hash", frontier, relationship_type)
                for row in rows:
                    target = row["target_hash"]
                    if target not in visited:
                        visited.add(target)
                        next_frontier.add(target)
                        if parents is not None:
                            parents[target] = row["source_hash"]

            if direction in ("incoming", "both"):
                rows = await self._query_edges("target_hash", frontier, relationship_type)
                for row in rows:
                    source = row["source_hash"]
                    if source not in visited:
                        visited.add(source)
                        next_frontier.add(source)
                        if parents is not None:
                            parents[source] = row["target_hash"]

            for h in next_frontier:
                result.append((h, depth))

            if stop_at and stop_at in next_frontier:
                break  # early termination for shortest_path

            frontier = next_frontier
            if not frontier:
                break

        return result, parents

    async def find_connected(
        self,
        memory_hash: str,
        max_hops: int = 2,
        relationship_type: Optional[str] = None,
        direction: str = "both",
    ) -> List[Tuple[str, int]]:
        """Find all memories connected within N hops using BFS."""
        if not self._ensure_ready():
            return []

        if not memory_hash:
            logger.error("Invalid memory hash (empty string)")
            return []

        if direction not in ("outgoing", "incoming", "both"):
            logger.error(
                "Invalid direction '%s', must be 'outgoing', 'incoming', or 'both'",
                direction,
            )
            return []

        try:
            result, _ = await self._bfs(
                memory_hash,
                max_hops,
                direction=direction,
                relationship_type=relationship_type,
            )
            result.sort(key=lambda x: (x[1], x[0]))
            return result
        except Exception as exc:
            logger.error("Failed to find connected memories: %s", exc)
            return []

    async def shortest_path(
        self,
        hash1: str,
        hash2: str,
        max_depth: int = 5,
        relationship_types: Optional[List[str]] = None,
    ) -> Optional[List[str]]:
        """Find shortest path between two memories using BFS."""
        if not self._ensure_ready():
            return None

        if not hash1 or not hash2:
            logger.error("Invalid hash provided (empty string)")
            return None

        if hash1 == hash2:
            return [hash1]

        try:
            # BFS uses "both" direction and the first relationship type if
            # a filter list is provided (matching GraphStorage semantics).
            rt_filter = None
            if relationship_types and len(relationship_types) > 0:
                # For simplicity, filter by first type. Multi-type filtering
                # would require OR queries per level — acceptable trade-off.
                rt_filter = relationship_types[0] if len(relationship_types) == 1 else None

            result, parents = await self._bfs(
                hash1,
                max_depth,
                direction="both",
                relationship_type=rt_filter,
                stop_at=hash2,
            )

            # Check if hash2 was reached
            reached = {h for h, _ in result}
            if hash2 not in reached or parents is None:
                logger.debug("No path found between %s and %s", hash1, hash2)
                return None

            # Reconstruct path from parents dict
            path = [hash2]
            current = hash2
            while current != hash1:
                parent = parents.get(current)
                if parent is None:
                    logger.debug("Path reconstruction failed")
                    return None
                path.append(parent)
                current = parent
            path.reverse()

            logger.debug(
                "Found path of length %d: %s → %s", len(path), hash1, hash2,
            )
            return path

        except Exception as exc:
            logger.error("Failed to find shortest path: %s", exc)
            return None

    async def get_subgraph(
        self,
        memory_hash: str,
        radius: int = 2,
        relationship_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract subgraph centered on a memory for visualization."""
        if not self._ensure_ready():
            return {"nodes": [], "edges": []}

        if not memory_hash:
            logger.error("Invalid memory hash (empty string)")
            return {"nodes": [], "edges": []}

        try:
            connected = await self.find_connected(
                memory_hash,
                max_hops=radius,
                relationship_type=relationship_type,
            )

            nodes: Set[str] = {memory_hash}
            nodes.update(h for h, _ in connected)

            # Fetch all edges between nodes in the subgraph
            rows = await self._query_edges(
                "source_hash", nodes, relationship_type,
            )

            edges: List[Dict[str, Any]] = []
            seen_edges: Set[Tuple[str, str]] = set()

            for row in rows:
                source = row["source_hash"]
                target = row["target_hash"]

                # Only include edges where both endpoints are in the node set
                if target not in nodes:
                    continue

                # Canonical edge key for deduplication
                edge_key = tuple(sorted([source, target]))
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)

                # Parse JSON fields with fallback
                raw_ct = row.get("connection_types", "[]")
                try:
                    ct = json.loads(raw_ct) if raw_ct else []
                except (json.JSONDecodeError, TypeError):
                    ct = [raw_ct] if raw_ct else []

                raw_meta = row.get("metadata", "{}")
                try:
                    meta = json.loads(raw_meta) if raw_meta else {}
                except (json.JSONDecodeError, TypeError):
                    meta = {}

                edges.append({
                    "source": source,
                    "target": target,
                    "similarity": row.get("similarity", 0.0),
                    "connection_types": ct,
                    "metadata": meta,
                    "relationship_type": row.get("relationship_type", "related"),
                })

            subgraph = {"nodes": list(nodes), "edges": edges}
            logger.debug(
                "Extracted subgraph: %d nodes, %d edges",
                len(nodes), len(edges),
            )
            return subgraph

        except Exception as exc:
            logger.error("Failed to extract subgraph: %s", exc)
            return {"nodes": [], "edges": []}

    async def get_association(
        self,
        source_hash: str,
        target_hash: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a specific association between two memories."""
        if not self._ensure_ready():
            return None

        if not source_hash or not target_hash:
            logger.error("Invalid hash provided (empty string)")
            return None

        try:
            # Query for either direction
            sh_esc = source_hash.replace('"', '\\"')
            th_esc = target_hash.replace('"', '\\"')
            expr = (
                f'(source_hash == "{sh_esc}" and target_hash == "{th_esc}") or '
                f'(source_hash == "{th_esc}" and target_hash == "{sh_esc}")'
            )
            results = await self._call_client(
                "query",
                collection_name=self.collection_name,
                filter=expr,
                output_fields=[
                    "source_hash", "target_hash", "similarity",
                    "connection_types", "metadata", "created_at",
                ],
                limit=1,
            )
            if not results:
                logger.debug(
                    "No association found: %s ↔ %s", source_hash, target_hash,
                )
                return None

            row = results[0]
            raw_ct = row.get("connection_types", "[]")
            try:
                ct = json.loads(raw_ct) if raw_ct else []
            except (json.JSONDecodeError, TypeError):
                ct = [raw_ct] if raw_ct else []

            raw_meta = row.get("metadata", "{}")
            try:
                meta = json.loads(raw_meta) if raw_meta else {}
            except (json.JSONDecodeError, TypeError):
                meta = {}

            return {
                "source_hash": row["source_hash"],
                "target_hash": row["target_hash"],
                "similarity": row.get("similarity", 0.0),
                "connection_types": ct,
                "metadata": meta,
                "created_at": row.get("created_at", 0.0),
            }

        except Exception as exc:
            logger.error("Failed to retrieve association: %s", exc)
            return None

    async def delete_association(
        self,
        source_hash: str,
        target_hash: str,
    ) -> bool:
        """Delete association between two memories (both directions for safety)."""
        if not self._ensure_ready():
            return False

        if not source_hash or not target_hash:
            logger.error("Invalid hash provided (empty string)")
            return False

        try:
            # Delete both directions (matches GraphStorage behavior)
            ids_to_delete = [
                f"{source_hash}:{target_hash}",
                f"{target_hash}:{source_hash}",
            ]
            await self._call_client(
                "delete",
                collection_name=self.collection_name,
                ids=ids_to_delete,
            )
            logger.debug(
                "Deleted association: %s ↔ %s", source_hash, target_hash,
            )
            return True

        except Exception as exc:
            logger.error("Failed to delete association: %s", exc)
            return False

    async def get_association_count(self, memory_hash: str) -> int:
        """Get count of direct associations for a memory."""
        if not self._ensure_ready():
            return 0

        if not memory_hash:
            return 0

        try:
            h_esc = memory_hash.replace('"', '\\"')
            expr = f'source_hash == "{h_esc}"'
            results = await self._call_client(
                "query",
                collection_name=self.collection_name,
                filter=expr,
                output_fields=["id"],
                limit=16384,
            )
            return len(results) if results else 0

        except Exception as exc:
            logger.error("Failed to count associations: %s", exc)
            return 0

    async def get_relationship_types(self, memory_hash: str) -> Dict[str, int]:
        """Get count of each relationship type for a given memory."""
        if not self._ensure_ready():
            return {}

        if not memory_hash:
            logger.error("Invalid memory hash (empty string)")
            return {}

        try:
            h_esc = memory_hash.replace('"', '\\"')
            expr = f'source_hash == "{h_esc}"'
            results = await self._call_client(
                "query",
                collection_name=self.collection_name,
                filter=expr,
                output_fields=["relationship_type"],
                limit=16384,
            )
            if not results:
                return {}

            counts: Dict[str, int] = {}
            for row in results:
                rt = row.get("relationship_type", "related")
                counts[rt] = counts.get(rt, 0) + 1
            return counts

        except Exception as exc:
            logger.error("Failed to get relationship types: %s", exc)
            return {}

    async def close(self) -> None:
        """Close the Milvus client connection."""
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None
            logger.info("Closed MilvusGraphStorage connection")
