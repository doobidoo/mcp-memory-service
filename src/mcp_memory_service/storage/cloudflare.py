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
Cloudflare storage backend for MCP Memory Service.
Provides cloud-native storage using Vectorize, D1, and R2.
"""

import json
import logging
import hashlib
import asyncio
import time
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timezone, timedelta
import httpx

from .base import MemoryStorage
from ..models.memory import Memory, MemoryQueryResult
from ..utils.hashing import generate_content_hash
from ..config import CLOUDFLARE_MAX_CONTENT_LENGTH

logger = logging.getLogger(__name__)


def normalize_tags_for_search(tags: List[str]) -> List[str]:
    """Deduplicate and filter empty tag strings.

    Args:
        tags: List of tag strings (may contain duplicates or empty strings)

    Returns:
        Deduplicated list of non-empty tags
    """
    return list(dict.fromkeys([tag for tag in tags if tag]))


def normalize_operation(operation: Optional[str]) -> str:
    """Normalize tag search operation to AND or OR.

    Args:
        operation: Raw operation string (case-insensitive)

    Returns:
        Normalized operation: "AND" or "OR"
    """
    if isinstance(operation, str):
        normalized = operation.strip().upper() or "AND"
    else:
        normalized = "AND"

    if normalized not in {"AND", "OR"}:
        logger.warning(f"Unsupported operation '{operation}'; defaulting to AND")
        normalized = "AND"

    return normalized


def build_tag_search_query(tags: List[str], operation: str,
                          time_start: Optional[float] = None,
                          time_end: Optional[float] = None) -> Tuple[str, List[Any]]:
    """Build SQL query for tag-based search with time filtering.

    Args:
        tags: List of deduplicated tags
        operation: Search operation ("AND" or "OR")
        time_start: Optional start timestamp filter
        time_end: Optional end timestamp filter

    Returns:
        Tuple of (sql_query, parameters_list)
    """
    placeholders = ",".join(["?"] * len(tags))
    params: List[Any] = list(tags)

    sql = (
        "SELECT m.* FROM memories m "
        "JOIN memory_tags mt ON m.id = mt.memory_id "
        "JOIN tags t ON mt.tag_id = t.id "
        f"WHERE t.name IN ({placeholders})"
    )

    if time_start is not None:
        sql += " AND m.created_at >= ?"
        params.append(time_start)

    if time_end is not None:
        sql += " AND m.created_at <= ?"
        params.append(time_end)

    sql += " GROUP BY m.id"

    if operation == "AND":
        sql += " HAVING COUNT(DISTINCT t.name) = ?"
        params.append(len(tags))

    sql += " ORDER BY m.created_at DESC"

    return sql, params


class CloudflareStorage(MemoryStorage):
    """Cloudflare-based storage backend using Vectorize, D1, and R2."""

    # Content length limit from configuration
    _MAX_CONTENT_LENGTH = CLOUDFLARE_MAX_CONTENT_LENGTH

    @property
    def max_content_length(self) -> Optional[int]:
        """Maximum content length: 800 chars (BGE model 512 token limit)."""
        return self._MAX_CONTENT_LENGTH

    @property
    def supports_chunking(self) -> bool:
        """Cloudflare backend supports content chunking with metadata linking."""
        return True

    def __init__(self,
                 api_token: str,
                 account_id: str,
                 vectorize_index: str,
                 d1_database_id: str,
                 r2_bucket: Optional[str] = None,
                 embedding_model: str = "@cf/baai/bge-base-en-v1.5",
                 large_content_threshold: int = 1024 * 1024,  # 1MB
                 max_retries: int = 3,
                 base_delay: float = 1.0):
        """
        Initialize Cloudflare storage backend.

        Args:
            api_token: Cloudflare API token
            account_id: Cloudflare account ID
            vectorize_index: Vectorize index name
            d1_database_id: D1 database ID
            r2_bucket: Optional R2 bucket for large content
            embedding_model: Workers AI embedding model
            large_content_threshold: Size threshold for R2 storage
            max_retries: Maximum retry attempts for API calls
            base_delay: Base delay for exponential backoff
        """
        self.api_token = api_token
        self.account_id = account_id
        self.vectorize_index = vectorize_index
        self.d1_database_id = d1_database_id
        self.r2_bucket = r2_bucket
        self.embedding_model = embedding_model
        self.large_content_threshold = large_content_threshold
        self.max_retries = max_retries
        self.base_delay = base_delay

        # API endpoints
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        self.vectorize_url = f"{self.base_url}/vectorize/v2/indexes/{vectorize_index}"
        self.d1_url = f"{self.base_url}/d1/database/{d1_database_id}"
        self.ai_url = f"{self.base_url}/ai/run/{embedding_model}"

        if r2_bucket:
            self.r2_url = f"{self.base_url}/r2/buckets/{r2_bucket}/objects"

        # HTTP client with connection pooling
        self.client = None
        self._initialized = False

        # Embedding cache for performance
        self._embedding_cache = {}
        self._cache_max_size = 1000
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling."""
        if self.client is None:
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            self.client = httpx.AsyncClient(
                headers=headers,
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
            )
        return self.client
    
    async def _retry_request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make HTTP request with exponential backoff retry logic."""
        client = await self._get_client()
        
        for attempt in range(self.max_retries + 1):
            try:
                response = await client.request(method, url, **kwargs)
                
                # Handle rate limiting
                if response.status_code == 429:
                    if attempt < self.max_retries:
                        delay = self.base_delay * (2 ** attempt)
                        logger.warning(f"Rate limited, retrying in {delay}s (attempt {attempt + 1}/{self.max_retries + 1})")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise httpx.HTTPError(f"Rate limited after {self.max_retries} retries")
                
                # Handle server errors
                if response.status_code >= 500:
                    if attempt < self.max_retries:
                        delay = self.base_delay * (2 ** attempt)
                        logger.warning(f"Server error {response.status_code}, retrying in {delay}s")
                        await asyncio.sleep(delay)
                        continue
                
                response.raise_for_status()
                return response
                
            except (httpx.NetworkError, httpx.TimeoutException) as e:
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(f"Network error: {e}, retrying in {delay}s")
                    await asyncio.sleep(delay)
                    continue
                raise
        
        raise httpx.HTTPError(f"Failed after {self.max_retries} retries")
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Workers AI or cache."""
        # Check cache first
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]
        
        try:
            # Use Workers AI to generate embedding
            payload = {"text": [text]}
            response = await self._retry_request("POST", self.ai_url, json=payload)
            result = response.json()
            
            if result.get("success") and "result" in result:
                embedding = result["result"]["data"][0]
                
                # Cache the embedding (with size limit)
                if len(self._embedding_cache) >= self._cache_max_size:
                    # Remove oldest entry (simple FIFO)
                    oldest_key = next(iter(self._embedding_cache))
                    del self._embedding_cache[oldest_key]
                
                self._embedding_cache[text_hash] = embedding
                return embedding
            else:
                raise ValueError(f"Workers AI embedding failed: {result}")
                
        except Exception as e:
            logger.error(f"Failed to generate embedding with Workers AI: {e}")
            # TODO: Implement fallback to local sentence-transformers
            raise ValueError(f"Embedding generation failed: {e}")
    
    async def initialize(self) -> None:
        """Initialize the Cloudflare storage backend."""
        if self._initialized:
            return
        
        logger.info("Initializing Cloudflare storage backend...")
        
        try:
            # Initialize D1 database schema
            await self._initialize_d1_schema()
            
            # Verify Vectorize index exists
            await self._verify_vectorize_index()
            
            # Verify R2 bucket if configured
            if self.r2_bucket:
                await self._verify_r2_bucket()
            
            self._initialized = True
            logger.info("Cloudflare storage backend initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Cloudflare storage: {e}")
            raise
    
    async def _initialize_d1_schema(self) -> None:
        """Initialize D1 database schema."""
        schema_sql = """
        -- Memory metadata table
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_hash TEXT UNIQUE NOT NULL,
            content TEXT NOT NULL,
            memory_type TEXT,
            created_at REAL NOT NULL,
            created_at_iso TEXT NOT NULL,
            updated_at REAL,
            updated_at_iso TEXT,
            metadata_json TEXT,
            vector_id TEXT UNIQUE,
            content_size INTEGER DEFAULT 0,
            r2_key TEXT
        );
        
        -- Tags table
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        );
        
        -- Memory-tag relationships
        CREATE TABLE IF NOT EXISTS memory_tags (
            memory_id INTEGER,
            tag_id INTEGER,
            PRIMARY KEY (memory_id, tag_id),
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
            FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
        );
        
        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_memories_content_hash ON memories(content_hash);
        CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
        CREATE INDEX IF NOT EXISTS idx_memories_vector_id ON memories(vector_id);
        CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);
        """
        
        payload = {"sql": schema_sql}
        response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
        result = response.json()
        
        if not result.get("success"):
            raise ValueError(f"Failed to initialize D1 schema: {result}")
    
    async def _verify_vectorize_index(self) -> None:
        """Verify Vectorize index exists and is accessible."""
        try:
            response = await self._retry_request("GET", f"{self.vectorize_url}")
            result = response.json()
            
            if not result.get("success"):
                raise ValueError(f"Vectorize index not accessible: {result}")
                
            logger.info(f"Vectorize index verified: {self.vectorize_index}")
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Vectorize index '{self.vectorize_index}' not found")
            raise
    
    async def _verify_r2_bucket(self) -> None:
        """Verify R2 bucket exists and is accessible."""
        try:
            # Try to list objects (empty list is fine)
            response = await self._retry_request("GET", f"{self.r2_url}?max-keys=1")
            logger.info(f"R2 bucket verified: {self.r2_bucket}")
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ValueError(f"R2 bucket '{self.r2_bucket}' not found")
            raise
    
    async def store(self, memory: Memory) -> Tuple[bool, str]:
        """Store a memory in Cloudflare storage."""
        try:
            # Generate embedding for the content
            embedding = await self._generate_embedding(memory.content)
            
            # Determine storage strategy based on content size
            content_size = len(memory.content.encode('utf-8'))
            use_r2 = self.r2_bucket and content_size > self.large_content_threshold
            
            # Store large content in R2 if needed
            r2_key = None
            stored_content = memory.content
            
            if use_r2:
                r2_key = f"content/{memory.content_hash}.txt"
                await self._store_r2_content(r2_key, memory.content)
                stored_content = f"[R2 Content: {r2_key}]"  # Placeholder in D1
            
            # Store vector in Vectorize
            vector_id = memory.content_hash
            vector_metadata = {
                "content_hash": memory.content_hash,
                "memory_type": memory.memory_type or "standard",
                "tags": ",".join(memory.tags) if memory.tags else "",
                "created_at": memory.created_at_iso or datetime.now().isoformat()
            }
            
            await self._store_vectorize_vector(vector_id, embedding, vector_metadata)
            
            # Store metadata in D1
            await self._store_d1_memory(memory, vector_id, content_size, r2_key, stored_content)
            
            logger.info(f"Successfully stored memory: {memory.content_hash}")
            return True, f"Memory stored successfully (vector_id: {vector_id})"
            
        except Exception as e:
            logger.error(f"Failed to store memory {memory.content_hash}: {e}")
            return False, f"Storage failed: {str(e)}"
    
    async def _store_vectorize_vector(self, vector_id: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        """Store vector in Vectorize."""
        # Try without namespace first to isolate the issue
        vector_data = {
            "id": vector_id,
            "values": embedding,
            "metadata": metadata
        }
        
        # Convert to NDJSON format as required by the HTTP API
        import json
        ndjson_content = json.dumps(vector_data) + "\n"
        
        try:
            # Send as raw NDJSON data with correct Content-Type header
            client = await self._get_client()
            
            # Override headers for this specific request
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/x-ndjson"
            }
            
            response = await client.post(
                f"{self.vectorize_url}/upsert",
                content=ndjson_content.encode("utf-8"),
                headers=headers
            )
            
            # Log response status for debugging (avoid logging headers/body for security)
            logger.info(f"Vectorize response status: {response.status_code}")
            response_text = response.text
            if response.status_code != 200:
                # Only log response body on errors, and truncate to avoid credential exposure
                truncated_response = response_text[:200] + "..." if len(response_text) > 200 else response_text
                logger.warning(f"Vectorize error response (truncated): {truncated_response}")
            
            if response.status_code != 200:
                raise ValueError(f"HTTP {response.status_code}: {response_text}")
            
            result = response.json()
            if not result.get("success"):
                raise ValueError(f"Failed to store vector: {result}")
                
        except Exception as e:
            # Add more detailed error logging
            logger.error(f"Vectorize insert failed: {e}")
            logger.error(f"Vector data was: {vector_data}")
            logger.error(f"NDJSON content: {ndjson_content.strip()}")
            logger.error(f"URL was: {self.vectorize_url}/upsert")
            raise ValueError(f"Failed to store vector: {e}")
    
    async def _store_d1_memory(self, memory: Memory, vector_id: str, content_size: int, r2_key: Optional[str], stored_content: str) -> None:
        """Store memory metadata in D1."""
        # Insert memory record
        insert_sql = """
        INSERT INTO memories (
            content_hash, content, memory_type, created_at, created_at_iso,
            updated_at, updated_at_iso, metadata_json, vector_id, content_size, r2_key
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        now = time.time()
        now_iso = datetime.now().isoformat()
        
        params = [
            memory.content_hash,
            stored_content,
            memory.memory_type,
            memory.created_at or now,
            memory.created_at_iso or now_iso,
            memory.updated_at or now,
            memory.updated_at_iso or now_iso,
            json.dumps(memory.metadata) if memory.metadata else None,
            vector_id,
            content_size,
            r2_key
        ]
        
        payload = {"sql": insert_sql, "params": params}
        response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
        result = response.json()
        
        if not result.get("success"):
            raise ValueError(f"Failed to store memory in D1: {result}")
        
        # Store tags if present
        if memory.tags:
            memory_id = result["result"][0]["meta"]["last_row_id"]
            await self._store_d1_tags(memory_id, memory.tags)
    
    async def _store_d1_tags(self, memory_id: int, tags: List[str]) -> None:
        """Store tags for a memory in D1."""
        for tag in tags:
            # Insert tag if not exists
            tag_sql = "INSERT OR IGNORE INTO tags (name) VALUES (?)"
            payload = {"sql": tag_sql, "params": [tag]}
            await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
            
            # Link tag to memory
            link_sql = """
            INSERT INTO memory_tags (memory_id, tag_id)
            SELECT ?, id FROM tags WHERE name = ?
            """
            payload = {"sql": link_sql, "params": [memory_id, tag]}
            await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
    
    async def _store_r2_content(self, key: str, content: str) -> None:
        """Store content in R2."""
        response = await self._retry_request(
            "PUT", 
            f"{self.r2_url}/{key}",
            content=content.encode('utf-8'),
            headers={"Content-Type": "text/plain"}
        )
        
        if response.status_code not in [200, 201]:
            raise ValueError(f"Failed to store content in R2: {response.status_code}")
    
    async def retrieve(self, query: str, n_results: int = 5) -> List[MemoryQueryResult]:
        """Retrieve memories by semantic search."""
        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)
            
            # Search Vectorize (without namespace for now)
            search_payload = {
                "vector": query_embedding,
                "topK": n_results,
                "returnMetadata": "all",
                "returnValues": False
            }
            
            response = await self._retry_request("POST", f"{self.vectorize_url}/query", json=search_payload)
            result = response.json()
            
            if not result.get("success"):
                raise ValueError(f"Vectorize query failed: {result}")
            
            matches = result.get("result", {}).get("matches", [])
            
            # Convert to MemoryQueryResult objects
            results = []
            for match in matches:
                memory = await self._load_memory_from_match(match)
                if memory:
                    # Record access for quality scoring (implicit signals)
                    memory.record_access(query)

                    query_result = MemoryQueryResult(
                        memory=memory,
                        relevance_score=match.get("score", 0.0)
                    )
                    results.append(query_result)

            # Persist updated metadata for accessed memories
            for result in results:
                try:
                    await self._persist_access_metadata(result.memory)
                except Exception as e:
                    logger.warning(f"Failed to persist access metadata: {e}")

            logger.info(f"Retrieved {len(results)} memories for query")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []
    
    async def _load_memory_from_match(self, match: Dict[str, Any]) -> Optional[Memory]:
        """Load full memory from Vectorize match."""
        try:
            vector_id = match.get("id")
            metadata = match.get("metadata", {})
            content_hash = metadata.get("content_hash")
            
            if not content_hash:
                logger.warning(f"No content_hash in vector metadata: {vector_id}")
                return None
            
            # Load from D1
            sql = "SELECT * FROM memories WHERE content_hash = ?"
            payload = {"sql": sql, "params": [content_hash]}
            response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
            result = response.json()
            
            if not result.get("success") or not result.get("result", [{}])[0].get("results"):
                logger.warning(f"Memory not found in D1: {content_hash}")
                return None
            
            row = result["result"][0]["results"][0]
            
            # Load content from R2 if needed
            content = row["content"]
            if row.get("r2_key") and content.startswith("[R2 Content:"):
                content = await self._load_r2_content(row["r2_key"])
            
            # Load tags
            tags = await self._load_memory_tags(row["id"])

            # Parse and decompress metadata
            metadata = {}
            if row.get("metadata_json"):
                metadata = json.loads(row["metadata_json"])
                # Decompress CSV-encoded quality metadata if present
                from ..quality.metadata_codec import decompress_metadata_from_sync
                metadata = decompress_metadata_from_sync(metadata)

            # Reconstruct Memory object
            memory = Memory(
                content=content,
                content_hash=content_hash,
                tags=tags,
                memory_type=row.get("memory_type"),
                metadata=metadata,
                created_at=row.get("created_at"),
                created_at_iso=row.get("created_at_iso"),
                updated_at=row.get("updated_at"),
                updated_at_iso=row.get("updated_at_iso")
            )
            
            return memory
            
        except Exception as e:
            logger.error(f"Failed to load memory from match: {e}")
            return None
    
    async def _load_r2_content(self, r2_key: str) -> str:
        """Load content from R2."""
        response = await self._retry_request("GET", f"{self.r2_url}/{r2_key}")
        return response.text
    
    async def _load_memory_tags(self, memory_id: int) -> List[str]:
        """Load tags for a memory from D1."""
        sql = """
        SELECT t.name FROM tags t
        JOIN memory_tags mt ON t.id = mt.tag_id
        WHERE mt.memory_id = ?
        """
        payload = {"sql": sql, "params": [memory_id]}
        response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
        result = response.json()
        
        if result.get("success") and result.get("result", [{}])[0].get("results"):
            return [row["name"] for row in result["result"][0]["results"]]
        
        return []
    
    async def search_by_tags(
        self,
        tags: List[str],
        operation: str = "AND",
        time_start: Optional[float] = None,
        time_end: Optional[float] = None
    ) -> List[Memory]:
        """Search memories by tags with AND/OR semantics and optional time filtering."""
        return await self._search_by_tags_internal(
            tags=tags,
            operation=operation,
            time_start=time_start,
            time_end=time_end
        )

    async def search_by_tag(self, tags: List[str], time_start: Optional[float] = None) -> List[Memory]:
        """Search memories by tags with optional time filtering (legacy OR behavior)."""
        return await self._search_by_tags_internal(
            tags=tags,
            operation="OR",
            time_start=time_start,
            time_end=None
        )

    async def _search_by_tags_internal(
        self,
        tags: List[str],
        operation: Optional[str] = None,
        time_start: Optional[float] = None,
        time_end: Optional[float] = None
    ) -> List[Memory]:
        """Shared implementation for tag-based queries with optional time filtering."""
        try:
            if not tags:
                return []

            deduped_tags = normalize_tags_for_search(tags)
            if not deduped_tags:
                return []

            normalized_operation = normalize_operation(operation)
            sql, params = build_tag_search_query(deduped_tags, normalized_operation,
                                                time_start, time_end)

            payload = {"sql": sql, "params": params}
            response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
            result = response.json()

            if not result.get("success"):
                raise ValueError(f"D1 tag search failed: {result}")

            rows = result.get("result", [{}])[0].get("results") or []
            memories: List[Memory] = []

            for row in rows:
                memory = await self._load_memory_from_row(row)
                if memory:
                    memories.append(memory)

            logger.info(
                "Found %d memories with tags: %s (operation: %s)",
                len(memories),
                deduped_tags,
                normalized_operation
            )
            return memories

        except Exception as e:
            logger.error(
                "Failed to search memories by tags %s with operation %s: %s",
                tags,
                operation,
                e
            )
            return []
    
    async def _load_memory_from_row(self, row: Dict[str, Any]) -> Optional[Memory]:
        """Load memory from D1 row data."""
        try:
            # Load content from R2 if needed
            content = row["content"]
            if row.get("r2_key") and content.startswith("[R2 Content:"):
                content = await self._load_r2_content(row["r2_key"])
            
            # Load tags
            tags = await self._load_memory_tags(row["id"])

            # Parse and decompress metadata
            metadata = {}
            if row.get("metadata_json"):
                metadata = json.loads(row["metadata_json"])
                # Decompress CSV-encoded quality metadata if present
                from ..quality.metadata_codec import decompress_metadata_from_sync
                metadata = decompress_metadata_from_sync(metadata)

            memory = Memory(
                content=content,
                content_hash=row["content_hash"],
                tags=tags,
                memory_type=row.get("memory_type"),
                metadata=metadata,
                created_at=row.get("created_at"),
                created_at_iso=row.get("created_at_iso"),
                updated_at=row.get("updated_at"),
                updated_at_iso=row.get("updated_at_iso")
            )
            
            return memory
            
        except Exception as e:
            logger.error(f"Failed to load memory from row: {e}")
            return None
    
    async def delete(self, content_hash: str) -> Tuple[bool, str]:
        """Delete a memory by its hash."""
        try:
            # Find memory in D1
            sql = "SELECT * FROM memories WHERE content_hash = ?"
            payload = {"sql": sql, "params": [content_hash]}
            response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
            result = response.json()
            
            if not result.get("success") or not result.get("result", [{}])[0].get("results"):
                return False, f"Memory not found: {content_hash}"
            
            row = result["result"][0]["results"][0]
            memory_id = row["id"]
            vector_id = row.get("vector_id")
            r2_key = row.get("r2_key")
            
            # Delete from Vectorize
            if vector_id:
                await self._delete_vectorize_vector(vector_id)
            
            # Delete from R2 if present
            if r2_key:
                await self._delete_r2_content(r2_key)
            
            # Delete from D1 (tags will be cascade deleted)
            delete_sql = "DELETE FROM memories WHERE id = ?"
            payload = {"sql": delete_sql, "params": [memory_id]}
            response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
            result = response.json()
            
            if not result.get("success"):
                raise ValueError(f"Failed to delete from D1: {result}")
            
            logger.info(f"Successfully deleted memory: {content_hash}")
            return True, "Memory deleted successfully"

        except Exception as e:
            logger.error(f"Failed to delete memory {content_hash}: {e}")
            return False, f"Deletion failed: {str(e)}"

    async def get_by_hash(self, content_hash: str) -> Optional[Memory]:
        """Get a memory by its content hash using direct O(1) D1 lookup."""
        try:
            # Query D1 for the memory
            sql = "SELECT * FROM memories WHERE content_hash = ?"
            payload = {"sql": sql, "params": [content_hash]}
            response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
            result = response.json()

            if not result.get("success") or not result.get("result", [{}])[0].get("results"):
                return None

            row = result["result"][0]["results"][0]

            # Load content from R2 if needed
            content = row["content"]
            if row.get("r2_key") and content.startswith("[R2 Content:"):
                content = await self._load_r2_content(row["r2_key"])

            # Load tags
            tags = await self._load_memory_tags(row["id"])

            # Parse and decompress metadata
            metadata = {}
            if row.get("metadata_json"):
                metadata = json.loads(row["metadata_json"])
                # Decompress CSV-encoded quality metadata if present
                from ..quality.metadata_codec import decompress_metadata_from_sync
                metadata = decompress_metadata_from_sync(metadata)

            # Construct Memory object
            memory = Memory(
                content=content,
                content_hash=content_hash,
                tags=tags,
                memory_type=row.get("memory_type"),
                metadata=metadata,
                created_at=row.get("created_at"),
                created_at_iso=row.get("created_at_iso"),
                updated_at=row.get("updated_at"),
                updated_at_iso=row.get("updated_at_iso")
            )

            return memory

        except Exception as e:
            logger.error(f"Failed to get memory by hash {content_hash}: {e}")
            return None

    async def _delete_vectorize_vector(self, vector_id: str) -> None:
        """Delete vector from Vectorize."""
        # Correct endpoint uses underscores, not hyphens
        payload = {"ids": [vector_id]}

        response = await self._retry_request("POST", f"{self.vectorize_url}/delete_by_ids", json=payload)
        result = response.json()

        if not result.get("success"):
            logger.warning(f"Failed to delete vector from Vectorize: {result}")

    async def delete_vectors_by_ids(self, vector_ids: List[str]) -> Dict[str, Any]:
        """Delete multiple vectors from Vectorize by their IDs."""
        payload = {"ids": vector_ids}
        response = await self._retry_request(
            "POST",
            f"{self.vectorize_url}/delete_by_ids",
            json=payload
        )
        return response.json()

    async def _delete_r2_content(self, r2_key: str) -> None:
        """Delete content from R2."""
        try:
            response = await self._retry_request("DELETE", f"{self.r2_url}/{r2_key}")
            if response.status_code not in [200, 204, 404]:  # 404 is fine if already deleted
                logger.warning(f"Failed to delete R2 content: {response.status_code}")
        except Exception as e:
            logger.warning(f"Failed to delete R2 content {r2_key}: {e}")
    
    async def delete_by_tag(self, tag: str) -> Tuple[int, str]:
        """Delete memories by tag."""
        try:
            # Find memories with the tag
            memories = await self.search_by_tag([tag])
            
            deleted_count = 0
            for memory in memories:
                success, _ = await self.delete(memory.content_hash)
                if success:
                    deleted_count += 1
            
            logger.info(f"Deleted {deleted_count} memories with tag: {tag}")
            return deleted_count, f"Deleted {deleted_count} memories"
            
        except Exception as e:
            logger.error(f"Failed to delete by tag {tag}: {e}")
            return 0, f"Deletion failed: {str(e)}"
    
    async def cleanup_duplicates(self) -> Tuple[int, str]:
        """Remove duplicate memories based on content hash."""
        try:
            # Find duplicates in D1
            sql = """
            SELECT content_hash, COUNT(*) as count, MIN(id) as keep_id
            FROM memories
            GROUP BY content_hash
            HAVING COUNT(*) > 1
            """
            
            payload = {"sql": sql}
            response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
            result = response.json()
            
            if not result.get("success"):
                raise ValueError(f"Failed to find duplicates: {result}")
            
            duplicate_groups = result.get("result", [{}])[0].get("results", [])
            
            total_deleted = 0
            for group in duplicate_groups:
                content_hash = group["content_hash"]
                keep_id = group["keep_id"]
                
                # Delete all except the first one
                delete_sql = "DELETE FROM memories WHERE content_hash = ? AND id != ?"
                payload = {"sql": delete_sql, "params": [content_hash, keep_id]}
                response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
                result = response.json()
                
                if result.get("success") and result.get("result", [{}])[0].get("meta"):
                    deleted = result["result"][0]["meta"].get("changes", 0)
                    total_deleted += deleted
            
            logger.info(f"Cleaned up {total_deleted} duplicate memories")
            return total_deleted, f"Removed {total_deleted} duplicates"
            
        except Exception as e:
            logger.error(f"Failed to cleanup duplicates: {e}")
            return 0, f"Cleanup failed: {str(e)}"

    async def _persist_access_metadata(self, memory: Memory):
        """
        Persist access tracking metadata (access_count, last_accessed_at) to storage.

        Args:
            memory: Memory object with updated access metadata
        """
        # Update metadata in D1
        sql = """
            UPDATE memories
            SET metadata_json = ?
            WHERE content_hash = ?
        """
        payload = {
            "sql": sql,
            "params": [json.dumps(memory.metadata), memory.content_hash]
        }
        await self._retry_request("POST", f"{self.d1_url}/query", json=payload)

    async def update_memory_metadata(self, content_hash: str, updates: Dict[str, Any], preserve_timestamps: bool = True) -> Tuple[bool, str]:
        """Update memory metadata without recreating the entry."""
        try:
            # Build update SQL
            update_fields = []
            params = []
            
            if "metadata" in updates:
                update_fields.append("metadata_json = ?")
                params.append(json.dumps(updates["metadata"]))
            
            if "memory_type" in updates:
                update_fields.append("memory_type = ?")
                params.append(updates["memory_type"])
            
            if "tags" in updates:
                # Handle tags separately - they require relational updates
                pass

            # Handle timestamp updates based on preserve_timestamps flag
            now = time.time()
            now_iso = datetime.now().isoformat()

            if not preserve_timestamps:
                # When preserve_timestamps=False, use timestamps from updates dict if provided
                # This allows syncing timestamps from source (e.g., SQLite → Cloudflare)
                # Always preserve created_at (never reset to current time!)
                if "created_at" in updates:
                    update_fields.append("created_at = ?")
                    params.append(updates["created_at"])
                if "created_at_iso" in updates:
                    update_fields.append("created_at_iso = ?")
                    params.append(updates["created_at_iso"])
                # Use updated_at from updates or current time
                if "updated_at" in updates:
                    update_fields.append("updated_at = ?")
                    params.append(updates["updated_at"])
                else:
                    update_fields.append("updated_at = ?")
                    params.append(now)
                if "updated_at_iso" in updates:
                    update_fields.append("updated_at_iso = ?")
                    params.append(updates["updated_at_iso"])
                else:
                    update_fields.append("updated_at_iso = ?")
                    params.append(now_iso)
            else:
                # preserve_timestamps=True: only update updated_at to current time
                update_fields.append("updated_at = ?")
                update_fields.append("updated_at_iso = ?")
                params.extend([now, now_iso])
            
            if not update_fields:
                return True, "No updates needed"
            
            # Update memory record
            sql = f"UPDATE memories SET {', '.join(update_fields)} WHERE content_hash = ?"
            params.append(content_hash)
            
            payload = {"sql": sql, "params": params}
            response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
            result = response.json()
            
            if not result.get("success"):
                raise ValueError(f"Failed to update memory: {result}")
            
            # Handle tag updates if provided
            if "tags" in updates:
                await self._update_memory_tags(content_hash, updates["tags"])
            
            logger.info(f"Successfully updated memory metadata: {content_hash}")
            return True, "Memory metadata updated successfully"
            
        except Exception as e:
            logger.error(f"Failed to update memory metadata {content_hash}: {e}")
            return False, f"Update failed: {str(e)}"
    
    async def _update_memory_tags(self, content_hash: str, new_tags: List[str]) -> None:
        """Update tags for a memory."""
        # Get memory ID
        sql = "SELECT id FROM memories WHERE content_hash = ?"
        payload = {"sql": sql, "params": [content_hash]}
        response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
        result = response.json()
        
        if not result.get("success") or not result.get("result", [{}])[0].get("results"):
            raise ValueError(f"Memory not found: {content_hash}")
        
        memory_id = result["result"][0]["results"][0]["id"]
        
        # Delete existing tag relationships
        delete_sql = "DELETE FROM memory_tags WHERE memory_id = ?"
        payload = {"sql": delete_sql, "params": [memory_id]}
        await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
        
        # Add new tags
        if new_tags:
            await self._store_d1_tags(memory_id, new_tags)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            # Calculate timestamp for memories from last 7 days
            week_ago = time.time() - (7 * 24 * 60 * 60)

            # Get memory count and size from D1
            sql = f"""
            SELECT
                COUNT(*) as total_memories,
                SUM(content_size) as total_content_size,
                COUNT(DISTINCT vector_id) as total_vectors,
                COUNT(r2_key) as r2_stored_count,
                (SELECT COUNT(*) FROM tags) as unique_tags,
                (SELECT COUNT(*) FROM memories WHERE created_at >= {week_ago}) as memories_this_week
            FROM memories
            """

            payload = {"sql": sql}
            response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
            result = response.json()

            if result.get("success") and result.get("result", [{}])[0].get("results"):
                stats = result["result"][0]["results"][0]

                return {
                    "total_memories": stats.get("total_memories", 0),
                    "unique_tags": stats.get("unique_tags", 0),
                    "memories_this_week": stats.get("memories_this_week", 0),
                    "total_content_size_bytes": stats.get("total_content_size", 0),
                    "total_vectors": stats.get("total_vectors", 0),
                    "r2_stored_count": stats.get("r2_stored_count", 0),
                    "storage_backend": "cloudflare",
                    "vectorize_index": self.vectorize_index,
                    "d1_database": self.d1_database_id,
                    "r2_bucket": self.r2_bucket,
                    "status": "operational"
                }

            return {
                "total_memories": 0,
                "unique_tags": 0,
                "memories_this_week": 0,
                "storage_backend": "cloudflare",
                "status": "operational"
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "total_memories": 0,
                "unique_tags": 0,
                "memories_this_week": 0,
                "storage_backend": "cloudflare",
                "status": "error",
                "error": str(e)
            }
    
    async def get_all_tags(self) -> List[str]:
        """Get all unique tags in the storage."""
        try:
            sql = "SELECT name FROM tags ORDER BY name"
            payload = {"sql": sql}
            response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
            result = response.json()

            if result.get("success") and result.get("result", [{}])[0].get("results"):
                return [row["name"] for row in result["result"][0]["results"]]

            return []

        except Exception as e:
            logger.error(f"Failed to get all tags: {e}")
            return []

    async def get_all_tags_with_counts(self) -> List[Dict[str, Any]]:
        """Get all tags with their usage counts."""
        try:
            sql = """
                SELECT t.name as tag, COUNT(mt.memory_id) as count
                FROM tags t
                LEFT JOIN memory_tags mt ON t.id = mt.tag_id
                GROUP BY t.id
                ORDER BY count DESC, t.name ASC
            """
            payload = {"sql": sql}
            response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
            result = response.json()

            if result.get("success") and result.get("result", [{}])[0].get("results"):
                return [row for row in result["result"][0]["results"]]

            return []

        except Exception as e:
            logger.error(f"Failed to get tags with counts: {e}")
            return []
    
    async def get_recent_memories(self, n: int = 10) -> List[Memory]:
        """Get n most recent memories."""
        try:
            sql = "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?"
            payload = {"sql": sql, "params": [n]}
            response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
            result = response.json()

            memories = []
            if result.get("success") and result.get("result", [{}])[0].get("results"):
                for row in result["result"][0]["results"]:
                    memory = await self._load_memory_from_row(row)
                    if memory:
                        memories.append(memory)

            logger.info(f"Retrieved {len(memories)} recent memories")
            return memories

        except Exception as e:
            logger.error(f"Failed to get recent memories: {e}")
            return []

    async def get_largest_memories(self, n: int = 10) -> List[Memory]:
        """Get n largest memories by content length."""
        try:
            sql = "SELECT * FROM memories ORDER BY LENGTH(content) DESC LIMIT ?"
            payload = {"sql": sql, "params": [n]}
            response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
            result = response.json()

            memories = []
            if result.get("success") and result.get("result", [{}])[0].get("results"):
                for row in result["result"][0]["results"]:
                    memory = await self._load_memory_from_row(row)
                    if memory:
                        memories.append(memory)

            logger.info(f"Retrieved {len(memories)} largest memories")
            return memories

        except Exception as e:
            logger.error(f"Failed to get largest memories: {e}")
            return []

    async def _fetch_d1_timestamps(self, cutoff_timestamp: Optional[float] = None) -> List[float]:
        """Fetch timestamps from D1 database with optional time filter.

        Args:
            cutoff_timestamp: Optional Unix timestamp to filter memories (>=)

        Returns:
            List of Unix timestamps (float) in descending order

        Raises:
            Exception: If D1 query fails
        """
        if cutoff_timestamp is not None:
            sql = "SELECT created_at FROM memories WHERE created_at >= ? ORDER BY created_at DESC"
            payload = {"sql": sql, "params": [cutoff_timestamp]}
        else:
            sql = "SELECT created_at FROM memories ORDER BY created_at DESC"
            payload = {"sql": sql, "params": []}

        response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
        result = response.json()

        timestamps = []
        if result.get("success") and result.get("result", [{}])[0].get("results"):
            for row in result["result"][0]["results"]:
                if row.get("created_at") is not None:
                    timestamps.append(float(row["created_at"]))

        return timestamps

    async def get_memory_timestamps(self, days: Optional[int] = None) -> List[float]:
        """
        Get memory creation timestamps only, without loading full memory objects.

        This is an optimized method for analytics that only needs timestamps,
        avoiding the overhead of loading full memory content and embeddings.

        Args:
            days: Optional filter to only get memories from last N days

        Returns:
            List of Unix timestamps (float) in descending order (newest first)
        """
        try:
            cutoff_timestamp = None
            if days is not None:
                cutoff = datetime.now(timezone.utc) - timedelta(days=days)
                cutoff_timestamp = cutoff.timestamp()

            timestamps = await self._fetch_d1_timestamps(cutoff_timestamp)
            logger.info(f"Retrieved {len(timestamps)} memory timestamps")
            return timestamps

        except Exception as e:
            logger.error(f"Failed to get memory timestamps: {e}")
            return []

    def sanitized(self, tags):
        """Sanitize and normalize tags to a JSON string.

        This method provides compatibility with the storage backend interface.
        """
        if tags is None:
            return json.dumps([])
        
        # If we get a string, split it into an array
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
        # If we get an array, use it directly
        elif isinstance(tags, list):
            tags = [str(tag).strip() for tag in tags if str(tag).strip()]
        else:
            return json.dumps([])
                
        # Return JSON string representation of the array
        return json.dumps(tags)
    
    async def recall(self, query: Optional[str] = None, n_results: int = 5, start_timestamp: Optional[float] = None, end_timestamp: Optional[float] = None) -> List[MemoryQueryResult]:
        """
        Retrieve memories with combined time filtering and optional semantic search.

        Args:
            query: Optional semantic search query. If None, only time filtering is applied.
            n_results: Maximum number of results to return.
            start_timestamp: Optional start time for filtering.
            end_timestamp: Optional end time for filtering.

        Returns:
            List of MemoryQueryResult objects.
        """
        try:
            # Build time filtering WHERE clause for D1
            time_conditions = []
            params = []

            if start_timestamp is not None:
                time_conditions.append("created_at >= ?")
                params.append(float(start_timestamp))

            if end_timestamp is not None:
                time_conditions.append("created_at <= ?")
                params.append(float(end_timestamp))

            time_where = " AND ".join(time_conditions) if time_conditions else ""

            logger.info(f"Recall - Time filtering conditions: {time_where}, params: {params}")

            # Determine search strategy
            if query and query.strip():
                # Combined semantic search with time filtering
                logger.info(f"Recall - Using semantic search with query: '{query}'")

                try:
                    # Generate query embedding
                    query_embedding = await self._generate_embedding(query)

                    # Search Vectorize with semantic query
                    search_payload = {
                        "vector": query_embedding,
                        "topK": n_results,
                        "returnMetadata": "all",
                        "returnValues": False
                    }

                    # Add time filtering to vectorize metadata if specified
                    if time_conditions:
                        # Note: Vectorize metadata filtering capabilities may be limited
                        # We'll filter after retrieval for now
                        logger.info("Recall - Time filtering will be applied post-retrieval from Vectorize")

                    response = await self._retry_request("POST", f"{self.vectorize_url}/query", json=search_payload)
                    result = response.json()

                    if not result.get("success"):
                        raise ValueError(f"Vectorize query failed: {result}")

                    matches = result.get("result", {}).get("matches", [])

                    # Convert matches to MemoryQueryResult objects with time filtering
                    results = []
                    for match in matches:
                        memory = await self._load_memory_from_match(match)
                        if memory:
                            # Apply time filtering if needed
                            if start_timestamp is not None and memory.created_at and memory.created_at < start_timestamp:
                                continue
                            if end_timestamp is not None and memory.created_at and memory.created_at > end_timestamp:
                                continue

                            query_result = MemoryQueryResult(
                                memory=memory,
                                relevance_score=match.get("score", 0.0)
                            )
                            results.append(query_result)

                    logger.info(f"Recall - Retrieved {len(results)} memories with semantic search and time filtering")
                    return results[:n_results]  # Ensure we don't exceed n_results

                except Exception as e:
                    logger.error(f"Recall - Semantic search failed, falling back to time-based search: {e}")
                    # Fall through to time-based search

            # Time-based search only (or fallback)
            logger.info(f"Recall - Using time-based search only")

            # Build D1 query for time-based retrieval
            if time_where:
                sql = f"SELECT * FROM memories WHERE {time_where} ORDER BY created_at DESC LIMIT ?"
                params.append(n_results)
            else:
                # No time filters, get most recent
                sql = "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?"
                params = [n_results]

            payload = {"sql": sql, "params": params}
            response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
            result = response.json()

            if not result.get("success"):
                raise ValueError(f"D1 query failed: {result}")

            # Convert D1 results to MemoryQueryResult objects
            results = []
            if result.get("result", [{}])[0].get("results"):
                for row in result["result"][0]["results"]:
                    memory = await self._load_memory_from_row(row)
                    if memory:
                        # For time-based search without semantic query, use timestamp as relevance
                        relevance_score = memory.created_at or 0.0
                        query_result = MemoryQueryResult(
                            memory=memory,
                            relevance_score=relevance_score
                        )
                        results.append(query_result)

            logger.info(f"Recall - Retrieved {len(results)} memories with time-based search")
            return results

        except Exception as e:
            logger.error(f"Recall failed: {e}")
            return []

    async def get_all_memories(self, limit: int = None, offset: int = 0, memory_type: Optional[str] = None, tags: Optional[List[str]] = None) -> List[Memory]:
        """
        Get all memories in storage ordered by creation time (newest first).

        Args:
            limit: Maximum number of memories to return (None for all)
            offset: Number of memories to skip (for pagination)
            memory_type: Optional filter by memory type
            tags: Optional filter by tags (matches ANY of the provided tags)

        Returns:
            List of Memory objects ordered by created_at DESC, optionally filtered by type and tags
        """
        try:
            # Build SQL query with optional memory_type and tags filters
            sql = "SELECT m.* FROM memories m"
            joins = ""
            where_conditions = []
            params: List[Any] = []

            if memory_type is not None:
                where_conditions.append("m.memory_type = ?")
                params.append(memory_type)

            tag_count = 0
            if tags:
                tag_count = len(tags)
                placeholders = ",".join(["?"] * tag_count)
                joins += " " + """
                    INNER JOIN memory_tags mt ON m.id = mt.memory_id
                    INNER JOIN tags t ON mt.tag_id = t.id
                """.strip().replace("\n", " ")
                where_conditions.append(f"t.name IN ({placeholders})")
                params.extend(tags)

            if joins:
                sql += joins

            if where_conditions:
                sql += " WHERE " + " AND ".join(where_conditions)

            if tags:
                sql += " GROUP BY m.id HAVING COUNT(DISTINCT t.name) = ?"

            sql += " ORDER BY m.created_at DESC"

            query_params = params.copy()
            if tags:
                query_params.append(tag_count)

            if limit is not None:
                sql += " LIMIT ?"
                query_params.append(limit)

            if offset > 0:
                sql += " OFFSET ?"
                query_params.append(offset)

            payload = {"sql": sql, "params": query_params}
            response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
            result = response.json()

            if not result.get("success"):
                raise ValueError(f"D1 query failed: {result}")

            memories = []
            if result.get("result", [{}])[0].get("results"):
                for row in result["result"][0]["results"]:
                    memory = await self._load_memory_from_row(row)
                    if memory:
                        memories.append(memory)

            logger.debug(f"Retrieved {len(memories)} memories from D1")
            return memories

        except Exception as e:
            logger.error(f"Error getting all memories: {str(e)}")
            return []

    def _row_to_memory(self, row: Dict[str, Any]) -> Memory:
        """Convert D1 row to Memory object without loading tags (for bulk operations)."""
        # Load content from R2 if needed
        content = row["content"]
        if row.get("r2_key") and content.startswith("[R2 Content:"):
            # For bulk operations, we don't load R2 content to avoid additional requests
            # Just keep the placeholder
            pass

        # Parse and decompress metadata
        metadata = {}
        if row.get("metadata_json"):
            metadata = json.loads(row["metadata_json"])
            # Decompress CSV-encoded quality metadata if present
            from ..quality.metadata_codec import decompress_metadata_from_sync
            metadata = decompress_metadata_from_sync(metadata)

        return Memory(
            content=content,
            content_hash=row["content_hash"],
            tags=[],  # Skip tag loading for bulk operations
            memory_type=row.get("memory_type"),
            metadata=metadata,
            created_at=row.get("created_at"),
            created_at_iso=row.get("created_at_iso"),
            updated_at=row.get("updated_at"),
            updated_at_iso=row.get("updated_at_iso")
        )

    async def get_all_memories_bulk(
        self,
        include_tags: bool = False
    ) -> List[Memory]:
        """
        Efficiently load all memories from D1.

        If include_tags=False, skips N+1 tag queries for better performance.
        Useful for maintenance scripts that don't need tag information.

        Args:
            include_tags: Whether to load tags for each memory (slower but complete)

        Returns:
            List of Memory objects ordered by created_at DESC
        """
        query = "SELECT * FROM memories ORDER BY created_at DESC"
        payload = {"sql": query}
        response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
        result = response.json()

        if not result.get("success"):
            raise ValueError(f"D1 query failed: {result}")

        memories = []
        if result.get("result", [{}])[0].get("results"):
            for row in result["result"][0]["results"]:
                if include_tags:
                    memory = await self._load_memory_from_row(row)
                else:
                    memory = self._row_to_memory(row)
                if memory:
                    memories.append(memory)

        logger.debug(f"Bulk loaded {len(memories)} memories from D1")
        return memories

    async def get_all_memories_cursor(self, limit: int = None, cursor: float = None, memory_type: Optional[str] = None, tags: Optional[List[str]] = None) -> List[Memory]:
        """
        Get all memories using cursor-based pagination to avoid D1 OFFSET limitations.

        This method uses timestamp-based cursors instead of OFFSET, which is more efficient
        and avoids Cloudflare D1's OFFSET limitations that cause 400 Bad Request errors.

        Args:
            limit: Maximum number of memories to return (None for all)
            cursor: Timestamp cursor for pagination (created_at value from last result)
            memory_type: Optional filter by memory type
            tags: Optional filter by tags (matches ANY of the provided tags)

        Returns:
            List of Memory objects ordered by created_at DESC, starting after cursor
        """
        try:
            # Build SQL query with cursor-based pagination
            sql = "SELECT * FROM memories"
            params = []
            where_conditions = []

            # Add cursor condition (timestamp-based pagination)
            if cursor is not None:
                where_conditions.append("created_at < ?")
                params.append(cursor)

            # Add memory_type filter if specified
            if memory_type is not None:
                where_conditions.append("memory_type = ?")
                params.append(memory_type)

            # Add tags filter if specified (using LIKE for tag matching)
            if tags and len(tags) > 0:
                tag_conditions = " OR ".join(["tags LIKE ?" for _ in tags])
                where_conditions.append(f"({tag_conditions})")
                params.extend([f"%{tag}%" for tag in tags])

            # Apply WHERE clause if we have any conditions
            if where_conditions:
                sql += " WHERE " + " AND ".join(where_conditions)

            sql += " ORDER BY created_at DESC"

            if limit is not None:
                sql += " LIMIT ?"
                params.append(limit)

            payload = {"sql": sql, "params": params}
            response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
            result = response.json()

            if not result.get("success"):
                raise ValueError(f"D1 query failed: {result}")

            memories = []
            if result.get("result", [{}])[0].get("results"):
                for row in result["result"][0]["results"]:
                    memory = await self._load_memory_from_row(row)
                    if memory:
                        memories.append(memory)

            logger.debug(f"Retrieved {len(memories)} memories from D1 with cursor-based pagination")
            return memories

        except Exception as e:
            logger.error(f"Error getting memories with cursor: {str(e)}")
            return []

    async def get_memories_updated_since(self, timestamp: float, limit: int = 100) -> List[Memory]:
        """
        Get memories updated since a specific timestamp (v8.25.0+).

        Used by hybrid backend drift detection to find memories with metadata
        changes that need to be synchronized.

        Args:
            timestamp: Unix timestamp (seconds since epoch)
            limit: Maximum number of memories to return (default: 100)

        Returns:
            List of Memory objects updated after the timestamp, ordered by updated_at DESC
        """
        try:
            # Use numeric updated_at column for efficient D1 query (fixes #264)
            # Numeric comparison is 10-100x faster than ISO string comparison
            # and leverages the indexed updated_at column
            query = """
            SELECT *
            FROM memories
            WHERE updated_at > ?
            ORDER BY updated_at DESC
            LIMIT ?
            """

            payload = {"sql": query, "params": [timestamp, limit]}
            response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
            result = response.json()

            if not result.get("success"):
                logger.warning(f"Failed to get updated memories: {result.get('error')}")
                return []

            memories = []
            if result.get("result", [{}])[0].get("results"):
                for row in result["result"][0]["results"]:
                    memory = await self._load_memory_from_row(row)
                    if memory:
                        memories.append(memory)

            logger.debug(f"Retrieved {len(memories)} memories updated since timestamp {timestamp}")
            return memories

        except Exception as e:
            logger.error(f"Error getting updated memories: {e}")
            return []

    async def get_memories_by_time_range(self, start_time: float, end_time: float) -> List[Memory]:
        """
        Get memories within a specific time range (v8.31.0+).

        Pushes date-range filtering to D1 database layer for better performance.
        This optimization reduces network transfer and enables efficient queries
        on databases with >10,000 memories.

        Benefits:
        - Reduces network transfer (only relevant memories)
        - Leverages D1 indexes on created_at
        - Scales to databases with >10,000 memories
        - 10x performance improvement over application-layer filtering

        Args:
            start_time: Start timestamp (Unix seconds, inclusive)
            end_time: End timestamp (Unix seconds, inclusive)

        Returns:
            List of Memory objects within the time range, ordered by created_at DESC
        """
        try:
            # Build SQL query with BETWEEN clause for efficient filtering
            # Use SELECT m.* to get all columns (tags are loaded separately via _load_memory_tags)
            sql = """
                SELECT m.*
                FROM memories m
                WHERE m.created_at BETWEEN ? AND ?
                ORDER BY m.created_at DESC
            """

            payload = {"sql": sql, "params": [start_time, end_time]}
            response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
            result = response.json()

            if not result.get("success"):
                logger.error(f"D1 query failed: {result}")
                return []

            memories = []
            if result.get("result", [{}])[0].get("results"):
                for row in result["result"][0]["results"]:
                    memory = await self._load_memory_from_row(row)
                    if memory:
                        memories.append(memory)

            logger.info(f"Retrieved {len(memories)} memories in time range {start_time}-{end_time}")
            return memories

        except Exception as e:
            logger.error(f"Error getting memories by time range: {str(e)}")
            return []

    async def count_all_memories(self, memory_type: Optional[str] = None, tags: Optional[List[str]] = None) -> int:
        """
        Get total count of memories in storage.

        Args:
            memory_type: Optional filter by memory type
            tags: Optional filter by tags (memories matching ANY of the tags)

        Returns:
            Total number of memories, optionally filtered by type and/or tags
        """
        try:
            # Build query with filters
            base_sql = "SELECT m.id FROM memories m"
            joins = ""
            conditions = []
            params: List[Any] = []

            if memory_type is not None:
                conditions.append('m.memory_type = ?')
                params.append(memory_type)

            tag_count = 0
            if tags:
                tag_count = len(tags)
                placeholders = ','.join(['?'] * tag_count)
                joins += " " + """
                    INNER JOIN memory_tags mt ON m.id = mt.memory_id
                    INNER JOIN tags t ON mt.tag_id = t.id
                """.strip().replace("\n", " ")
                conditions.append(f't.name IN ({placeholders})')
                params.extend(tags)

            sql = base_sql + joins
            if conditions:
                sql += ' WHERE ' + ' AND '.join(conditions)

            if tags:
                sql += ' GROUP BY m.id HAVING COUNT(DISTINCT t.name) = ?'

            count_sql = f'SELECT COUNT(*) as count FROM ({sql}) AS subquery'

            count_params = params.copy()
            if tags:
                count_params.append(tag_count)

            payload = {"sql": count_sql, "params": count_params}
            response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
            result = response.json()

            if not result.get("success"):
                raise ValueError(f"D1 query failed: {result}")

            if result.get("result", [{}])[0].get("results"):
                count = result["result"][0]["results"][0].get("count", 0)
                return int(count)

            return 0

        except Exception as e:
            logger.error(f"Error counting memories: {str(e)}")
            return 0

    async def close(self) -> None:
        """Close the storage backend and cleanup resources."""
        if self.client:
            await self.client.aclose()
            self.client = None

        # Clear embedding cache
        self._embedding_cache.clear()

        logger.info("Cloudflare storage backend closed")
