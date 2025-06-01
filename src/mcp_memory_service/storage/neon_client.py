"""
Neon PostgreSQL Client for EchoVault Memory Service
Copyright (c) 2025 EchoVault
Licensed under the MIT License.

This module provides an asyncpg-based client for connecting to Neon PostgreSQL,
with support for pgvector operations and connection pooling.
"""

import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union

import asyncpg
from asyncpg.pool import Pool

logger = logging.getLogger(__name__)

class NeonClient:
    """
    Client for Neon PostgreSQL with pgvector support.
    Provides async operations for storing and retrieving vector embeddings.
    """
    
    def __init__(self):
        """Initialize the Neon client."""
        self.pool = None
        self.dsn = os.environ.get("NEON_DSN")
        self.pool_size = int(os.environ.get("NEON_POOL_SIZE", "5"))
        self._is_initialized = False
    
    async def initialize(self):
        """Initialize the connection pool."""
        if self._is_initialized:
            return
            
        if not self.dsn:
            logger.error("NEON_DSN environment variable is not set")
            raise ValueError("NEON_DSN environment variable is not set")
            
        try:
            # Create connection pool
            logger.info(f"Creating connection pool with {self.pool_size} connections")
            self.pool = await asyncpg.create_pool(
                dsn=self.dsn,
                min_size=1,
                max_size=self.pool_size,
                command_timeout=30,
                setup=self._setup_connection
            )
            
            # Initialize database schema if needed
            await self._init_schema()
            
            self._is_initialized = True
            logger.info("Neon client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Neon client: {e}")
            raise
    
    async def _setup_connection(self, connection: asyncpg.Connection):
        """Set up connection with vector extension."""
        # Enable pgvector extension
        try:
            await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except asyncpg.exceptions.UndefinedObjectError:
            logger.error("pgvector extension is not available on this server")
            raise ValueError("pgvector extension is not available on this server")
            
        # Set up custom data types
        await connection.set_type_codec(
            'jsonb',
            encoder=json.dumps,
            decoder=json.loads,
            schema='pg_catalog'
        )
    
    async def _init_schema(self):
        """Initialize database schema."""
        async with self.pool.acquire() as conn:
            # Create memories table with vector support
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding VECTOR(1536),
                    metadata JSONB,
                    tags JSONB,
                    timestamp BIGINT,
                    content_hash TEXT UNIQUE,
                    memory_type TEXT,
                    payload_url TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index on content_hash for faster lookups
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_content_hash 
                ON memories(content_hash)
            """)
            
            # Create vector index for fast similarity search
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_embedding 
                ON memories USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100)
            """)
            
            # Create timestamp index for time-based filtering
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_timestamp 
                ON memories(timestamp)
            """)
            
            # Create index on tags for tag-based search
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_tags 
                ON memories USING GIN (tags)
            """)
            
            # Create telemetry table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS telemetry (
                    id SERIAL PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    event_data JSONB,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    async def close(self):
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            self._is_initialized = False
    
    async def insert_event(self, 
                          content: str, 
                          content_hash: str,
                          embedding: List[float],
                          tags: List[str],
                          memory_type: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None,
                          timestamp: Optional[int] = None,
                          payload_url: Optional[str] = None) -> bool:
        """
        Insert a memory event into the database.
        
        Args:
            content: Memory content
            content_hash: Hash of the content
            embedding: Vector embedding of the content
            tags: List of tags
            memory_type: Type of memory
            metadata: Additional metadata
            timestamp: Unix timestamp
            payload_url: URL to blob storage if content is large
            
        Returns:
            True if insertion was successful
        """
        if not self._is_initialized:
            await self.initialize()
        
        async with self.pool.acquire() as conn:
            try:
                # Insert into memories table
                await conn.execute("""
                    INSERT INTO memories(
                        id, content, embedding, tags, timestamp, 
                        content_hash, memory_type, metadata, payload_url
                    )
                    VALUES($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT(content_hash) DO NOTHING
                """, 
                content_hash,  # Using content_hash as ID
                content,
                embedding,
                json.dumps(tags) if isinstance(tags, list) else tags,
                timestamp,
                content_hash,
                memory_type or "",
                metadata or {},
                payload_url
                )
                
                # Record telemetry
                await conn.execute("""
                    INSERT INTO telemetry(event_type, event_data)
                    VALUES($1, $2)
                """,
                "memory_insert",
                {
                    "content_hash": content_hash,
                    "memory_type": memory_type,
                    "tags_count": len(tags) if isinstance(tags, list) else 0,
                    "has_payload_url": payload_url is not None,
                    "content_length": len(content)
                })
                
                return True
            except Exception as e:
                logger.error(f"Failed to insert memory event: {e}")
                raise
    
    async def search_by_vector(self, 
                             embedding: List[float], 
                             limit: int = 5,
                             similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search memories by vector similarity.
        
        Args:
            embedding: Vector embedding to search for
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of memories sorted by similarity
        """
        if not self._is_initialized:
            await self.initialize()
        
        async with self.pool.acquire() as conn:
            try:
                rows = await conn.fetch("""
                    SELECT 
                        id, content, content_hash, memory_type, tags, metadata,
                        timestamp, payload_url,
                        1 - (embedding <=> $1) as similarity
                    FROM 
                        memories
                    WHERE 
                        embedding IS NOT NULL AND
                        1 - (embedding <=> $1) >= $3
                    ORDER BY 
                        similarity DESC
                    LIMIT $2
                """, embedding, limit, similarity_threshold)
                
                # Record telemetry
                await conn.execute("""
                    INSERT INTO telemetry(event_type, event_data)
                    VALUES($1, $2)
                """,
                "vector_search",
                {
                    "limit": limit,
                    "similarity_threshold": similarity_threshold,
                    "results_count": len(rows)
                })
                
                # Convert rows to dictionaries
                results = []
                for row in rows:
                    # Parse tags from JSON if needed
                    try:
                        tags = json.loads(row["tags"]) if isinstance(row["tags"], str) else row["tags"]
                    except (json.JSONDecodeError, TypeError):
                        tags = []
                    
                    results.append({
                        "id": row["id"],
                        "content": row["content"],
                        "content_hash": row["content_hash"],
                        "memory_type": row["memory_type"],
                        "tags": tags,
                        "metadata": row["metadata"],
                        "timestamp": row["timestamp"],
                        "payload_url": row["payload_url"],
                        "similarity": row["similarity"]
                    })
                
                return results
            except Exception as e:
                logger.error(f"Failed to search memories by vector: {e}")
                raise
    
    async def search_by_tags(self, tags: List[str], limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search memories by tags.
        
        Args:
            tags: List of tags to search for
            limit: Maximum number of results to return
            
        Returns:
            List of memories that have any of the specified tags
        """
        if not self._is_initialized:
            await self.initialize()
        
        async with self.pool.acquire() as conn:
            try:
                # Use ANY operator for tag matching
                tags_json = [json.dumps(tag) for tag in tags]
                
                rows = await conn.fetch("""
                    SELECT 
                        id, content, content_hash, memory_type, tags, metadata,
                        timestamp, payload_url
                    FROM 
                        memories
                    WHERE 
                        tags ?| $1::text[]
                    LIMIT $2
                """, tags_json, limit)
                
                # Record telemetry
                await conn.execute("""
                    INSERT INTO telemetry(event_type, event_data)
                    VALUES($1, $2)
                """,
                "tag_search",
                {
                    "tags": tags,
                    "limit": limit,
                    "results_count": len(rows)
                })
                
                # Convert rows to dictionaries
                results = []
                for row in rows:
                    # Parse tags from JSON if needed
                    try:
                        row_tags = json.loads(row["tags"]) if isinstance(row["tags"], str) else row["tags"]
                    except (json.JSONDecodeError, TypeError):
                        row_tags = []
                    
                    results.append({
                        "id": row["id"],
                        "content": row["content"],
                        "content_hash": row["content_hash"],
                        "memory_type": row["memory_type"],
                        "tags": row_tags,
                        "metadata": row["metadata"],
                        "timestamp": row["timestamp"],
                        "payload_url": row["payload_url"]
                    })
                
                return results
            except Exception as e:
                logger.error(f"Failed to search memories by tags: {e}")
                raise
    
    async def delete_memory(self, content_hash: str) -> bool:
        """
        Delete a memory by content hash.
        
        Args:
            content_hash: Hash of the memory to delete
            
        Returns:
            True if the memory was deleted
        """
        if not self._is_initialized:
            await self.initialize()
        
        async with self.pool.acquire() as conn:
            try:
                # Start a transaction
                async with conn.transaction():
                    # Get the memory payload_url before deleting
                    row = await conn.fetchrow("""
                        SELECT payload_url FROM memories WHERE content_hash = $1
                    """, content_hash)
                    
                    # Delete the memory
                    result = await conn.execute("""
                        DELETE FROM memories WHERE content_hash = $1
                    """, content_hash)
                    
                    # Record telemetry
                    await conn.execute("""
                        INSERT INTO telemetry(event_type, event_data)
                        VALUES($1, $2)
                    """,
                    "memory_delete",
                    {
                        "content_hash": content_hash,
                        "had_payload_url": row and row["payload_url"] is not None
                    })
                    
                    # Return True if a row was deleted
                    return result.split()[1] == "1"
            except Exception as e:
                logger.error(f"Failed to delete memory: {e}")
                raise
    
    async def delete_by_tag(self, tag: str) -> int:
        """
        Delete memories by tag.
        
        Args:
            tag: Tag to delete memories for
            
        Returns:
            Number of memories deleted
        """
        if not self._is_initialized:
            await self.initialize()
        
        async with self.pool.acquire() as conn:
            try:
                # Convert tag to JSON string for comparison
                tag_json = json.dumps(tag)
                
                # Get content_hashes of memories to delete
                rows = await conn.fetch("""
                    SELECT content_hash, payload_url 
                    FROM memories 
                    WHERE tags @> $1
                """, [tag_json])
                
                # Store content hashes and payload URLs
                content_hashes = [row["content_hash"] for row in rows]
                payload_urls = [row["payload_url"] for row in rows if row["payload_url"]]
                
                if not content_hashes:
                    return 0
                    
                # Delete memories
                result = await conn.execute("""
                    DELETE FROM memories WHERE content_hash = ANY($1::text[])
                """, content_hashes)
                
                # Record telemetry
                await conn.execute("""
                    INSERT INTO telemetry(event_type, event_data)
                    VALUES($1, $2)
                """,
                "tag_delete",
                {
                    "tag": tag,
                    "memories_count": len(content_hashes),
                    "payload_urls_count": len(payload_urls)
                })
                
                # Return number of deleted memories
                return len(content_hashes)
            except Exception as e:
                logger.error(f"Failed to delete memories by tag: {e}")
                raise
    
    async def search_by_timeframe(self, 
                                start_timestamp: int,
                                end_timestamp: int,
                                limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search memories by timeframe.
        
        Args:
            start_timestamp: Start timestamp (inclusive)
            end_timestamp: End timestamp (inclusive)
            limit: Maximum number of results to return
            
        Returns:
            List of memories within the timeframe
        """
        if not self._is_initialized:
            await self.initialize()
        
        async with self.pool.acquire() as conn:
            try:
                rows = await conn.fetch("""
                    SELECT 
                        id, content, content_hash, memory_type, tags, metadata,
                        timestamp, payload_url
                    FROM 
                        memories
                    WHERE 
                        timestamp >= $1 AND timestamp <= $2
                    ORDER BY
                        timestamp DESC
                    LIMIT $3
                """, start_timestamp, end_timestamp, limit)
                
                # Record telemetry
                await conn.execute("""
                    INSERT INTO telemetry(event_type, event_data)
                    VALUES($1, $2)
                """,
                "timeframe_search",
                {
                    "start_timestamp": start_timestamp,
                    "end_timestamp": end_timestamp,
                    "limit": limit,
                    "results_count": len(rows)
                })
                
                # Convert rows to dictionaries
                results = []
                for row in rows:
                    # Parse tags from JSON if needed
                    try:
                        tags = json.loads(row["tags"]) if isinstance(row["tags"], str) else row["tags"]
                    except (json.JSONDecodeError, TypeError):
                        tags = []
                    
                    results.append({
                        "id": row["id"],
                        "content": row["content"],
                        "content_hash": row["content_hash"],
                        "memory_type": row["memory_type"],
                        "tags": tags,
                        "metadata": row["metadata"],
                        "timestamp": row["timestamp"],
                        "payload_url": row["payload_url"]
                    })
                
                return results
            except Exception as e:
                logger.error(f"Failed to search memories by timeframe: {e}")
                raise
    
    async def exact_match_retrieve(self, content: str) -> List[Dict[str, Any]]:
        """
        Retrieve memories with exact matching content.
        
        Args:
            content: Content to match exactly
            
        Returns:
            List of memories with matching content
        """
        if not self._is_initialized:
            await self.initialize()
        
        async with self.pool.acquire() as conn:
            try:
                rows = await conn.fetch("""
                    SELECT 
                        id, content, content_hash, memory_type, tags, metadata,
                        timestamp, payload_url
                    FROM 
                        memories
                    WHERE 
                        content = $1
                """, content)
                
                # Record telemetry
                await conn.execute("""
                    INSERT INTO telemetry(event_type, event_data)
                    VALUES($1, $2)
                """,
                "exact_match_search",
                {
                    "content_length": len(content),
                    "results_count": len(rows)
                })
                
                # Convert rows to dictionaries
                results = []
                for row in rows:
                    # Parse tags from JSON if needed
                    try:
                        tags = json.loads(row["tags"]) if isinstance(row["tags"], str) else row["tags"]
                    except (json.JSONDecodeError, TypeError):
                        tags = []
                    
                    results.append({
                        "id": row["id"],
                        "content": row["content"],
                        "content_hash": row["content_hash"],
                        "memory_type": row["memory_type"],
                        "tags": tags,
                        "metadata": row["metadata"],
                        "timestamp": row["timestamp"],
                        "payload_url": row["payload_url"]
                    })
                
                return results
            except Exception as e:
                logger.error(f"Failed to retrieve memories by exact match: {e}")
                raise
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        if not self._is_initialized:
            await self.initialize()
        
        async with self.pool.acquire() as conn:
            try:
                # Get memory count
                memory_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM memories
                """)
                
                # Get blob count
                blob_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM memories WHERE payload_url IS NOT NULL
                """)
                
                # Get average vector size
                avg_vector_size = await conn.fetchval("""
                    SELECT AVG(vector_dims(embedding::vector)) FROM memories 
                    WHERE embedding IS NOT NULL
                """)
                
                # Get oldest and newest memory timestamp
                oldest = await conn.fetchval("""
                    SELECT MIN(timestamp) FROM memories
                """)
                
                newest = await conn.fetchval("""
                    SELECT MAX(timestamp) FROM memories
                """)
                
                return {
                    "memory_count": memory_count,
                    "blob_count": blob_count,
                    "avg_vector_size": avg_vector_size,
                    "oldest_timestamp": oldest,
                    "newest_timestamp": newest
                }
            except Exception as e:
                logger.error(f"Failed to get memory statistics: {e}")
                raise