"""
EchoVault Storage Implementation for MCP Memory Service
Copyright (c) 2025 EchoVault
Licensed under the MIT License.

This module provides an EchoVault implementation of the MemoryStorage interface
that uses Neon PostgreSQL, Qdrant, and Cloudflare R2 for enhanced performance and durability.
"""

import os
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import asyncio

from .base import MemoryStorage
from .neon_client import NeonClient
from .vector_store import VectorStoreClient
from .blob_store import BlobStoreClient
from ..models.memory import Memory, MemoryQueryResult
from ..utils.hashing import generate_content_hash # generate_content_hash is not used in this file, but keeping it just in case.
from ..utils import otel_prom # Ensure otel_prom is correctly imported

logger = logging.getLogger(__name__) # Define logger at module level

class EchoVaultStorage(MemoryStorage):
    """
    EchoVault storage implementation that uses Neon, Qdrant, and R2.
    Implements the MemoryStorage interface for drop-in replacement.
    """
    
    def __init__(self, path: Optional[str] = None):
        logger.debug(f"EchoVaultStorage __init__ called with path: {path}")
        self.path = path 

        self.neon_client = NeonClient()
        self.vector_store = VectorStoreClient()
        self.blob_store = BlobStoreClient()
        self.model = None 
        self._is_initialized = False
        
        if hasattr(otel_prom, 'initialize') and callable(otel_prom.initialize):
             if not getattr(otel_prom, '_otel_initialized_echovault', False):
                logger.info("Initializing OpenTelemetry for echovault-memory-service via EchoVaultStorage.")
                otel_prom.initialize("echovault-memory-service")
                setattr(otel_prom, '_otel_initialized_echovault', True)
             else:
                logger.debug("OpenTelemetry for echovault-memory-service already initialized.")
        else:
            logger.warning("otel_prom.initialize not available or not callable.")

        logger.info("EchoVaultStorage instance created.")

    @property
    def collection(self):
        """ChromaDB compatibility attribute. Returns None."""
        logger.debug("Access to compatibility property EchoVaultStorage.collection (returns None)")
        return None

    @property
    def client(self):
        """ChromaDB compatibility attribute. Returns None."""
        logger.debug("Access to compatibility property EchoVaultStorage.client (returns None)")
        return None

    async def initialize(self):
        """Initialize all clients and connections."""
        if self._is_initialized:
            return
        
        try:
            # Initialize clients
            await self.neon_client.initialize()
            await self.vector_store.initialize()
            await self.blob_store.initialize()
            
            # Get embedding model from vector store if available
            if hasattr(self.vector_store, "model") and self.vector_store.model:
                self.model = self.vector_store.model
            elif hasattr(self.neon_client, "model") and self.neon_client.model:
                self.model = self.neon_client.model
            
            self._is_initialized = True
            logger.info("EchoVault storage initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EchoVault storage: {e}")
            raise
    
    @otel_prom.trace_async("store_memory")
    async def store(self, memory: Memory) -> Tuple[bool, str]:
        """
        Store a memory.
        
        Args:
            memory: Memory to store
            
        Returns:
            Tuple of (success, message)
        """
        if not self._is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Generate embedding if not provided
            if not memory.embedding and self.model:
                memory.embedding = self.model.encode(memory.content).tolist()
            
            # Check if content should be stored in blob storage
            content_length = len(memory.content.encode('utf-8'))
            original_content = memory.content
            payload_url = None
            
            if self.blob_store.is_configured() and content_length > self.blob_store.blob_threshold:
                # Store content in blob storage
                memory.content, payload_url = await self.blob_store.save_if_large(memory.content, memory.content_hash)
            
            # Store in vector store
            await self.vector_store.upsert(
                id=memory.content_hash,
                content=original_content,  # Always use original content for embedding
                embedding=memory.embedding,
                metadata={
                    "content_hash": memory.content_hash,
                    "memory_type": memory.memory_type if memory.memory_type else "",
                    "tags": memory.tags,
                    "timestamp": int(memory.timestamp.timestamp()) if hasattr(memory.timestamp, "timestamp") else int(time.time()),
                    "payload_url": payload_url,
                    **memory.metadata
                }
            )
            
            # Record telemetry
            duration_ms = (time.time() - start_time) * 1000
            otel_prom.trace_write(
                content_length=content_length,
                has_payload_url=payload_url is not None,
                tags_count=len(memory.tags)
            )
            
            return True, f"Successfully stored memory {memory.content_hash}"
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return False, f"Failed to store memory: {e}"
    
    @otel_prom.trace_async("retrieve_memory")
    async def retrieve(self, query: str, n_results: int = 5) -> List[MemoryQueryResult]:
        """
        Retrieve memories by semantic search.
        
        Args:
            query: Query string for semantic search
            n_results: Maximum number of results to return
            
        Returns:
            List of memory query results
        """
        if not self._is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Generate query embedding
            if self.model:
                query_embedding = self.model.encode(query).tolist()
            else:
                # Fallback - try to use vector store's embedding method
                if hasattr(self.vector_store, "_generate_embedding"):
                    query_embedding = await self.vector_store._generate_embedding(query)
                else:
                    logger.error("No embedding model available")
                    return []
            
            # Search in vector store
            results = await self.vector_store.search(
                embedding=query_embedding,
                limit=n_results,
                similarity_threshold=0.0  # Return all results, we'll filter later
            )
            
            # Convert to MemoryQueryResult
            memory_results = []
            for result in results:
                # Reconstruct content if needed
                content = result["content"]
                payload_url = result.get("metadata", {}).get("payload_url")
                
                if payload_url and self.blob_store.is_configured():
                    # Retrieve full content from blob storage
                    full_content = await self.blob_store.retrieve_content(payload_url)
                    if full_content:
                        content = full_content
                
                # Extract metadata and tags
                metadata = result.get("metadata", {})
                
                if "tags" in metadata:
                    tags = metadata.pop("tags")
                    if isinstance(tags, str):
                        try:
                            tags = json.loads(tags)
                        except json.JSONDecodeError:
                            tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
                else:
                    tags = []
                
                if "memory_type" in metadata:
                    memory_type = metadata.pop("memory_type")
                else:
                    memory_type = ""
                
                if "content_hash" in metadata:
                    content_hash = metadata.pop("content_hash")
                else:
                    content_hash = result.get("id", "")
                
                # Create memory object
                memory = Memory(
                    content=content,
                    content_hash=content_hash,
                    tags=tags,
                    memory_type=memory_type,
                    metadata=metadata
                )
                
                # Add to results
                memory_results.append(MemoryQueryResult(
                    memory=memory,
                    relevance_score=result.get("similarity", 0.0)
                ))
            
            # Record telemetry
            duration_ms = (time.time() - start_time) * 1000
            otel_prom.trace_read(
                latency_ms=duration_ms,
                results_count=len(memory_results)
            )
            
            return memory_results
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []
    
    @otel_prom.trace_async("search_by_tag")
    async def search_by_tag(self, tags: List[str]) -> List[Memory]:
        """
        Search memories by tags.
        
        Args:
            tags: List of tags to search for
            
        Returns:
            List of matching memories
        """
        if not self._is_initialized:
            await self.initialize()
        
        try:
            # Search by tag in Neon
            results = await self.neon_client.search_by_tags(tags)
            
            # Convert to Memory objects
            memories = []
            for result in results:
                # Check if content should be retrieved from blob storage
                content = result["content"]
                payload_url = result.get("payload_url")
                
                if payload_url and self.blob_store.is_configured():
                    # Retrieve full content from blob storage
                    full_content = await self.blob_store.retrieve_content(payload_url)
                    if full_content:
                        content = full_content
                
                # Create Memory object
                memory = Memory(
                    content=content,
                    content_hash=result["content_hash"],
                    tags=result["tags"],
                    memory_type=result.get("memory_type", ""),
                    metadata={k: v for k, v in result.items() if k not in ["content", "content_hash", "tags", "memory_type", "payload_url"]}
                )
                
                memories.append(memory)
            
            return memories
        except Exception as e:
            logger.error(f"Failed to search memories by tag: {e}")
            return []
    
    @otel_prom.trace_async("delete_memory")
    async def delete(self, content_hash: str) -> Tuple[bool, str]:
        """
        Delete a memory by its hash.
        
        Args:
            content_hash: Hash of the memory to delete
            
        Returns:
            Tuple of (success, message)
        """
        if not self._is_initialized:
            await self.initialize()
        
        try:
            # Get the memory payload_url before deleting
            async with self.neon_client.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT payload_url FROM memories WHERE content_hash = $1
                """, content_hash)
                
                payload_url = row["payload_url"] if row else None
            
            # Delete from vector store
            success = await self.vector_store.delete(content_hash)
            
            # Delete from blob storage if needed
            if payload_url and self.blob_store.is_configured():
                await self.blob_store.delete_blob(payload_url)
            
            if success:
                return True, f"Successfully deleted memory {content_hash}"
            else:
                return False, f"Memory {content_hash} not found"
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            return False, f"Failed to delete memory: {e}"
    
    @otel_prom.trace_async("delete_by_tag")
    async def delete_by_tag(self, tag: str) -> Tuple[int, str]:
        """
        Delete memories by tag.
        
        Args:
            tag: Tag to delete by
            
        Returns:
            Tuple of (count_deleted, message)
        """
        if not self._is_initialized:
            await self.initialize()
        
        try:
            # Delete from vector store (which handles both Neon and Qdrant)
            count = await self.vector_store.delete_by_tag(tag)
            
            if count > 0:
                return count, f"Successfully deleted {count} memories with tag '{tag}'"
            else:
                return 0, f"No memories found with tag '{tag}'"
        except Exception as e:
            logger.error(f"Failed to delete memories by tag: {e}")
            return 0, f"Failed to delete memories by tag: {e}"
    
    @otel_prom.trace_async("cleanup_duplicates")
    async def cleanup_duplicates(self) -> Tuple[int, str]:
        """
        Remove duplicate memories.
        
        Returns:
            Tuple of (count_removed, message)
        """
        if not self._is_initialized:
            await self.initialize()
        
        try:
            # Get all memories
            async with self.neon_client.pool.acquire() as conn:
                # Find duplicates by content
                rows = await conn.fetch("""
                    WITH duplicates AS (
                        SELECT 
                            content, 
                            array_agg(content_hash) AS content_hashes,
                            array_agg(payload_url) AS payload_urls
                        FROM 
                            memories
                        GROUP BY 
                            content
                        HAVING 
                            COUNT(*) > 1
                    )
                    SELECT content_hashes, payload_urls FROM duplicates
                """)
                
                if not rows:
                    return 0, "No duplicate memories found"
                
                # Process each set of duplicates
                total_deleted = 0
                for row in rows:
                    content_hashes = row["content_hashes"]
                    payload_urls = row["payload_urls"]
                    
                    # Keep the first hash, delete the rest
                    keep_hash = content_hashes[0]
                    delete_hashes = content_hashes[1:]
                    
                    # Delete duplicates from vector store
                    for hash_to_delete in delete_hashes:
                        await self.vector_store.delete(hash_to_delete)
                    
                    # Delete duplicate blobs if any
                    payload_urls = [url for url in payload_urls if url]
                    if len(payload_urls) > 1 and self.blob_store.is_configured():
                        # Keep the first URL, delete the rest
                        delete_urls = payload_urls[1:]
                        await self.blob_store.batch_delete_blobs(delete_urls)
                    
                    total_deleted += len(delete_hashes)
                
                return total_deleted, f"Successfully removed {total_deleted} duplicate memories"
        except Exception as e:
            logger.error(f"Failed to cleanup duplicates: {e}")
            return 0, f"Failed to cleanup duplicates: {e}"