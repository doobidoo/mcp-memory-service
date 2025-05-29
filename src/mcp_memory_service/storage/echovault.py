"""
EchoVault Storage Implementation for MCP Memory Service
Copyright (c) 2025 EchoVault
Licensed under the MIT License.

This module provides an implementation of the MemoryStorage interface
that uses Neon PostgreSQL and Qdrant for durable, high-performance
vector storage with large blob support via Cloudflare R2.
"""

import os
import sys
import json
import time
import logging
import asyncio
import hashlib
import traceback
from typing import List, Dict, Any, Tuple, Optional, Set, Union
from datetime import datetime, date

import asyncpg
import boto3
from botocore.client import Config
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer

from .base import MemoryStorage
from ..models.memory import Memory, MemoryQueryResult
from ..utils.hashing import generate_content_hash
from ..utils.system_detection import (
    get_system_info,
    get_optimal_embedding_settings,
    get_torch_device,
    print_system_diagnostics,
    AcceleratorType
)

logger = logging.getLogger(__name__)

# Configuration constants
COLLECTION_NAME = "memories"
VECTOR_SIZE = 384  # Default for all-MiniLM-L6-v2
BLOB_THRESHOLD = int(os.environ.get("BLOB_THRESHOLD", "32768"))  # 32KB default threshold for R2 storage

class EchoVaultStorage(MemoryStorage):
    """
    EchoVault implementation of MemoryStorage using Neon PostgreSQL and Qdrant.
    
    This implementation provides:
    - Durable storage in Neon PostgreSQL (with pgvector)
    - Fast vector search via Qdrant
    - Large blob storage in Cloudflare R2
    - Observability via OpenTelemetry and Prometheus
    """
    
    def __init__(self, path: str = None):
        """
        Initialize EchoVault storage.
        
        Args:
            path: Optional local path for fallback storage (not used in cloud mode)
        """
        self.path = path
        self.model = None
        self.embedding_function = None
        self.pg_pool = None
        self.qdrant_client = None
        self.r2_client = None
        self.system_info = get_system_info()
        self.embedding_settings = get_optimal_embedding_settings()
        
        # Log system information
        logger.info(f"Detected system: {self.system_info.os_name} {self.system_info.architecture}")
        logger.info(f"Accelerator: {self.system_info.accelerator}")
        logger.info(f"Memory: {self.system_info.memory_gb:.2f} GB")
        logger.info(f"Using device: {self.embedding_settings['device']}")
        
        # Initialize components
        self._initialize_embedding_model()
        self._initialize_storage()
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model with fallbacks for different hardware."""
        # Start with the optimal model for this system
        preferred_model = self.embedding_settings["model_name"]
        device = self.embedding_settings["device"]
        batch_size = self.embedding_settings["batch_size"]
        
        # Try the preferred model first, then fall back to alternatives
        models_to_try = [preferred_model] + [m for m in [
            'all-mpnet-base-v2',      # High quality, larger model
            'all-MiniLM-L6-v2',       # Good balance of quality and size
            'paraphrase-MiniLM-L6-v2', # Alternative with similar size
            'paraphrase-MiniLM-L3-v2', # Smaller model for constrained environments
            'paraphrase-albert-small-v2' # Smallest model, last resort
        ] if m != preferred_model]
        
        for model_name in models_to_try:
            try:
                logger.info(f"Attempting to load model: {model_name} on {device}")
                start_time = time.time()
                
                # Try to initialize the model with the current settings
                self.model = SentenceTransformer(
                    model_name,
                    device=device
                )
                
                # Set batch size based on available resources
                self.model.max_seq_length = 384  # Default max sequence length
                
                # Test the model with a simple encoding
                _ = self.model.encode("Test encoding", batch_size=batch_size)
                
                load_time = time.time() - start_time
                logger.info(f"Successfully loaded model {model_name} in {load_time:.2f}s")
                
                # Create embedding function for direct use
                self.embedding_function = lambda texts: self.model.encode(
                    texts, 
                    batch_size=batch_size,
                    show_progress_bar=False
                )
                
                logger.info(f"Embedding function initialized with model {model_name}")
                return
                
            except Exception as e:
                logger.warning(f"Failed to initialize model {model_name} on {device}: {str(e)}")
                
                # If we're not on CPU already, try falling back to CPU
                if device != "cpu":
                    try:
                        logger.info(f"Falling back to CPU for model: {model_name}")
                        self.model = SentenceTransformer(model_name, device="cpu")
                        _ = self.model.encode("Test encoding", batch_size=max(1, batch_size // 2))
                        
                        # Update settings to reflect CPU usage
                        self.embedding_settings["device"] = "cpu"
                        self.embedding_settings["batch_size"] = max(1, batch_size // 2)
                        
                        # Create embedding function
                        self.embedding_function = lambda texts: self.model.encode(
                            texts, 
                            batch_size=max(1, batch_size // 2),
                            show_progress_bar=False
                        )
                        
                        logger.info(f"Successfully loaded model {model_name} on CPU")
                        return
                    except Exception as cpu_e:
                        logger.warning(f"Failed to initialize model {model_name} on CPU: {str(cpu_e)}")
        
        # If we've tried all models and none worked, raise an exception
        error_msg = "Failed to initialize any embedding model. Service may not function correctly."
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    async def _initialize_storage(self):
        """Initialize PostgreSQL, Qdrant, and R2 connections."""
        try:
            # Initialize PostgreSQL connection pool
            dsn = os.environ.get("NEON_DSN")
            pool_size = int(os.environ.get("NEON_POOL_SIZE", "5"))
            
            if not dsn:
                raise ValueError("NEON_DSN environment variable is required")
            
            logger.info(f"Initializing PostgreSQL connection pool with size {pool_size}")
            self.pg_pool = await asyncpg.create_pool(
                dsn=dsn,
                min_size=2,
                max_size=pool_size
            )
            
            # Initialize database schema
            await self._initialize_database_schema()
            
            # Initialize Qdrant client if enabled
            if os.environ.get("USE_QDRANT", "").lower() in ("true", "1", "yes"):
                qdrant_url = os.environ.get("QDRANT_URL")
                qdrant_api_key = os.environ.get("QDRANT_API_KEY")
                
                if not qdrant_url or not qdrant_api_key:
                    raise ValueError("QDRANT_URL and QDRANT_API_KEY are required when USE_QDRANT=true")
                
                logger.info(f"Initializing Qdrant client at {qdrant_url}")
                self.qdrant_client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key
                )
                
                # Initialize Qdrant collection
                await self._initialize_qdrant_collection()
            
            # Initialize R2 client
            r2_endpoint = os.environ.get("R2_ENDPOINT")
            r2_access_key = os.environ.get("R2_ACCESS_KEY_ID")
            r2_secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
            r2_bucket = os.environ.get("R2_BUCKET")
            
            if r2_endpoint and r2_access_key and r2_secret_key and r2_bucket:
                logger.info(f"Initializing R2 client for bucket {r2_bucket}")
                self.r2_client = boto3.client(
                    's3',
                    endpoint_url=r2_endpoint,
                    aws_access_key_id=r2_access_key,
                    aws_secret_access_key=r2_secret_key,
                    config=Config(signature_version='s3v4')
                )
                
                # Ensure bucket exists
                try:
                    self.r2_client.head_bucket(Bucket=r2_bucket)
                    logger.info(f"R2 bucket {r2_bucket} exists")
                except Exception as e:
                    logger.warning(f"R2 bucket {r2_bucket} may not exist: {str(e)}")
            else:
                logger.warning("R2 configuration incomplete, large blob storage will be unavailable")
                self.r2_client = None
            
            logger.info("Storage initialization complete")
            
        except Exception as e:
            logger.error(f"Error initializing storage: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    async def _initialize_database_schema(self):
        """Initialize PostgreSQL database schema."""
        async with self.pg_pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create memories table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    memory_type TEXT,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                    embedding VECTOR(384),
                    metadata JSONB DEFAULT '{}'::JSONB,
                    is_blob BOOLEAN DEFAULT FALSE,
                    blob_key TEXT
                )
            """)
            
            # Create tags table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_tags (
                    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (memory_id, tag)
                )
            """)
            
            # Create index on content_hash
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_content_hash ON memories(content_hash)
            """)
            
            # Create index on memory_type
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_memory_type ON memories(memory_type)
            """)
            
            # Create index on created_at
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)
            """)
            
            # Create index on tags
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_tags_tag ON memory_tags(tag)
            """)
            
            # Create vector index
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
            
            logger.info("Database schema initialized")
    
    async def _initialize_qdrant_collection(self):
        """Initialize Qdrant collection."""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if COLLECTION_NAME not in collection_names:
                logger.info(f"Creating Qdrant collection {COLLECTION_NAME}")
                self.qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=qdrant_models.VectorParams(
                        size=VECTOR_SIZE,
                        distance=qdrant_models.Distance.COSINE
                    )
                )
            else:
                logger.info(f"Qdrant collection {COLLECTION_NAME} already exists")
        except Exception as e:
            logger.error(f"Error initializing Qdrant collection: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _format_tags(self, tags: Union[List[str], str, None]) -> List[str]:
        """Format tags to ensure they are a list of strings."""
        if tags is None:
            return []
        
        if isinstance(tags, str):
            # Handle comma-separated string
            return [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        if isinstance(tags, list):
            # Ensure all tags are strings
            return [str(tag).strip() for tag in tags if str(tag).strip()]
        
        # Default case
        return []
    
    async def _store_blob(self, content: str) -> str:
        """Store large content in R2 and return the blob key."""
        if not self.r2_client:
            raise ValueError("R2 client not initialized, cannot store large content")
        
        # Generate a unique key for the blob
        blob_key = f"blob_{hashlib.sha256(content.encode()).hexdigest()}"
        bucket = os.environ.get("R2_BUCKET")
        
        # Store the blob in R2
        self.r2_client.put_object(
            Bucket=bucket,
            Key=blob_key,
            Body=content.encode('utf-8'),
            ContentType='text/plain'
        )
        
        logger.info(f"Stored blob with key {blob_key} in bucket {bucket}")
        return blob_key
    
    async def _retrieve_blob(self, blob_key: str) -> str:
        """Retrieve content from R2 blob storage."""
        if not self.r2_client:
            raise ValueError("R2 client not initialized, cannot retrieve blob")
        
        bucket = os.environ.get("R2_BUCKET")
        
        # Retrieve the blob from R2
        response = self.r2_client.get_object(
            Bucket=bucket,
            Key=blob_key
        )
        
        # Read and decode the content
        content = response['Body'].read().decode('utf-8')
        return content
    
    async def store(self, memory: Memory) -> Tuple[bool, Optional[str]]:
        """Store a memory in PostgreSQL and Qdrant."""
        try:
            # Generate content hash if not provided
            if not memory.content_hash:
                memory.content_hash = generate_content_hash(memory.content, memory.metadata)
            
            # Format tags
            tags = self._format_tags(memory.tags)
            
            # Check if content exceeds blob threshold
            is_blob = len(memory.content) > BLOB_THRESHOLD
            blob_key = None
            
            if is_blob and self.r2_client:
                blob_key = await self._store_blob(memory.content)
                # Store a truncated version in the database
                stored_content = f"{memory.content[:1000]}... [TRUNCATED - Full content in blob {blob_key}]"
            else:
                stored_content = memory.content
            
            # Generate embedding
            embedding = self.embedding_function([memory.content])[0].tolist()
            
            async with self.pg_pool.acquire() as conn:
                # Check for duplicates
                existing = await conn.fetchval(
                    "SELECT id FROM memories WHERE content_hash = $1",
                    memory.content_hash
                )
                
                if existing:
                    return False, "Duplicate content detected"
                
                # Start transaction
                async with conn.transaction():
                    # Insert memory
                    await conn.execute(
                        """
                        INSERT INTO memories (
                            id, content, content_hash, memory_type, 
                            created_at, updated_at, embedding, metadata,
                            is_blob, blob_key
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        """,
                        memory.content_hash,
                        stored_content,
                        memory.content_hash,
                        memory.memory_type or "",
                        datetime.now(),
                        datetime.now(),
                        embedding,
                        json.dumps(memory.metadata or {}),
                        is_blob,
                        blob_key
                    )
                    
                    # Insert tags
                    if tags:
                        await conn.executemany(
                            "INSERT INTO memory_tags (memory_id, tag) VALUES ($1, $2)",
                            [(memory.content_hash, tag) for tag in tags]
                        )
            
            # If Qdrant is enabled, store the vector there as well
            if self.qdrant_client:
                self.qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[
                        qdrant_models.PointStruct(
                            id=memory.content_hash,
                            vector=embedding,
                            payload={
                                "content_hash": memory.content_hash,
                                "memory_type": memory.memory_type or "",
                                "tags": tags,
                                "created_at": datetime.now().isoformat(),
                                "is_blob": is_blob
                            }
                        )
                    ]
                )
            
            return True, None
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}")
            logger.error(traceback.format_exc())
            return False, str(e)
    
    async def retrieve(self, query: str, n_results: int = 5) -> List[MemoryQueryResult]:
        """Retrieve memories using semantic search."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_function([query])[0].tolist()
            
            # Use Qdrant for vector search if available
            if self.qdrant_client:
                search_results = self.qdrant_client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_embedding,
                    limit=n_results,
                    with_payload=True,
                    score_threshold=0.0  # No threshold, we'll filter later if needed
                )
                
                # Get memory IDs and scores
                memory_ids = [hit.id for hit in search_results]
                scores = [hit.score for hit in search_results]
                
                if not memory_ids:
                    return []
                
                # Fetch full memory data from PostgreSQL
                async with self.pg_pool.acquire() as conn:
                    memories = await conn.fetch(
                        """
                        SELECT m.*, array_agg(t.tag) as tags
                        FROM memories m
                        LEFT JOIN memory_tags t ON m.id = t.memory_id
                        WHERE m.id = ANY($1)
                        GROUP BY m.id
                        """,
                        memory_ids
                    )
                
                # Create memory objects
                memory_results = []
                for i, memory_row in enumerate(memories):
                    # Get the corresponding score
                    idx = memory_ids.index(memory_row['id'])
                    score = scores[idx] if idx < len(scores) else 0.0
                    
                    # Retrieve full content if it's a blob
                    content = memory_row['content']
                    if memory_row['is_blob'] and memory_row['blob_key'] and self.r2_client:
                        try:
                            content = await self._retrieve_blob(memory_row['blob_key'])
                        except Exception as e:
                            logger.error(f"Error retrieving blob {memory_row['blob_key']}: {str(e)}")
                    
                    # Parse metadata
                    metadata = json.loads(memory_row['metadata']) if memory_row['metadata'] else {}
                    
                    # Create memory object
                    memory = Memory(
                        content=content,
                        content_hash=memory_row['content_hash'],
                        tags=memory_row['tags'] if memory_row['tags'] else [],
                        memory_type=memory_row['memory_type'],
                        metadata=metadata
                    )
                    
                    memory_results.append(MemoryQueryResult(memory=memory, relevance_score=score))
                
                return memory_results
            else:
                # Fall back to PostgreSQL vector search
                async with self.pg_pool.acquire() as conn:
                    results = await conn.fetch(
                        """
                        SELECT m.*, array_agg(t.tag) as tags,
                               1 - (m.embedding <=> $1) as similarity
                        FROM memories m
                        LEFT JOIN memory_tags t ON m.id = t.memory_id
                        GROUP BY m.id
                        ORDER BY similarity DESC
                        LIMIT $2
                        """,
                        query_embedding,
                        n_results
                    )
                
                # Create memory objects
                memory_results = []
                for row in results:
                    # Retrieve full content if it's a blob
                    content = row['content']
                    if row['is_blob'] and row['blob_key'] and self.r2_client:
                        try:
                            content = await self._retrieve_blob(row['blob_key'])
                        except Exception as e:
                            logger.error(f"Error retrieving blob {row['blob_key']}: {str(e)}")
                    
                    # Parse metadata
                    metadata = json.loads(row['metadata']) if row['metadata'] else {}
                    
                    # Create memory object
                    memory = Memory(
                        content=content,
                        content_hash=row['content_hash'],
                        tags=row['tags'] if row['tags'] else [],
                        memory_type=row['memory_type'],
                        metadata=metadata
                    )
                    
                    memory_results.append(MemoryQueryResult(memory=memory, relevance_score=row['similarity']))
                
                return memory_results
        except Exception as e:
            logger.error(f"Error retrieving memories: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    async def search_by_tag(self, tags: List[str]) -> List[Memory]:
        """Search memories by tags."""
        try:
            if not tags:
                return []
            
            async with self.pg_pool.acquire() as conn:
                results = await conn.fetch(
                    """
                    SELECT m.*, array_agg(t.tag) as all_tags
                    FROM memories m
                    JOIN memory_tags t ON m.id = t.memory_id
                    WHERE t.tag = ANY($1)
                    GROUP BY m.id
                    """,
                    tags
                )
                
                memories = []
                for row in results:
                    # Retrieve full content if it's a blob
                    content = row['content']
                    if row['is_blob'] and row['blob_key'] and self.r2_client:
                        try:
                            content = await self._retrieve_blob(row['blob_key'])
                        except Exception as e:
                            logger.error(f"Error retrieving blob {row['blob_key']}: {str(e)}")
                    
                    # Parse metadata
                    metadata = json.loads(row['metadata']) if row['metadata'] else {}
                    
                    # Create memory object
                    memory = Memory(
                        content=content,
                        content_hash=row['content_hash'],
                        tags=row['all_tags'] if row['all_tags'] else [],
                        memory_type=row['memory_type'],
                        metadata=metadata
                    )
                    
                    memories.append(memory)
                
                return memories
        except Exception as e:
            logger.error(f"Error searching by tags: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    async def delete(self, content_hash: str) -> Tuple[bool, str]:
        """Delete a memory by its hash."""
        try:
            # Get blob key before deleting
            blob_key = None
            async with self.pg_pool.acquire() as conn:
                memory = await conn.fetchrow(
                    "SELECT is_blob, blob_key FROM memories WHERE content_hash = $1",
                    content_hash
                )
                
                if not memory:
                    return False, f"No memory found with hash {content_hash}"
                
                if memory['is_blob'] and memory['blob_key']:
                    blob_key = memory['blob_key']
                
                # Delete from PostgreSQL (cascade will delete tags)
                await conn.execute(
                    "DELETE FROM memories WHERE content_hash = $1",
                    content_hash
                )
            
            # Delete from Qdrant if enabled
            if self.qdrant_client:
                self.qdrant_client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=qdrant_models.PointIdsList(
                        points=[content_hash]
                    )
                )
            
            # Delete blob if exists
            if blob_key and self.r2_client:
                try:
                    self.r2_client.delete_object(
                        Bucket=os.environ.get("R2_BUCKET"),
                        Key=blob_key
                    )
                    logger.info(f"Deleted blob {blob_key}")
                except Exception as e:
                    logger.warning(f"Error deleting blob {blob_key}: {str(e)}")
            
            return True, f"Successfully deleted memory with hash {content_hash}"
        except Exception as e:
            logger.error(f"Error deleting memory: {str(e)}")
            logger.error(traceback.format_exc())
            return False, str(e)
    
    async def delete_by_tag(self, tag: str) -> Tuple[int, str]:
        """Delete memories by tag."""
        try:
            # Get memory IDs with the tag
            async with self.pg_pool.acquire() as conn:
                memory_ids = await conn.fetch(
                    "SELECT memory_id FROM memory_tags WHERE tag = $1",
                    tag
                )
                
                if not memory_ids:
                    return 0, f"No memories found with tag: {tag}"
                
                # Get blob keys for memories to be deleted
                blob_keys = await conn.fetch(
                    """
                    SELECT blob_key FROM memories 
                    WHERE id = ANY($1) AND is_blob = true AND blob_key IS NOT NULL
                    """,
                    [row['memory_id'] for row in memory_ids]
                )
                
                # Delete memories (cascade will delete tags)
                await conn.execute(
                    "DELETE FROM memories WHERE id = ANY($1)",
                    [row['memory_id'] for row in memory_ids]
                )
            
            # Delete from Qdrant if enabled
            if self.qdrant_client:
                self.qdrant_client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=qdrant_models.PointIdsList(
                        points=[row['memory_id'] for row in memory_ids]
                    )
                )
            
            # Delete blobs if any
            if blob_keys and self.r2_client:
                bucket = os.environ.get("R2_BUCKET")
                for row in blob_keys:
                    try:
                        self.r2_client.delete_object(
                            Bucket=bucket,
                            Key=row['blob_key']
                        )
                    except Exception as e:
                        logger.warning(f"Error deleting blob {row['blob_key']}: {str(e)}")
            
            return len(memory_ids), f"Successfully deleted {len(memory_ids)} memories with tag: {tag}"
        except Exception as e:
            logger.error(f"Error deleting memories by tag: {str(e)}")
            logger.error(traceback.format_exc())
            return 0, str(e)
    
    async def cleanup_duplicates(self) -> Tuple[int, str]:
        """Remove duplicate memories."""
        try:
            async with self.pg_pool.acquire() as conn:
                # Find duplicates based on content_hash
                duplicates = await conn.fetch(
                    """
                    SELECT content_hash, COUNT(*) as count, array_agg(id) as ids
                    FROM memories
                    GROUP BY content_hash
                    HAVING COUNT(*) > 1
                    """
                )
                
                if not duplicates:
                    return 0, "No duplicate memories found"
                
                # For each set of duplicates, keep the first one and delete the rest
                total_deleted = 0
                for dup in duplicates:
                    # Keep the first ID, delete the rest
                    ids_to_delete = dup['ids'][1:]
                    
                    # Get blob keys for memories to be deleted
                    blob_keys = await conn.fetch(
                        """
                        SELECT blob_key FROM memories 
                        WHERE id = ANY($1) AND is_blob = true AND blob_key IS NOT NULL
                        """,
                        ids_to_delete
                    )
                    
                    # Delete duplicates
                    await conn.execute(
                        "DELETE FROM memories WHERE id = ANY($1)",
                        ids_to_delete
                    )
                    
                    # Delete from Qdrant if enabled
                    if self.qdrant_client:
                        self.qdrant_client.delete(
                            collection_name=COLLECTION_NAME,
                            points_selector=qdrant_models.PointIdsList(
                                points=ids_to_delete
                            )
                        )
                    
                    # Delete blobs if any
                    if blob_keys and self.r2_client:
                        bucket = os.environ.get("R2_BUCKET")
                        for row in blob_keys:
                            try:
                                self.r2_client.delete_object(
                                    Bucket=bucket,
                                    Key=row['blob_key']
                                )
                            except Exception as e:
                                logger.warning(f"Error deleting blob {row['blob_key']}: {str(e)}")
                    
                    total_deleted += len(ids_to_delete)
            
            return total_deleted, f"Successfully removed {total_deleted} duplicate memories"
        except Exception as e:
            logger.error(f"Error cleaning up duplicates: {str(e)}")
            logger.error(traceback.format_exc())
            return 0, str(e)
    
    async def recall(self, query: Optional[str] = None, n_results: int = 5, 
                    start_timestamp: Optional[float] = None, 
                    end_timestamp: Optional[float] = None) -> List[MemoryQueryResult]:
        """
        Retrieve memories with combined time filtering and optional semantic search.
        """
        try:
            # Build SQL query parts
            where_clauses = []
            params = []
            param_idx = 1
            
            if start_timestamp is not None:
                where_clauses.append(f"created_at >= to_timestamp(${param_idx})")
                params.append(start_timestamp)
                param_idx += 1
            
            if end_timestamp is not None:
                where_clauses.append(f"created_at <= to_timestamp(${param_idx})")
                params.append(end_timestamp)
                param_idx += 1
            
            where_sql = " AND ".join(where_clauses)
            if where_sql:
                where_sql = "WHERE " + where_sql
            
            # If query is provided, do semantic search
            if query:
                # Generate query embedding
                query_embedding = self.embedding_function([query])[0].tolist()
                
                # Use Qdrant for vector search if available
                if self.qdrant_client:
                    # Build Qdrant filter
                    filter_obj = None
                    if start_timestamp is not None or end_timestamp is not None:
                        filter_conditions = []
                        
                        if start_timestamp is not None:
                            start_dt = datetime.fromtimestamp(start_timestamp).isoformat()
                            filter_conditions.append(
                                qdrant_models.FieldCondition(
                                    key="created_at",
                                    match=qdrant_models.MatchValue(
                                        value=start_dt
                                    ),
                                    range=qdrant_models.Range(
                                        gte=start_dt
                                    )
                                )
                            )
                        
                        if end_timestamp is not None:
                            end_dt = datetime.fromtimestamp(end_timestamp).isoformat()
                            filter_conditions.append(
                                qdrant_models.FieldCondition(
                                    key="created_at",
                                    match=qdrant_models.MatchValue(
                                        value=end_dt
                                    ),
                                    range=qdrant_models.Range(
                                        lte=end_dt
                                    )
                                )
                            )
                        
                        if filter_conditions:
                            filter_obj = qdrant_models.Filter(
                                must=filter_conditions
                            )
                    
                    # Search in Qdrant
                    search_results = self.qdrant_client.search(
                        collection_name=COLLECTION_NAME,
                        query_vector=query_embedding,
                        limit=n_results,
                        with_payload=True,
                        filter=filter_obj
                    )
                    
                    # Get memory IDs and scores
                    memory_ids = [hit.id for hit in search_results]
                    scores = [hit.score for hit in search_results]
                    
                    if not memory_ids:
                        return []
                    
                    # Fetch full memory data from PostgreSQL
                    async with self.pg_pool.acquire() as conn:
                        memories = await conn.fetch(
                            """
                            SELECT m.*, array_agg(t.tag) as tags
                            FROM memories m
                            LEFT JOIN memory_tags t ON m.id = t.memory_id
                            WHERE m.id = ANY($1)
                            GROUP BY m.id
                            """,
                            memory_ids
                        )
                    
                    # Create memory objects
                    memory_results = []
                    for memory_row in memories:
                        # Get the corresponding score
                        idx = memory_ids.index(memory_row['id'])
                        score = scores[idx] if idx < len(scores) else 0.0
                        
                        # Retrieve full content if it's a blob
                        content = memory_row['content']
                        if memory_row['is_blob'] and memory_row['blob_key'] and self.r2_client:
                            try:
                                content = await self._retrieve_blob(memory_row['blob_key'])
                            except Exception as e:
                                logger.error(f"Error retrieving blob {memory_row['blob_key']}: {str(e)}")
                        
                        # Parse metadata
                        metadata = json.loads(memory_row['metadata']) if memory_row['metadata'] else {}
                        
                        # Create memory object
                        memory = Memory(
                            content=content,
                            content_hash=memory_row['content_hash'],
                            tags=memory_row['tags'] if memory_row['tags'] else [],
                            memory_type=memory_row['memory_type'],
                            metadata=metadata
                        )
                        
                        memory_results.append(MemoryQueryResult(memory=memory, relevance_score=score))
                    
                    return memory_results
                else:
                    # Fall back to PostgreSQL vector search
                    async with self.pg_pool.acquire() as conn:
                        # Add embedding parameter
                        params.append(query_embedding)
                        
                        # Build SQL query
                        sql = f"""
                            SELECT m.*, array_agg(t.tag) as tags,
                                   1 - (m.embedding <=> ${param_idx}) as similarity
                            FROM memories m
                            LEFT JOIN memory_tags t ON m.id = t.memory_id
                            {where_sql}
                            GROUP BY m.id
                            ORDER BY similarity DESC
                            LIMIT ${param_idx + 1}
                        """
                        
                        # Add limit parameter
                        params.append(n_results)
                        
                        # Execute query
                        results = await conn.fetch(sql, *params)
                        
                        # Create memory objects
                        memory_results = []
                        for row in results:
                            # Retrieve full content if it's a blob
                            content = row['content']
                            if row['is_blob'] and row['blob_key'] and self.r2_client:
                                try:
                                    content = await self._retrieve_blob(row['blob_key'])
                                except Exception as e:
                                    logger.error(f"Error retrieving blob {row['blob_key']}: {str(e)}")
                            
                            # Parse metadata
                            metadata = json.loads(row['metadata']) if row['metadata'] else {}
                            
                            # Create memory object
                            memory = Memory(
                                content=content,
                                content_hash=row['content_hash'],
                                tags=row['tags'] if row['tags'] else [],
                                memory_type=row['memory_type'],
                                metadata=metadata
                            )
                            
                            memory_results.append(MemoryQueryResult(memory=memory, relevance_score=row['similarity']))
                        
                        return memory_results
            else:
                # Time-based filtering only
                async with self.pg_pool.acquire() as conn:
                    # Build SQL query
                    sql = f"""
                        SELECT m.*, array_agg(t.tag) as tags
                        FROM memories m
                        LEFT JOIN memory_tags t ON m.id = t.memory_id
                        {where_sql}
                        GROUP BY m.id
                        ORDER BY m.created_at DESC
                        LIMIT ${param_idx}
                    """
                    
                    # Add limit parameter
                    params.append(n_results)
                    
                    # Execute query
                    results = await conn.fetch(sql, *params)
                    
                    # Create memory objects
                    memory_results = []
                    for row in results:
                        # Retrieve full content if it's a blob
                        content = row['content']
                        if row['is_blob'] and row['blob_key'] and self.r2_client:
                            try:
                                content = await self._retrieve_blob(row['blob_key'])
                            except Exception as e:
                                logger.error(f"Error retrieving blob {row['blob_key']}: {str(e)}")
                        
                        # Parse metadata
                        metadata = json.loads(row['metadata']) if row['metadata'] else {}
                        
                        # Create memory object
                        memory = Memory(
                            content=content,
                            content_hash=row['content_hash'],
                            tags=row['tags'] if row['tags'] else [],
                            memory_type=row['memory_type'],
                            metadata=metadata
                        )
                        
                        memory_results.append(MemoryQueryResult(memory=memory, relevance_score=None))
                    
                    return memory_results
        except Exception as e:
            logger.error(f"Error in recall: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    async def delete_by_timeframe(self, start_date: date, end_date: Optional[date] = None, 
                                 tag: Optional[str] = None) -> Tuple[int, str]:
        """Delete memories within a timeframe and optionally filtered by tag."""
        try:
            if end_date is None:
                end_date = start_date
            
            start_datetime = datetime.combine(start_date, datetime.min.time())
            end_datetime = datetime.combine(end_date, datetime.max.time())
            
            # Build query conditions
            conditions = ["created_at BETWEEN $1 AND $2"]
            params = [start_datetime, end_datetime]
            
            if tag:
                conditions.append("id IN (SELECT memory_id FROM memory_tags WHERE tag = $3)")
                params.append(tag)
            
            where_clause = " AND ".join(conditions)
            
            # Get memory IDs and blob keys to be deleted
            async with self.pg_pool.acquire() as conn:
                # Get blob keys
                blob_keys = await conn.fetch(
                    f"""
                    SELECT blob_key FROM memories 
                    WHERE {where_clause} AND is_blob = true AND blob_key IS NOT NULL
                    """,
                    *params
                )
                
                # Get memory IDs for Qdrant
                if self.qdrant_client:
                    memory_ids = await conn.fetch(
                        f"SELECT id FROM memories WHERE {where_clause}",
                        *params
                    )
                
                # Delete memories
                result = await conn.execute(
                    f"DELETE FROM memories WHERE {where_clause}",
                    *params
                )
                
                # Parse delete count
                count = int(result.split(" ")[1])
                
                if count == 0:
                    return 0, "No memories found matching the criteria"
            
            # Delete from Qdrant if enabled
            if self.qdrant_client and memory_ids:
                self.qdrant_client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=qdrant_models.PointIdsList(
                        points=[row['id'] for row in memory_ids]
                    )
                )
            
            # Delete blobs if any
            if blob_keys and self.r2_client:
                bucket = os.environ.get("R2_BUCKET")
                for row in blob_keys:
                    try:
                        self.r2_client.delete_object(
                            Bucket=bucket,
                            Key=row['blob_key']
                        )
                    except Exception as e:
                        logger.warning(f"Error deleting blob {row['blob_key']}: {str(e)}")
            
            return count, f"Successfully deleted {count} memories"
        except Exception as e:
            logger.error(f"Error deleting memories by timeframe: {str(e)}")
            logger.error(traceback.format_exc())
            return 0, str(e)
    
    async def delete_before_date(self, before_date: date, tag: Optional[str] = None) -> Tuple[int, str]:
        """Delete memories before a given date and optionally filtered by tag."""
        try:
            before_datetime = datetime.combine(before_date, datetime.max.time())
            
            # Build query conditions
            conditions = ["created_at < $1"]
            params = [before_datetime]
            
            if tag:
                conditions.append("id IN (SELECT memory_id FROM memory_tags WHERE tag = $2)")
                params.append(tag)
            
            where_clause = " AND ".join(conditions)
            
            # Get memory IDs and blob keys to be deleted
            async with self.pg_pool.acquire() as conn:
                # Get blob keys
                blob_keys = await conn.fetch(
                    f"""
                    SELECT blob_key FROM memories 
                    WHERE {where_clause} AND is_blob = true AND blob_key IS NOT NULL
                    """,
                    *params
                )
                
                # Get memory IDs for Qdrant
                if self.qdrant_client:
                    memory_ids = await conn.fetch(
                        f"SELECT id FROM memories WHERE {where_clause}",
                        *params
                    )
                
                # Delete memories
                result = await conn.execute(
                    f"DELETE FROM memories WHERE {where_clause}",
                    *params
                )
                
                # Parse delete count
                count = int(result.split(" ")[1])
                
                if count == 0:
                    return 0, "No memories found matching the criteria"
            
            # Delete from Qdrant if enabled
            if self.qdrant_client and memory_ids:
                self.qdrant_client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=qdrant_models.PointIdsList(
                        points=[row['id'] for row in memory_ids]
                    )
                )
            
            # Delete blobs if any
            if blob_keys and self.r2_client:
                bucket = os.environ.get("R2_BUCKET")
                for row in blob_keys:
                    try:
                        self.r2_client.delete_object(
                            Bucket=bucket,
                            Key=row['blob_key']
                        )
                    except Exception as e:
                        logger.warning(f"Error deleting blob {row['blob_key']}: {str(e)}")
            
            return count, f"Successfully deleted {count} memories"
        except Exception as e:
            logger.error(f"Error deleting memories before date: {str(e)}")
            logger.error(traceback.format_exc())
            return 0, str(e)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            stats = {
                "status": "healthy",
                "storage_type": "echovault",
                "providers": []
            }
            
            # PostgreSQL stats
            if self.pg_pool:
                async with self.pg_pool.acquire() as conn:
                    # Get memory count
                    memory_count = await conn.fetchval("SELECT COUNT(*) FROM memories")
                    
                    # Get tag count
                    tag_count = await conn.fetchval("SELECT COUNT(DISTINCT tag) FROM memory_tags")
                    
                    # Get blob count
                    blob_count = await conn.fetchval("SELECT COUNT(*) FROM memories WHERE is_blob = true")
                    
                    # Get database size
                    db_size = await conn.fetchval(
                        "SELECT pg_size_pretty(pg_database_size(current_database()))"
                    )
                    
                    stats["providers"].append({
                        "name": "postgresql",
                        "status": "connected",
                        "memory_count": memory_count,
                        "tag_count": tag_count,
                        "blob_count": blob_count,
                        "database_size": db_size
                    })
            
            # Qdrant stats
            if self.qdrant_client:
                collection_info = self.qdrant_client.get_collection(COLLECTION_NAME)
                
                stats["providers"].append({
                    "name": "qdrant",
                    "status": "connected",
                    "vector_count": collection_info.vectors_count,
                    "vector_size": VECTOR_SIZE,
                    "distance": "cosine"
                })
            
            # R2 stats
            if self.r2_client:
                bucket = os.environ.get("R2_BUCKET")
                try:
                    # Get bucket objects
                    response = self.r2_client.list_objects_v2(Bucket=bucket)
                    object_count = response.get('KeyCount', 0)
                    
                    stats["providers"].append({
                        "name": "r2",
                        "status": "connected",
                        "bucket": bucket,
                        "object_count": object_count
                    })
                except Exception as e:
                    stats["providers"].append({
                        "name": "r2",
                        "status": "error",
                        "error": str(e)
                    })
            
            return stats
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def create_backup(self) -> Tuple[bool, str, Optional[str]]:
        """Create a backup of all memories."""
        try:
            # Generate backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"echovault_backup_{timestamp}.json"
            
            # Get all memories
            async with self.pg_pool.acquire() as conn:
                memories = await conn.fetch(
                    """
                    SELECT m.*, array_agg(t.tag) as tags
                    FROM memories m
                    LEFT JOIN memory_tags t ON m.id = t.memory_id
                    GROUP BY m.id
                    """
                )
            
            # Create backup data structure
            backup_data = {
                "timestamp": datetime.now().isoformat(),
                "total_memories": len(memories),
                "memories": []
            }
            
            # Process each memory
            for memory_row in memories:
                # Retrieve full content if it's a blob
                content = memory_row['content']
                if memory_row['is_blob'] and memory_row['blob_key'] and self.r2_client:
                    try:
                        content = await self._retrieve_blob(memory_row['blob_key'])
                    except Exception as e:
                        logger.error(f"Error retrieving blob {memory_row['blob_key']}: {str(e)}")
                
                # Parse metadata
                metadata = json.loads(memory_row['metadata']) if memory_row['metadata'] else {}
                
                # Add to backup
                backup_data["memories"].append({
                    "id": memory_row['id'],
                    "content": content,
                    "content_hash": memory_row['content_hash'],
                    "memory_type": memory_row['memory_type'],
                    "tags": memory_row['tags'] if memory_row['tags'] else [],
                    "created_at": memory_row['created_at'].isoformat(),
                    "updated_at": memory_row['updated_at'].isoformat(),
                    "metadata": metadata,
                    "is_blob": memory_row['is_blob'],
                    "blob_key": memory_row['blob_key']
                })
            
            # Store backup in R2
            if self.r2_client:
                bucket = os.environ.get("R2_BUCKET")
                backup_key = f"backups/{backup_file}"
                
                self.r2_client.put_object(
                    Bucket=bucket,
                    Key=backup_key,
                    Body=json.dumps(backup_data, indent=2).encode('utf-8'),
                    ContentType='application/json'
                )
                
                logger.info(f"Backup created in R2: {backup_key}")
                return True, f"Backup created successfully in R2: {backup_key}", backup_key
            else:
                # Fall back to local backup
                if not os.path.exists("backups"):
                    os.makedirs("backups")
                
                backup_path = os.path.join("backups", backup_file)
                with open(backup_path, 'w') as f:
                    json.dump(backup_data, f, indent=2)
                
                logger.info(f"Backup created locally: {backup_path}")
                return True, f"Backup created successfully: {backup_path}", backup_path
        except Exception as e:
            logger.error(f"Error creating backup: {str(e)}")
            logger.error(traceback.format_exc())
            return False, f"Error creating backup: {str(e)}", None
    
    async def restore_backup(self, backup_path: str) -> Tuple[bool, str]:
        """Restore memories from a backup file."""
        try:
            # Load backup data
            if backup_path.startswith("backups/") and self.r2_client:
                # Load from R2
                bucket = os.environ.get("R2_BUCKET")
                response = self.r2_client.get_object(
                    Bucket=bucket,
                    Key=backup_path
                )
                backup_data = json.loads(response['Body'].read().decode('utf-8'))
            else:
                # Load from local file
                with open(backup_path, 'r') as f:
                    backup_data = json.load(f)
            
            # Validate backup data
            if not isinstance(backup_data, dict) or "memories" not in backup_data:
                return False, "Invalid backup format"
            
            # Clear existing data
            async with self.pg_pool.acquire() as conn:
                await conn.execute("DELETE FROM memories")
            
            if self.qdrant_client:
                self.qdrant_client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="content_hash",
                                match=qdrant_models.MatchAny(any=["*"])
                            )
                        ]
                    )
                )
            
            # Restore memories
            total_memories = len(backup_data["memories"])
            restored_count = 0
            
            for memory_data in backup_data["memories"]:
                try:
                    # Check if content is a blob
                    content = memory_data["content"]
                    is_blob = memory_data.get("is_blob", False)
                    blob_key = memory_data.get("blob_key")
                    
                    # Store blob if needed
                    if is_blob and self.r2_client and len(content) > BLOB_THRESHOLD:
                        if not blob_key:
                            blob_key = await self._store_blob(content)
                        else:
                            # Check if blob exists, if not, store it
                            try:
                                self.r2_client.head_object(
                                    Bucket=os.environ.get("R2_BUCKET"),
                                    Key=blob_key
                                )
                            except Exception:
                                # Blob doesn't exist, store it
                                self.r2_client.put_object(
                                    Bucket=os.environ.get("R2_BUCKET"),
                                    Key=blob_key,
                                    Body=content.encode('utf-8'),
                                    ContentType='text/plain'
                                )
                        
                        # Store truncated content in database
                        stored_content = f"{content[:1000]}... [TRUNCATED - Full content in blob {blob_key}]"
                    else:
                        stored_content = content
                        is_blob = False
                        blob_key = None
                    
                    # Parse created_at and updated_at
                    created_at = datetime.fromisoformat(memory_data["created_at"].replace('Z', '+00:00'))
                    updated_at = datetime.fromisoformat(memory_data["updated_at"].replace('Z', '+00:00'))
                    
                    # Generate embedding
                    embedding = self.embedding_function([content])[0].tolist()
                    
                    # Store in PostgreSQL
                    async with self.pg_pool.acquire() as conn:
                        async with conn.transaction():
                            # Insert memory
                            await conn.execute(
                                """
                                INSERT INTO memories (
                                    id, content, content_hash, memory_type, 
                                    created_at, updated_at, embedding, metadata,
                                    is_blob, blob_key
                                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                                ON CONFLICT (id) DO NOTHING
                                """,
                                memory_data["id"],
                                stored_content,
                                memory_data["content_hash"],
                                memory_data["memory_type"],
                                created_at,
                                updated_at,
                                embedding,
                                json.dumps(memory_data["metadata"]),
                                is_blob,
                                blob_key
                            )
                            
                            # Insert tags
                            if memory_data["tags"]:
                                await conn.executemany(
                                    """
                                    INSERT INTO memory_tags (memory_id, tag)
                                    VALUES ($1, $2)
                                    ON CONFLICT (memory_id, tag) DO NOTHING
                                    """,
                                    [(memory_data["id"], tag) for tag in memory_data["tags"]]
                                )
                    
                    # Store in Qdrant if enabled
                    if self.qdrant_client:
                        self.qdrant_client.upsert(
                            collection_name=COLLECTION_NAME,
                            points=[
                                qdrant_models.PointStruct(
                                    id=memory_data["id"],
                                    vector=embedding,
                                    payload={
                                        "content_hash": memory_data["content_hash"],
                                        "memory_type": memory_data["memory_type"],
                                        "tags": memory_data["tags"],
                                        "created_at": memory_data["created_at"],
                                        "is_blob": is_blob
                                    }
                                )
                            ]
                        )
                    
                    restored_count += 1
                except Exception as e:
                    logger.error(f"Error restoring memory {memory_data.get('id')}: {str(e)}")
            
            return True, f"Restored {restored_count}/{total_memories} memories successfully"
        except Exception as e:
            logger.error(f"Error restoring backup: {str(e)}")
            logger.error(traceback.format_exc())
            return False, f"Error restoring backup: {str(e)}"