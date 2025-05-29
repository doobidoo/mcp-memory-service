"""
Vector Store Client for EchoVault Memory Service
Copyright (c) 2025 EchoVault
Licensed under the MIT License.

This module provides a client for vector stores like Qdrant,
with a fallback to pgvector or ChromaDB.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class VectorStoreClient:
    """
    Client for vector stores with support for Qdrant and pgvector.
    Provides a unified interface for vector operations.
    """
    
    def __init__(self):
        """Initialize the vector store client."""
        self.use_qdrant = os.environ.get("USE_QDRANT", "").lower() in ("true", "1", "yes")
        self.qdrant_client = None
        self.neon_client = None
        self.collection_name = "memories"
        self._is_initialized = False
    
    async def initialize(self):
        """Initialize the vector store client."""
        if self._is_initialized:
            return
            
        try:
            # Initialize Neon client for fallback
            try:
                from .neon_client import NeonClient
                self.neon_client = NeonClient()
                await self.neon_client.initialize()
                logger.info("Initialized Neon client for vector operations")
            except ImportError:
                logger.warning("NeonClient not available, pgvector operations will not be supported")
            except Exception as e:
                logger.error(f"Failed to initialize Neon client: {e}")
            
            # Initialize Qdrant client if enabled
            if self.use_qdrant:
                try:
                    import qdrant_client
                    from qdrant_client import QdrantClient
                    from qdrant_client.http import models
                    
                    qdrant_url = os.environ.get("QDRANT_URL")
                    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
                    
                    if not qdrant_url:
                        logger.error("QDRANT_URL environment variable is not set")
                        self.use_qdrant = False
                    else:
                        # Connect to Qdrant
                        self.qdrant_client = QdrantClient(
                            url=qdrant_url,
                            api_key=qdrant_api_key
                        )
                        
                        # Create collection if it doesn't exist
                        collections = self.qdrant_client.get_collections().collections
                        collection_names = [c.name for c in collections]
                        
                        if self.collection_name not in collection_names:
                            self.qdrant_client.create_collection(
                                collection_name=self.collection_name,
                                vectors_config=models.VectorParams(
                                    size=1536,  # OpenAI ada-002 embedding size
                                    distance=models.Distance.COSINE
                                )
                            )
                        
                        logger.info(f"Initialized Qdrant client at {qdrant_url}")
                except ImportError:
                    logger.warning("Qdrant client not available, falling back to pgvector")
                    self.use_qdrant = False
                except Exception as e:
                    logger.error(f"Failed to initialize Qdrant client: {e}")
                    self.use_qdrant = False
            
            self._is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store client: {e}")
            raise
    
    async def close(self):
        """Close connections to vector stores."""
        if self.neon_client:
            await self.neon_client.close()
        
        if self.qdrant_client:
            self.qdrant_client.close()
        
        self._is_initialized = False
    
    async def upsert(self, 
                    id: str,
                    content: str,
                    embedding: List[float],
                    metadata: Dict[str, Any]) -> bool:
        """
        Insert or update a vector in the store.
        
        Args:
            id: Unique identifier for the vector
            content: Content associated with the vector
            embedding: Vector embedding
            metadata: Additional metadata
            
        Returns:
            True if the operation was successful
        """
        if not self._is_initialized:
            await self.initialize()
        
        success = True
        
        # Try Qdrant first if enabled
        if self.use_qdrant and self.qdrant_client:
            try:
                import qdrant_client
                from qdrant_client.http import models
                
                # Prepare payload with all metadata
                payload = {
                    "content": content,
                    **metadata
                }
                
                # Insert into Qdrant
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        models.PointStruct(
                            id=id,
                            vector=embedding,
                            payload=payload
                        )
                    ]
                )
                
                logger.debug(f"Inserted vector into Qdrant with ID: {id}")
            except Exception as e:
                logger.error(f"Failed to insert vector into Qdrant: {e}")
                success = False
        
        # Always insert into Neon for durability
        if self.neon_client:
            try:
                # Extract tags from metadata
                tags = metadata.get("tags", [])
                memory_type = metadata.get("memory_type", "")
                timestamp = metadata.get("timestamp")
                payload_url = metadata.get("payload_url")
                content_hash = metadata.get("content_hash", id)
                
                # Remove special fields from metadata
                clean_metadata = {k: v for k, v in metadata.items() 
                                 if k not in ["tags", "memory_type", "timestamp", "payload_url", "content_hash"]}
                
                # Insert into Neon
                await self.neon_client.insert_event(
                    content=content,
                    content_hash=content_hash,
                    embedding=embedding,
                    tags=tags,
                    memory_type=memory_type,
                    metadata=clean_metadata,
                    timestamp=timestamp,
                    payload_url=payload_url
                )
                
                logger.debug(f"Inserted vector into Neon with ID: {id}")
            except Exception as e:
                logger.error(f"Failed to insert vector into Neon: {e}")
                success = False
        
        return success
    
    async def search(self, 
                   embedding: List[float],
                   limit: int = 5,
                   similarity_threshold: float = 0.7,
                   filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            embedding: Query vector embedding
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity threshold
            filter_dict: Optional filter criteria
            
        Returns:
            List of matching results with similarity scores
        """
        if not self._is_initialized:
            await self.initialize()
        
        results = []
        
        # Try Qdrant first if enabled
        if self.use_qdrant and self.qdrant_client:
            try:
                import qdrant_client
                from qdrant_client.http import models
                
                # Prepare filter if provided
                filter_obj = None
                if filter_dict:
                    # Convert filter_dict to Qdrant filter format
                    filter_conditions = []
                    
                    # Handle time range filtering
                    if "start_timestamp" in filter_dict and "end_timestamp" in filter_dict:
                        filter_conditions.append(
                            models.FieldCondition(
                                key="timestamp",
                                range=models.Range(
                                    gte=filter_dict["start_timestamp"],
                                    lte=filter_dict["end_timestamp"]
                                )
                            )
                        )
                    
                    # Handle tag filtering
                    if "tags" in filter_dict:
                        tags = filter_dict["tags"]
                        if isinstance(tags, list) and tags:
                            # Match any of the provided tags
                            filter_conditions.append(
                                models.FieldCondition(
                                    key="tags",
                                    match=models.MatchAny(any=tags)
                                )
                            )
                    
                    if filter_conditions:
                        filter_obj = models.Filter(
                            must=filter_conditions
                        )
                
                # Search in Qdrant
                qdrant_results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=embedding,
                    limit=limit,
                    score_threshold=similarity_threshold,
                    with_payload=True,
                    filter=filter_obj
                )
                
                # Process results
                for point in qdrant_results:
                    payload = point.payload
                    
                    # Extract content and metadata
                    content = payload.pop("content", "")
                    
                    results.append({
                        "id": point.id,
                        "content": content,
                        "similarity": point.score,
                        "metadata": payload
                    })
                
                logger.debug(f"Found {len(results)} results in Qdrant")
                
                # If we got results from Qdrant, return them
                if results:
                    return results
            except Exception as e:
                logger.error(f"Failed to search in Qdrant: {e}")
        
        # Fall back to Neon
        if self.neon_client:
            try:
                # Build filter for Neon
                filter_kwargs = {}
                
                if filter_dict:
                    if "start_timestamp" in filter_dict and "end_timestamp" in filter_dict:
                        filter_kwargs["start_timestamp"] = filter_dict["start_timestamp"]
                        filter_kwargs["end_timestamp"] = filter_dict["end_timestamp"]
                
                # Search in Neon
                neon_results = await self.neon_client.search_by_vector(
                    embedding=embedding,
                    limit=limit,
                    similarity_threshold=similarity_threshold
                )
                
                results = neon_results
                logger.debug(f"Found {len(results)} results in Neon")
            except Exception as e:
                logger.error(f"Failed to search in Neon: {e}")
        
        return results
    
    async def delete(self, id: str) -> bool:
        """
        Delete a vector from the store.
        
        Args:
            id: Unique identifier for the vector
            
        Returns:
            True if the deletion was successful
        """
        if not self._is_initialized:
            await self.initialize()
        
        success = True
        
        # Try Qdrant first if enabled
        if self.use_qdrant and self.qdrant_client:
            try:
                import qdrant_client
                
                # Delete from Qdrant
                self.qdrant_client.delete(
                    collection_name=self.collection_name,
                    points_selector=qdrant_client.http.models.PointIdsList(
                        points=[id]
                    )
                )
                
                logger.debug(f"Deleted vector from Qdrant with ID: {id}")
            except Exception as e:
                logger.error(f"Failed to delete vector from Qdrant: {e}")
                success = False
        
        # Always delete from Neon for consistency
        if self.neon_client:
            try:
                # Delete from Neon
                neon_success = await self.neon_client.delete_memory(id)
                if not neon_success:
                    success = False
                
                logger.debug(f"Deleted vector from Neon with ID: {id}")
            except Exception as e:
                logger.error(f"Failed to delete vector from Neon: {e}")
                success = False
        
        return success
    
    async def delete_by_tag(self, tag: str) -> int:
        """
        Delete vectors by tag.
        
        Args:
            tag: Tag to filter by
            
        Returns:
            Number of vectors deleted
        """
        if not self._is_initialized:
            await self.initialize()
        
        count = 0
        
        # Delete from Neon first to get IDs
        if self.neon_client:
            try:
                # Delete from Neon and get count
                count = await self.neon_client.delete_by_tag(tag)
                
                logger.debug(f"Deleted {count} vectors from Neon with tag: {tag}")
            except Exception as e:
                logger.error(f"Failed to delete vectors from Neon: {e}")
        
        # Delete from Qdrant if enabled
        if self.use_qdrant and self.qdrant_client and count > 0:
            try:
                import qdrant_client
                from qdrant_client.http import models
                
                # Delete from Qdrant by tag
                self.qdrant_client.delete(
                    collection_name=self.collection_name,
                    points_selector=qdrant_client.http.models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="tags",
                                    match=models.MatchText(text=tag)
                                )
                            ]
                        )
                    )
                )
                
                logger.debug(f"Deleted vectors from Qdrant with tag: {tag}")
            except Exception as e:
                logger.error(f"Failed to delete vectors from Qdrant: {e}")
        
        return count
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        if not self._is_initialized:
            await self.initialize()
        
        stats = {
            "vector_count": 0,
            "providers": []
        }
        
        # Get stats from Neon
        if self.neon_client:
            try:
                neon_stats = await self.neon_client.get_memory_stats()
                stats["vector_count"] = neon_stats.get("memory_count", 0)
                stats["providers"].append({
                    "name": "neon_pgvector",
                    "stats": neon_stats
                })
            except Exception as e:
                logger.error(f"Failed to get stats from Neon: {e}")
        
        # Get stats from Qdrant if enabled
        if self.use_qdrant and self.qdrant_client:
            try:
                import qdrant_client
                
                # Get collection info
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                
                qdrant_stats = {
                    "vectors_count": collection_info.vectors_count,
                    "points_count": collection_info.points_count,
                    "segments_count": collection_info.segments_count,
                    "status": collection_info.status
                }
                
                stats["providers"].append({
                    "name": "qdrant",
                    "stats": qdrant_stats
                })
                
                # Use Qdrant's count as the primary count if available
                if collection_info.points_count > 0:
                    stats["vector_count"] = collection_info.points_count
            except Exception as e:
                logger.error(f"Failed to get stats from Qdrant: {e}")
        
        return stats