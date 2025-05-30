#!/usr/bin/env python3
"""
Nightly Memory Summarization Script for EchoVault Memory Service
Copyright (c) 2025 EchoVault
Licensed under the MIT License.

This script summarizes old memories to reduce database size while preserving
semantic information. It is designed to be run as a nightly cron job.
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("summarize_events.log")
    ]
)
logger = logging.getLogger("summarize_events")

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    # Import required modules
    from src.mcp_memory_service.storage.neon_client import NeonClient
    from src.mcp_memory_service.storage.vector_store import VectorStoreClient
    from src.mcp_memory_service.storage.blob_store import BlobStoreClient
    from src.mcp_memory_service.utils.hashing import generate_content_hash
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

# Try to import OpenAI for summarization
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI package not available. Will use simple concatenation for summaries.")
    OPENAI_AVAILABLE = False

class MemorySummarizer:
    """Memory summarization utility for old events."""
    
    def __init__(self):
        """Initialize the memory summarizer."""
        self.neon_client = NeonClient()
        self.vector_store = VectorStoreClient()
        self.blob_store = BlobStoreClient()
        self._is_initialized = False
        
        # Configuration
        self.summary_threshold_days = int(os.environ.get("SUMMARY_THRESHOLD_DAYS", "30"))
        self.max_memories_per_summary = int(os.environ.get("MAX_MEMORIES_PER_SUMMARY", "20"))
        self.min_memories_per_summary = int(os.environ.get("MIN_MEMORIES_PER_SUMMARY", "5"))
        self.max_summary_length = int(os.environ.get("MAX_SUMMARY_LENGTH", "4096"))
        self.retention_days = int(os.environ.get("RETENTION_DAYS", "365"))
        self.deletion_batch_size = int(os.environ.get("DELETION_BATCH_SIZE", "100"))
        
        # OpenAI configuration
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.openai_model = os.environ.get("OPENAI_SUMMARY_MODEL", "gpt-3.5-turbo")
    
    async def initialize(self):
        """Initialize connections to services."""
        if self._is_initialized:
            return
        
        try:
            await self.neon_client.initialize()
            await self.vector_store.initialize()
            await self.blob_store.initialize()
            
            # Initialize OpenAI if available
            if OPENAI_AVAILABLE and self.openai_api_key:
                openai.api_key = self.openai_api_key
                logger.info(f"OpenAI initialized with model: {self.openai_model}")
            
            self._is_initialized = True
            logger.info("Memory summarizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize memory summarizer: {e}")
            raise
    
    async def get_old_memories(self, days: int) -> List[Dict[str, Any]]:
        """
        Retrieve memories older than the specified number of days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of old memories
        """
        if not self._is_initialized:
            await self.initialize()
        
        try:
            # Calculate the timestamp threshold
            threshold_time = datetime.now(timezone.utc) - timedelta(days=days)
            threshold_timestamp = int(threshold_time.timestamp())
            
            # Get memories older than the threshold, grouped by tags and memory_type
            async with self.neon_client.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT 
                        id, content, content_hash, memory_type, tags, metadata,
                        timestamp, payload_url, embedding
                    FROM 
                        memories
                    WHERE 
                        timestamp < $1
                    ORDER BY
                        memory_type, tags, timestamp
                    LIMIT 1000
                """, threshold_timestamp)
                
                # Convert rows to dictionaries
                memories = []
                for row in rows:
                    # Parse tags from JSON if needed
                    try:
                        tags = json.loads(row["tags"]) if isinstance(row["tags"], str) else row["tags"]
                    except (json.JSONDecodeError, TypeError):
                        tags = []
                    
                    memories.append({
                        "id": row["id"],
                        "content": row["content"],
                        "content_hash": row["content_hash"],
                        "memory_type": row["memory_type"],
                        "tags": tags,
                        "metadata": row["metadata"],
                        "timestamp": row["timestamp"],
                        "payload_url": row["payload_url"],
                        "embedding": row["embedding"]
                    })
                
                logger.info(f"Found {len(memories)} memories older than {days} days")
                return memories
        except Exception as e:
            logger.error(f"Failed to retrieve old memories: {e}")
            raise
    
    async def group_memories_by_similarity(self, memories: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Group memories by semantic similarity.
        
        Args:
            memories: List of memories to group
            
        Returns:
            List of groups of similar memories
        """
        if len(memories) <= self.min_memories_per_summary:
            # Not enough memories to group, return as a single group
            return [memories]
        
        # First, try grouping by tags and memory_type
        tag_groups = {}
        for memory in memories:
            # Create a group key based on memory_type and tags
            memory_type = memory.get("memory_type", "")
            tags = memory.get("tags", [])
            
            # Sort tags for consistent grouping
            if isinstance(tags, list):
                sorted_tags = sorted(tags)
                tags_key = ",".join(sorted_tags)
            else:
                tags_key = str(tags)
            
            group_key = f"{memory_type}|{tags_key}"
            
            if group_key not in tag_groups:
                tag_groups[group_key] = []
            
            tag_groups[group_key].append(memory)
        
        # Split large groups based on time proximity
        final_groups = []
        for group in tag_groups.values():
            if len(group) <= self.max_memories_per_summary:
                final_groups.append(group)
            else:
                # Sort by timestamp
                group.sort(key=lambda m: m.get("timestamp", 0))
                
                # Split into smaller groups
                for i in range(0, len(group), self.max_memories_per_summary):
                    final_groups.append(group[i:i+self.max_memories_per_summary])
        
        # Merge small groups if possible
        merged_groups = []
        current_group = []
        
        # Sort groups by size (ascending)
        final_groups.sort(key=len)
        
        for group in final_groups:
            if len(current_group) + len(group) <= self.max_memories_per_summary:
                current_group.extend(group)
            else:
                if current_group:
                    merged_groups.append(current_group)
                current_group = group
        
        if current_group:
            merged_groups.append(current_group)
        
        logger.info(f"Grouped {len(memories)} memories into {len(merged_groups)} groups")
        
        # Log group sizes
        group_sizes = [len(g) for g in merged_groups]
        logger.info(f"Group sizes: min={min(group_sizes)}, max={max(group_sizes)}, avg={sum(group_sizes)/len(group_sizes):.1f}")
        
        return merged_groups
    
    async def summarize_memory_group(self, memory_group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Summarize a group of memories.
        
        Args:
            memory_group: Group of memories to summarize
            
        Returns:
            Summary memory
        """
        if not memory_group:
            return None
        
        try:
            # Extract all content
            contents = [memory["content"] for memory in memory_group]
            
            # Collect all tags
            all_tags = set()
            for memory in memory_group:
                tags = memory.get("tags", [])
                if isinstance(tags, list):
                    all_tags.update(tags)
            
            # Get the most common memory_type
            memory_types = {}
            for memory in memory_group:
                memory_type = memory.get("memory_type", "")
                memory_types[memory_type] = memory_types.get(memory_type, 0) + 1
            
            most_common_type = max(memory_types.items(), key=lambda x: x[1])[0] if memory_types else ""
            
            # Get timestamp range
            timestamps = [memory.get("timestamp", 0) for memory in memory_group]
            start_timestamp = min(timestamps) if timestamps else 0
            end_timestamp = max(timestamps) if timestamps else 0
            
            # Source memory IDs
            source_memories = [memory.get("content_hash", memory.get("id", "")) for memory in memory_group]
            
            # Generate summary content
            if OPENAI_AVAILABLE and self.openai_api_key:
                summary_content = await self._generate_ai_summary(contents, list(all_tags), most_common_type)
            else:
                # Simple concatenation with truncation
                summary_content = self._generate_simple_summary(contents)
            
            # Generate a hash for the summary
            summary_hash = generate_content_hash(summary_content, {
                "type": "summary",
                "source_count": len(memory_group),
                "tags": list(all_tags)
            })
            
            # Create summary memory
            summary = {
                "id": summary_hash,
                "content": summary_content,
                "content_hash": summary_hash,
                "memory_type": "summary",
                "tags": list(all_tags),
                "metadata": {
                    "summary": True,
                    "source_count": len(memory_group),
                    "source_types": list(memory_types.keys())
                },
                "source_memories": source_memories,
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp
            }
            
            logger.info(f"Created summary for {len(memory_group)} memories: {summary_hash}")
            return summary
        except Exception as e:
            logger.error(f"Failed to summarize memory group: {e}")
            return None
    
    async def _generate_ai_summary(self, contents: List[str], tags: List[str], memory_type: str) -> str:
        """
        Generate a summary using OpenAI API.
        
        Args:
            contents: List of contents to summarize
            tags: List of tags
            memory_type: Memory type
            
        Returns:
            Generated summary
        """
        try:
            # Prepare the input for the API
            combined_content = "\n---\n".join(contents)
            
            # Truncate if too long
            if len(combined_content) > 16000:  # OpenAI token limit safety
                combined_content = combined_content[:16000] + "..."
            
            # Create prompt
            prompt = f"""Summarize the following related memory entries into a single comprehensive summary.
            These memories are of type: {memory_type}
            Tags: {', '.join(tags)}
            
            MEMORIES:
            {combined_content}
            
            Create a detailed summary that preserves the key information while condensing multiple related points.
            The summary should be comprehensive enough that the original memories could be deleted without significant loss of information.
            """
            
            # Call OpenAI API
            response = await openai.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are an expert memory summarizer that creates detailed, comprehensive summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            logger.error(f"Failed to generate AI summary: {e}")
            # Fall back to simple summarization
            return self._generate_simple_summary(contents)
    
    def _generate_simple_summary(self, contents: List[str]) -> str:
        """
        Generate a simple summary by concatenating and truncating contents.
        
        Args:
            contents: List of contents to summarize
            
        Returns:
            Simple summary
        """
        # Add a header
        header = f"[Summary of {len(contents)} related memories]\n\n"
        
        # Concatenate contents with separators
        summary = header
        for i, content in enumerate(contents):
            # Truncate long contents
            if len(content) > 500:
                content = content[:497] + "..."
            
            # Add to summary with a separator
            summary += f"Memory {i+1}: {content}\n---\n"
            
            # Check if we've reached the maximum summary length
            if len(summary) > self.max_summary_length - 100:
                summary += f"\n[Truncated - {len(contents) - (i+1)} more memories not shown]"
                break
        
        return summary
    
    async def store_summary(self, summary: Dict[str, Any]) -> bool:
        """
        Store a memory summary.
        
        Args:
            summary: Summary to store
            
        Returns:
            True if storing was successful
        """
        if not self._is_initialized:
            await self.initialize()
        
        try:
            # Check if we need to generate an embedding
            if "embedding" not in summary or not summary["embedding"]:
                # Get embedding from the vector store
                embedding = await self._generate_embedding(summary["content"])
                summary["embedding"] = embedding
            
            # Store in memory_summaries table
            async with self.neon_client.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO memory_summaries(
                        id, content, embedding, metadata, tags,
                        source_memories, start_timestamp, end_timestamp
                    )
                    VALUES($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT(id) DO NOTHING
                """, 
                summary["id"],
                summary["content"],
                summary["embedding"],
                json.dumps(summary["metadata"]) if isinstance(summary["metadata"], dict) else summary["metadata"],
                json.dumps(summary["tags"]) if isinstance(summary["tags"], list) else summary["tags"],
                json.dumps(summary["source_memories"]),
                summary["start_timestamp"],
                summary["end_timestamp"]
                )
            
            # Also store in the main vector store for retrieval
            await self.vector_store.upsert(
                id=summary["id"],
                content=summary["content"],
                embedding=summary["embedding"],
                metadata={
                    "content_hash": summary["content_hash"],
                    "memory_type": summary["memory_type"],
                    "tags": summary["tags"],
                    "timestamp": summary["end_timestamp"],  # Use the latest timestamp
                    **summary["metadata"]
                }
            )
            
            logger.info(f"Stored summary: {summary['id']}")
            return True
        except Exception as e:
            logger.error(f"Failed to store summary: {e}")
            return False
    
    async def _generate_embedding(self, content: str) -> List[float]:
        """
        Generate an embedding for the given content.
        
        Args:
            content: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            # Try to use the model from the vector store
            if hasattr(self.vector_store, "model") and self.vector_store.model:
                return self.vector_store.model.encode(content).tolist()
            
            # Fallback to OpenAI embeddings if available
            if OPENAI_AVAILABLE and self.openai_api_key:
                response = await openai.embeddings.create(
                    input=content,
                    model="text-embedding-ada-002"
                )
                return response.data[0].embedding
            
            # If all else fails, return a zero vector
            logger.warning("No embedding model available, using zeros")
            return [0.0] * 1536  # Default embedding size
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return [0.0] * 1536
    
    async def delete_original_memories(self, source_memories: List[str]) -> int:
        """
        Delete original memories that have been summarized.
        
        Args:
            source_memories: List of memory IDs to delete
            
        Returns:
            Number of memories deleted
        """
        if not self._is_initialized:
            await self.initialize()
        
        if not source_memories:
            return 0
        
        try:
            # Collect payload URLs for blob deletion
            async with self.neon_client.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT content_hash, payload_url FROM memories
                    WHERE content_hash = ANY($1::text[]) AND payload_url IS NOT NULL
                """, source_memories)
                
                payload_urls = [row["payload_url"] for row in rows if row["payload_url"]]
            
            # Delete memories in batches
            total_deleted = 0
            for i in range(0, len(source_memories), self.deletion_batch_size):
                batch = source_memories[i:i+self.deletion_batch_size]
                
                # Delete from vector store (will handle both Qdrant and Neon)
                for memory_id in batch:
                    await self.vector_store.delete(memory_id)
                
                total_deleted += len(batch)
            
            # Delete blobs in one batch
            if payload_urls:
                await self.blob_store.batch_delete_blobs(payload_urls)
                logger.info(f"Deleted {len(payload_urls)} blobs from storage")
            
            logger.info(f"Deleted {total_deleted} original memories after summarization")
            return total_deleted
        except Exception as e:
            logger.error(f"Failed to delete original memories: {e}")
            return 0
    
    async def delete_old_memories(self, days: int) -> int:
        """
        Delete memories older than the specified retention period.
        
        Args:
            days: Retention period in days
            
        Returns:
            Number of memories deleted
        """
        if not self._is_initialized:
            await self.initialize()
        
        try:
            # Calculate the timestamp threshold
            threshold_time = datetime.now(timezone.utc) - timedelta(days=days)
            threshold_timestamp = int(threshold_time.timestamp())
            
            # Get memories older than retention period
            async with self.neon_client.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT content_hash, payload_url FROM memories
                    WHERE timestamp < $1
                    LIMIT $2
                """, threshold_timestamp, self.deletion_batch_size)
                
                if not rows:
                    logger.info(f"No memories older than {days} days found")
                    return 0
                
                memory_ids = [row["content_hash"] for row in rows]
                payload_urls = [row["payload_url"] for row in rows if row["payload_url"]]
                
                # Delete from Neon (vector store deletion will be handled by neon_client)
                await conn.execute("""
                    DELETE FROM memories WHERE content_hash = ANY($1::text[])
                """, memory_ids)
                
                # Delete from vector store (Qdrant)
                if self.vector_store.use_qdrant and self.vector_store.qdrant_client:
                    try:
                        import qdrant_client
                        
                        # Delete from Qdrant in batches
                        for i in range(0, len(memory_ids), 100):
                            batch = memory_ids[i:i+100]
                            
                            self.vector_store.qdrant_client.delete(
                                collection_name=self.vector_store.collection_name,
                                points_selector=qdrant_client.http.models.PointIdsList(
                                    points=batch
                                )
                            )
                    except Exception as e:
                        logger.error(f"Failed to delete from Qdrant: {e}")
                
                # Delete blobs if any
                if payload_urls:
                    await self.blob_store.batch_delete_blobs(payload_urls)
                    logger.info(f"Deleted {len(payload_urls)} blobs from storage")
                
                deleted_count = len(memory_ids)
                logger.info(f"Deleted {deleted_count} memories older than {days} days")
                return deleted_count
        except Exception as e:
            logger.error(f"Failed to delete old memories: {e}")
            return 0
    
    async def run_summarization(self, threshold_days: Optional[int] = None, dry_run: bool = False) -> Dict[str, Any]:
        """
        Run the summarization process.
        
        Args:
            threshold_days: Number of days to look back (overrides environment variable)
            dry_run: If True, don't actually store summaries or delete memories
            
        Returns:
            Summary statistics
        """
        if threshold_days is None:
            threshold_days = self.summary_threshold_days
        
        stats = {
            "old_memories_count": 0,
            "groups_count": 0,
            "summaries_created": 0,
            "memories_deleted": 0,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "errors": []
        }
        
        if not self._is_initialized:
            await self.initialize()
        
        try:
            # Get old memories
            memories = await self.get_old_memories(threshold_days)
            stats["old_memories_count"] = len(memories)
            
            if not memories:
                logger.info(f"No memories older than {threshold_days} days found")
                stats["completed_at"] = datetime.now(timezone.utc).isoformat()
                return stats
            
            # Group memories by similarity
            memory_groups = await self.group_memories_by_similarity(memories)
            stats["groups_count"] = len(memory_groups)
            
            # Process each group
            for i, group in enumerate(memory_groups):
                logger.info(f"Processing group {i+1}/{len(memory_groups)} with {len(group)} memories")
                
                # Summarize the group
                summary = await self.summarize_memory_group(group)
                
                if summary:
                    # Store the summary unless dry_run is True
                    if not dry_run:
                        success = await self.store_summary(summary)
                        
                        if success:
                            stats["summaries_created"] += 1
                            
                            # Delete original memories
                            deleted = await self.delete_original_memories(summary["source_memories"])
                            stats["memories_deleted"] += deleted
                    else:
                        logger.info(f"[DRY RUN] Would store summary {summary['id']} and delete {len(group)} original memories")
                        stats["summaries_created"] += 1
                        stats["memories_deleted"] += len(group)
            
            # Delete memories beyond retention period
            if not dry_run and self.retention_days > 0:
                # Calculate effective retention days (beyond threshold)
                effective_retention = self.retention_days - threshold_days
                
                if effective_retention > 0:
                    logger.info(f"Looking for memories older than {self.retention_days} days for deletion")
                    
                    # Delete in batches until no more found
                    total_deleted = 0
                    batch_deleted = 1  # Initialize to enter loop
                    
                    while batch_deleted > 0:
                        batch_deleted = await self.delete_old_memories(self.retention_days)
                        total_deleted += batch_deleted
                        
                        if batch_deleted > 0:
                            logger.info(f"Deleted batch of {batch_deleted} memories beyond retention period")
                    
                    if total_deleted > 0:
                        logger.info(f"Deleted total of {total_deleted} memories beyond retention period of {self.retention_days} days")
                        stats["memories_deleted"] += total_deleted
        except Exception as e:
            error_msg = f"Error during summarization: {str(e)}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
        finally:
            # Close connections
            await self.vector_store.close()
            await self.neon_client.close()
            
            # Update stats
            stats["completed_at"] = datetime.now(timezone.utc).isoformat()
        
        return stats

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Summarize old memory events")
    parser.add_argument("--days", type=int, help="Threshold in days (overrides environment variable)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (don't store summaries or delete memories)")
    args = parser.parse_args()
    
    logger.info(f"Starting memory summarization{' (DRY RUN)' if args.dry_run else ''}")
    
    summarizer = MemorySummarizer()
    
    try:
        # Run summarization
        stats = await summarizer.run_summarization(
            threshold_days=args.days,
            dry_run=args.dry_run
        )
        
        # Log statistics
        logger.info("Summarization completed:")
        logger.info(f"- Processed {stats['old_memories_count']} old memories")
        logger.info(f"- Created {stats['groups_count']} memory groups")
        logger.info(f"- Generated {stats['summaries_created']} summaries")
        logger.info(f"- Deleted {stats['memories_deleted']} original memories")
        
        # Return success
        return 0
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)