#!/usr/bin/env python3
"""
Regenerate embeddings for all memories after cosine distance migration.

This script regenerates embeddings for all existing memories in the database.
Useful after migrations that drop the embeddings table but preserve memories.

Usage:
    python scripts/maintenance/regenerate_embeddings.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.mcp_memory_service.config import SQLITE_VEC_PATH
from src.mcp_memory_service.storage.factory import create_storage_instance

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def regenerate_embeddings():
    """Regenerate embeddings for all memories."""

    database_path = SQLITE_VEC_PATH
    logger.info(f"Using database: {database_path}")

    # Create storage instance
    logger.info("Initializing storage backend...")
    storage = await create_storage_instance(database_path)

    try:
        # Get all memories (this accesses the memories table, not embeddings)
        logger.info("Fetching all memories from database...")

        # Access the primary storage directly for hybrid backend
        if hasattr(storage, "primary"):
            actual_storage = storage.primary
        else:
            actual_storage = storage

        # Get count first
        if hasattr(actual_storage, "conn"):
            cursor = actual_storage.conn.execute("SELECT COUNT(*) FROM memories")
            total_count = cursor.fetchone()[0]
            logger.info(f"Found {total_count} memories to process")

            # Get all memories
            cursor = actual_storage.conn.execute("""
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, created_at_iso, updated_at_iso
                FROM memories
            """)

            memories = []
            for row in cursor.fetchall():
                content_hash, content, tags_str, memory_type, metadata_str = row[:5]
                created_at, updated_at, created_at_iso, updated_at_iso = row[5:]

                # Parse tags
                tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []

                # Parse metadata
                import json

                metadata = json.loads(metadata_str) if metadata_str else {}

                memories.append(
                    {
                        "content_hash": content_hash,
                        "content": content,
                        "tags": tags,
                        "memory_type": memory_type,
                        "metadata": metadata,
                        "created_at": created_at,
                        "updated_at": updated_at,
                        "created_at_iso": created_at_iso,
                        "updated_at_iso": updated_at_iso,
                    }
                )

            logger.info(f"Loaded {len(memories)} memories")

            # Regenerate embeddings
            logger.info("Regenerating embeddings...")
            success_count = 0
            error_count = 0

            for i, mem in enumerate(memories, 1):
                try:
                    # Generate embedding
                    embedding = actual_storage._generate_embedding(mem["content"])

                    # Get the rowid for this memory
                    cursor = actual_storage.conn.execute(
                        "SELECT id FROM memories WHERE content_hash = ?", (mem["content_hash"],)
                    )
                    result = cursor.fetchone()
                    if not result:
                        logger.warning(f"Memory {mem['content_hash'][:8]} not found, skipping")
                        error_count += 1
                        continue

                    memory_id = result[0]

                    # Insert embedding
                    from src.mcp_memory_service.storage.sqlite_vec import serialize_float32

                    actual_storage.conn.execute(
                        "INSERT OR REPLACE INTO memory_embeddings(rowid, content_embedding) VALUES (?, ?)",
                        (memory_id, serialize_float32(embedding)),
                    )

                    success_count += 1

                    if i % 10 == 0:
                        logger.info(f"Progress: {i}/{len(memories)} ({(i / len(memories) * 100):.1f}%)")
                        actual_storage.conn.commit()

                except Exception as e:
                    logger.error(f"Error processing memory {mem['content_hash'][:8]}: {e}")
                    error_count += 1
                    continue

            # Final commit
            actual_storage.conn.commit()

            logger.info(f"\n{'=' * 60}")
            logger.info("Regeneration complete!")
            logger.info(f"  ✅ Success: {success_count} embeddings")
            logger.info(f"  ❌ Errors: {error_count}")
            logger.info(f"  📊 Total: {len(memories)} memories")
            logger.info(f"{'=' * 60}\n")

            # Verify
            cursor = actual_storage.conn.execute("SELECT COUNT(*) FROM memory_embeddings")
            embedding_count = cursor.fetchone()[0]
            logger.info(f"Verification: {embedding_count} embeddings in database")

        else:
            logger.error("Storage backend doesn't support direct database access")
            return False

        return True

    finally:
        # Cleanup
        if hasattr(storage, "close"):
            await storage.close()


if __name__ == "__main__":
    try:
        result = asyncio.run(regenerate_embeddings())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
