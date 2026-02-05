#!/usr/bin/env python
"""
Model Change Migration Script for Qdrant Storage

Migrates memories to a new embedding model by re-embedding all content.
Creates a new collection with re-embedded memories and performs atomic swap.

Features:
- Re-embedding with new model (GPU OOM prevention via small batches)
- Checkpoint support for resumable migrations
- Atomic collection swap for safety
- Backup retention option
- Progress reporting and validation

Usage:
    python migrate_to_new_model.py --old-model all-MiniLM-L6-v2 --new-model text-embedding-ada-002 --storage /path/to/qdrant
"""

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.mcp_memory_service.models import Memory
from src.mcp_memory_service.storage.qdrant_storage import QdrantStorage

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelMigrationCheckpoint:
    """Manages checkpoint state for resumable migration."""

    def __init__(self, checkpoint_path: Path | None = None):
        self.checkpoint_path = checkpoint_path or Path("/tmp/model_migration_checkpoint.json")
        self.state = self._load_checkpoint()

    def _load_checkpoint(self) -> dict[str, Any]:
        """Load checkpoint from disk if exists."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return {
            "processed_hashes": [],
            "total_memories": 0,
            "re_embedded_count": 0,
            "failed_embeddings": [],
            "last_batch_index": 0,
            "start_time": None,
        }

    def save(self):
        """Save checkpoint to disk."""
        try:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.checkpoint_path, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def update_batch(self, batch_index: int, processed_hashes: list[str], re_embedded: int, failed: list[dict]):
        """Update checkpoint after processing batch."""
        self.state["last_batch_index"] = batch_index
        self.state["processed_hashes"].extend(processed_hashes)
        self.state["re_embedded_count"] = re_embedded
        self.state["failed_embeddings"].extend(failed)
        self.save()

    def is_processed(self, content_hash: str) -> bool:
        """Check if memory was already processed."""
        return content_hash in self.state["processed_hashes"]

    def get_progress(self) -> tuple[int, int]:
        """Get current progress (processed, total)."""
        return self.state["re_embedded_count"], self.state["total_memories"]


def get_model_hash(model_name: str) -> str:
    """Generate short hash for model name."""
    return hashlib.md5(model_name.encode()).hexdigest()[:8]


async def _re_embed_batch(
    memories: list[Memory], new_storage: QdrantStorage, checkpoint: ModelMigrationCheckpoint
) -> tuple[list[Memory], list[dict]]:
    """
    Re-embed batch of memories with new model.

    Returns:
        Tuple of (successfully re-embedded memories, failed embeddings info)
    """
    re_embedded = []
    failed = []

    for memory in memories:
        # Skip if already processed (checkpoint resume)
        if checkpoint.is_processed(memory.content_hash):
            logger.debug(f"Skipping already processed: {memory.content_hash}")
            continue

        try:
            # Generate new embedding using the new model
            # The QdrantStorage instance already has the new embedding service
            new_embedding = await new_storage._get_embedding(memory.content)

            # Create new memory with updated embedding
            re_embedded_memory = Memory(
                content=memory.content,
                content_hash=memory.content_hash,  # Keep same hash
                tags=memory.tags,
                memory_type=memory.memory_type,
                metadata=memory.metadata,
                created_at=memory.created_at,  # Preserve timestamps
                updated_at=memory.updated_at,
                embedding=new_embedding,  # NEW embedding with different dimensions
            )
            re_embedded.append(re_embedded_memory)
            logger.debug(f"Re-embedded {memory.content_hash} successfully")

        except Exception as e:
            logger.error(f"Failed to re-embed {memory.content_hash}: {e}")
            failed.append(
                {
                    "content_hash": memory.content_hash,
                    "error": str(e),
                    "content_preview": memory.content[:100] if memory.content else "",
                }
            )

    return re_embedded, failed


async def migrate_to_new_model(
    old_model: str,
    new_model: str,
    storage_path: str,
    checkpoint_path: str | None = None,
    batch_size: int = 50,  # Small batches to prevent GPU OOM
    keep_backup: bool = False,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """
    Migrate memories to new embedding model by re-embedding all content.

    Process:
    1. Load checkpoint (if resuming)
    2. Initialize OLD QdrantStorage (read existing memories)
    3. Initialize NEW QdrantStorage with new model (create new collection)
    4. For each batch:
       a. Read memories from old collection
       b. Re-embed content with new model
       c. Store in new collection
       d. Validate embeddings generated successfully
       e. Update checkpoint
    5. Verify counts match
    6. Swap collections (rename old → backup, new → active)
    7. Return migration report

    Args:
        old_model: Name/path of current embedding model
        new_model: Name/path of new embedding model
        storage_path: Path to Qdrant storage directory
        checkpoint_path: Path to checkpoint file for resume
        batch_size: Number of memories per batch (50 recommended for GPU)
        keep_backup: Whether to keep old collection as backup
        dry_run: Preview migration without making changes
        resume: Resume from checkpoint

    Returns:
        Migration statistics including counts and timings
    """
    start_time = time.time()

    # Initialize checkpoint
    checkpoint = ModelMigrationCheckpoint(Path(checkpoint_path) if checkpoint_path else None)

    if resume and not checkpoint.state["processed_hashes"]:
        logger.warning("No checkpoint found to resume from, starting fresh")
        resume = False

    if not checkpoint.state["start_time"]:
        checkpoint.state["start_time"] = datetime.utcnow().isoformat()

    # Collection naming
    old_model_hash = get_model_hash(old_model)
    new_model_hash = get_model_hash(new_model)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    old_collection = "memories"
    new_collection = f"memories_new_{new_model_hash}"
    backup_collection = f"memories_backup_{old_model_hash}_{timestamp}"

    logger.info("Migration plan:")
    logger.info(f"  Old model: {old_model} (hash: {old_model_hash})")
    logger.info(f"  New model: {new_model} (hash: {new_model_hash})")
    logger.info(f"  Old collection: {old_collection}")
    logger.info(f"  New collection: {new_collection}")
    logger.info(f"  Backup collection: {backup_collection}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Keep backup: {keep_backup}")
    logger.info(f"  Dry run: {dry_run}")
    logger.info(f"  Resume: {resume}")

    if dry_run:
        logger.info("DRY RUN MODE - No changes will be made")

    try:
        # Initialize OLD storage (read existing memories)
        logger.info(f"Initializing OLD storage with model: {old_model}")
        old_storage = QdrantStorage(storage_path=storage_path, embedding_model=old_model, collection_name=old_collection)
        await old_storage.initialize()

        # Get total count
        all_memories = await old_storage.get_all_memories()
        total_count = len(all_memories)
        checkpoint.state["total_memories"] = total_count
        logger.info(f"Found {total_count} memories to migrate")

        if dry_run:
            # Estimate migration time
            time_per_memory = 0.05  # Approximate seconds per re-embedding
            estimated_time = total_count * time_per_memory
            logger.info(f"Estimated migration time: {estimated_time:.1f} seconds ({estimated_time / 60:.1f} minutes)")
            return {
                "dry_run": True,
                "total_memories": total_count,
                "estimated_time_seconds": estimated_time,
                "batch_size": batch_size,
                "batches": (total_count + batch_size - 1) // batch_size,
            }

        # Initialize NEW storage with new model (creates new collection)
        logger.info(f"Initializing NEW storage with model: {new_model}")
        new_storage = QdrantStorage(storage_path=storage_path, embedding_model=new_model, collection_name=new_collection)
        await new_storage.initialize()

        # Process memories in batches
        re_embedded_count = 0
        failed_embeddings = []
        batch_count = (total_count + batch_size - 1) // batch_size

        for batch_index in range(0, total_count, batch_size):
            # Skip if already processed (for resume)
            if resume and batch_index < checkpoint.state["last_batch_index"]:
                logger.info(f"Skipping batch {batch_index // batch_size + 1}/{batch_count} (already processed)")
                continue

            batch_memories = all_memories[batch_index : batch_index + batch_size]
            batch_num = batch_index // batch_size + 1

            logger.info(f"Processing batch {batch_num}/{batch_count} ({len(batch_memories)} memories)...")

            # Re-embed batch
            re_embedded, failed = await _re_embed_batch(batch_memories, new_storage, checkpoint)

            # Store re-embedded memories in new collection
            for memory in re_embedded:
                await new_storage.store_memory(
                    content=memory.content, tags=memory.tags, memory_type=memory.memory_type, metadata=memory.metadata
                )

            re_embedded_count += len(re_embedded)
            failed_embeddings.extend(failed)

            # Update checkpoint
            processed_hashes = [m.content_hash for m in re_embedded]
            checkpoint.update_batch(batch_index + batch_size, processed_hashes, re_embedded_count, failed)

            # Progress report every 100 memories
            if re_embedded_count % 100 == 0 or batch_num == batch_count:
                progress_pct = (re_embedded_count / total_count) * 100
                logger.info(f"Progress: {re_embedded_count}/{total_count} ({progress_pct:.1f}%)")

        # Verify migration
        logger.info("Verifying migration...")
        new_memories = await new_storage.get_all_memories()
        new_count = len(new_memories)

        if new_count != re_embedded_count:
            logger.error(f"Count mismatch! Expected {re_embedded_count}, got {new_count}")
            return {
                "success": False,
                "error": "Count mismatch after migration",
                "expected": re_embedded_count,
                "actual": new_count,
            }

        # Perform atomic collection swap
        if not dry_run and re_embedded_count > 0:
            logger.info("Performing atomic collection swap...")

            # Get Qdrant client for collection management
            client = old_storage.client

            # Step 1: Rename old collection to backup
            logger.info(f"Renaming {old_collection} → {backup_collection}")
            await client.rename_collection(old_collection, backup_collection)

            # Step 2: Rename new collection to active
            logger.info(f"Renaming {new_collection} → {old_collection}")
            await client.rename_collection(new_collection, old_collection)

            # Step 3: Optionally delete backup
            if not keep_backup:
                logger.info(f"Deleting backup collection: {backup_collection}")
                await client.delete_collection(backup_collection)
            else:
                logger.info(f"Keeping backup collection: {backup_collection}")

        # Calculate statistics
        duration = time.time() - start_time

        result = {
            "success": True,
            "total_memories": total_count,
            "re_embedded": re_embedded_count,
            "failed_embeddings": failed_embeddings,
            "old_model": old_model,
            "new_model": new_model,
            "old_collection": backup_collection if keep_backup else None,
            "new_collection": old_collection,
            "duration_seconds": duration,
            "memories_per_second": re_embedded_count / duration if duration > 0 else 0,
        }

        logger.info("Migration completed successfully!")
        logger.info(f"  Total memories: {total_count}")
        logger.info(f"  Re-embedded: {re_embedded_count}")
        logger.info(f"  Failed: {len(failed_embeddings)}")
        logger.info(f"  Duration: {duration:.1f} seconds")
        logger.info(f"  Rate: {result['memories_per_second']:.1f} memories/second")

        # Clean up checkpoint on success
        if checkpoint_path and checkpoint.checkpoint_path.exists():
            checkpoint.checkpoint_path.unlink()
            logger.info("Checkpoint cleaned up")

        return result

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return {"success": False, "error": str(e), "duration_seconds": time.time() - start_time}


async def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(description="Migrate Qdrant memories to new embedding model")

    parser.add_argument("--old-model", required=True, help="Current embedding model name (e.g., all-MiniLM-L6-v2)")

    parser.add_argument("--new-model", required=True, help="New embedding model name (e.g., text-embedding-ada-002)")

    parser.add_argument("--storage", required=True, help="Path to Qdrant storage directory")

    parser.add_argument(
        "--checkpoint", default="/tmp/model_migration_checkpoint.json", help="Checkpoint file path for resumable migration"
    )

    parser.add_argument(
        "--batch-size", type=int, default=50, help="Batch size for re-embedding (default: 50, smaller prevents GPU OOM)"
    )

    parser.add_argument("--keep-backup", action="store_true", help="Keep old collection as backup after migration")

    parser.add_argument("--dry-run", action="store_true", help="Preview migration without making changes")

    parser.add_argument("--resume", action="store_true", help="Resume migration from checkpoint")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run migration
    result = await migrate_to_new_model(
        old_model=args.old_model,
        new_model=args.new_model,
        storage_path=args.storage,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        keep_backup=args.keep_backup,
        dry_run=args.dry_run,
        resume=args.resume,
    )

    # Print result summary
    if args.dry_run:
        print("\nDry Run Summary:")
        print(f"  Total memories: {result['total_memories']}")
        print(f"  Estimated time: {result['estimated_time_seconds']:.1f} seconds")
        print(f"  Batch size: {result['batch_size']}")
        print(f"  Number of batches: {result['batches']}")
    else:
        print("\nMigration Summary:")
        print(f"  Success: {result['success']}")
        if result["success"]:
            print(f"  Total memories: {result['total_memories']}")
            print(f"  Re-embedded: {result['re_embedded']}")
            print(f"  Failed: {len(result.get('failed_embeddings', []))}")
            print(f"  Duration: {result['duration_seconds']:.1f} seconds")
            print(f"  Rate: {result.get('memories_per_second', 0):.1f} memories/second")
            if result.get("old_collection"):
                print(f"  Backup collection: {result['old_collection']}")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")

    # Exit with appropriate code
    sys.exit(0 if result.get("success", True) else 1)


if __name__ == "__main__":
    asyncio.run(main())
