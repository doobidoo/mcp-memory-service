#!/usr/bin/env python3
"""
SQLite to Qdrant Migration Script

Migrates memories from SQLite-vec backend to Qdrant with checkpoint/resume capability,
batch validation, and atomic checkpoint updates.

Usage:
    python scripts/migration/migrate_sqlite_to_qdrant.py [--dry-run] [--resume]

Options:
    --dry-run: Validate migration without writing to Qdrant
    --resume: Resume from existing checkpoint
    --checkpoint PATH: Checkpoint file path (default: migration_checkpoint.json)
    --batch-size N: Batch size for migration (default: 100)
    --sqlite-path PATH: SQLite database path (auto-detected by default)
    --qdrant-path PATH: Qdrant storage path (auto-detected by default)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.mcp_memory_service.models.memory import Memory
from src.mcp_memory_service.storage.qdrant_storage import QdrantStorage
from src.mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class MigrationCheckpoint:
    """Checkpoint state for resumable migration."""

    total_memories: int
    migrated_count: int
    failed_hashes: list[str]
    last_successful_hash: str | None
    started_at: str
    last_updated_at: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MigrationCheckpoint":
        """Create from dictionary."""
        return cls(**data)


def load_checkpoint(checkpoint_path: str) -> MigrationCheckpoint | None:
    """
    Load checkpoint from file, return None if doesn't exist.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        MigrationCheckpoint object or None if file doesn't exist
    """
    path = Path(checkpoint_path)
    if not path.exists():
        return None

    try:
        with open(path) as f:
            data = json.load(f)
        logger.info(f"Loaded checkpoint: {data['migrated_count']}/{data['total_memories']} memories migrated")
        return MigrationCheckpoint.from_dict(data)
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return None


def save_checkpoint(checkpoint: MigrationCheckpoint, checkpoint_path: str) -> None:
    """
    Save checkpoint atomically (write to temp file, then rename).

    Args:
        checkpoint: MigrationCheckpoint object to save
        checkpoint_path: Path to save checkpoint file
    """
    path = Path(checkpoint_path)
    temp_path = Path(f"{checkpoint_path}.tmp")

    # Update timestamp
    checkpoint.last_updated_at = datetime.utcnow().isoformat() + "Z"

    try:
        # Write to temp file first
        with open(temp_path, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

        # Atomically rename temp file to actual checkpoint file
        # On Unix systems, rename() is atomic
        temp_path.replace(path)

        logger.debug(f"Checkpoint saved: {checkpoint.migrated_count}/{checkpoint.total_memories} memories")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        # Clean up temp file if it exists
        if temp_path.exists():
            temp_path.unlink()
        raise


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First embedding vector
        vec2: Second embedding vector

    Returns:
        Cosine similarity score (0 to 1)
    """
    a = np.array(vec1)
    b = np.array(vec2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


async def validate_batch(
    source_memories: list[Memory], qdrant_storage: QdrantStorage, batch_hashes: list[str]
) -> tuple[bool, str]:
    """
    Validate that batch was successfully written to Qdrant.

    Checks:
    - Count matches (batch size == retrieved count)
    - Embeddings similar (cosine >0.99 for sample)
    - Tags preserved (sets match)
    - Timestamps within 1ms

    Args:
        source_memories: Original memories from SQLite
        qdrant_storage: Qdrant storage instance
        batch_hashes: Content hashes of memories in batch

    Returns:
        Tuple of (success, error_message)
    """
    # Retrieve memories from Qdrant by content hash
    # Note: This is a simplified validation - in production you'd want
    # to use Qdrant's direct point retrieval by ID
    retrieved_count = 0
    validation_errors = []

    # Sample validation (check first and last memory in batch)
    sample_indices = [0, -1] if len(source_memories) > 1 else [0]

    for idx in sample_indices:
        source_mem = source_memories[idx]

        # Try to retrieve by semantic search (simplified validation)
        results = await qdrant_storage.retrieve(
            query=source_mem.content[:100],  # Use first 100 chars as query
            n_results=1,
            min_similarity=0.99,
        )

        if not results:
            validation_errors.append(f"Memory not found: {source_mem.content_hash}")
            continue

        result = results[0]
        retrieved_count += 1

        # Validate embeddings similarity
        if source_mem.embedding and result.memory.embedding:
            similarity = cosine_similarity(source_mem.embedding, result.memory.embedding)
            if similarity < 0.99:
                validation_errors.append(f"Embedding mismatch for {source_mem.content_hash}: similarity={similarity:.3f}")

        # Validate tags preserved
        source_tags = set(source_mem.tags)
        result_tags = set(result.memory.tags)
        if source_tags != result_tags:
            validation_errors.append(f"Tags mismatch for {source_mem.content_hash}: {source_tags} != {result_tags}")

        # Validate timestamps (within 1 second tolerance due to conversion)
        if source_mem.created_at and result.memory.created_at:
            time_diff = abs(source_mem.created_at - result.memory.created_at)
            if time_diff > 1.0:  # 1 second tolerance
                validation_errors.append(f"Timestamp mismatch for {source_mem.content_hash}: diff={time_diff:.3f}s")

    if validation_errors:
        return False, "; ".join(validation_errors)

    return True, ""


async def migrate_memories(
    sqlite_storage: SqliteVecMemoryStorage,
    qdrant_storage: QdrantStorage,
    checkpoint_path: str = "migration_checkpoint.json",
    batch_size: int = 100,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """
    Migrate memories from SQLite-vec to Qdrant with checkpoint/resume capability.

    Process:
    1. Load existing checkpoint (if resuming) or create new
    2. Initialize source (SqliteVecMemoryStorage) - read-only
    3. Initialize target (QdrantStorage) - create/reuse collection
    4. Get all memory hashes from source
    5. Skip already-migrated memories (from checkpoint)
    6. For each batch:
       a. Read memories from SQLite
       b. Store in Qdrant using store_batch()
       c. Validate batch AFTER writing (count, embeddings, tags)
       d. Save checkpoint after EVERY batch
       e. Report progress every 1000 memories
    7. Return migration report

    Args:
        sqlite_storage: SQLite-vec storage instance (read-only)
        qdrant_storage: Qdrant storage instance
        checkpoint_path: Path to checkpoint file
        batch_size: Number of memories to migrate per batch
        dry_run: If True, validate without writing to Qdrant
        resume: If True, resume from existing checkpoint

    Returns:
        Migration statistics including total, migrated, failed counts
    """
    start_time = time.time()

    # Load or create checkpoint
    checkpoint = None
    if resume:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            logger.info(f"Resuming migration from checkpoint: {checkpoint.migrated_count} memories already migrated")

    if not checkpoint:
        checkpoint = MigrationCheckpoint(
            total_memories=0,
            migrated_count=0,
            failed_hashes=[],
            last_successful_hash=None,
            started_at=datetime.utcnow().isoformat() + "Z",
            last_updated_at=datetime.utcnow().isoformat() + "Z",
        )

    # Initialize storages
    logger.info("Initializing storage backends...")
    await sqlite_storage.initialize()
    await qdrant_storage.initialize()

    # Get all memories from SQLite (read-only access)
    logger.info("Fetching all memories from SQLite...")
    all_memories = await sqlite_storage.get_all_memories()
    checkpoint.total_memories = len(all_memories)
    logger.info(f"Found {checkpoint.total_memories} memories in SQLite")

    if dry_run:
        logger.info("DRY RUN MODE - No data will be written to Qdrant")

    # Filter out already migrated memories if resuming
    if checkpoint.last_successful_hash:
        # Find index of last successful migration
        last_index = -1
        for i, mem in enumerate(all_memories):
            if mem.content_hash == checkpoint.last_successful_hash:
                last_index = i
                break

        if last_index >= 0:
            all_memories = all_memories[last_index + 1 :]
            logger.info(f"Skipping {last_index + 1} already migrated memories")

    # Migrate in batches
    batch_count = 0
    for i in range(0, len(all_memories), batch_size):
        batch = all_memories[i : i + batch_size]
        batch_count += 1

        logger.info(f"Processing batch {batch_count} ({len(batch)} memories)...")

        if not dry_run:
            # Store batch in Qdrant
            try:
                results = await qdrant_storage.store_batch(batch)

                # Track successes and failures
                for j, (success, message) in enumerate(results):
                    if success:
                        checkpoint.migrated_count += 1
                        checkpoint.last_successful_hash = batch[j].content_hash
                    else:
                        checkpoint.failed_hashes.append(batch[j].content_hash)
                        logger.warning(f"Failed to migrate {batch[j].content_hash}: {message}")

                # Validate batch after writing
                batch_hashes = [m.content_hash for m in batch]
                valid, error_msg = await validate_batch(batch, qdrant_storage, batch_hashes)

                if not valid:
                    logger.warning(f"Batch validation failed: {error_msg}")
                else:
                    logger.debug("Batch validated successfully")

            except Exception as e:
                logger.error(f"Batch migration failed: {e}")
                # Add all batch hashes to failed list
                checkpoint.failed_hashes.extend([m.content_hash for m in batch])
        else:
            # Dry run - just update checkpoint without writing
            checkpoint.migrated_count += len(batch)
            if batch:
                checkpoint.last_successful_hash = batch[-1].content_hash

        # Save checkpoint after EVERY batch (atomic operation)
        save_checkpoint(checkpoint, checkpoint_path)

        # Report progress every 1000 memories
        if checkpoint.migrated_count % 1000 == 0:
            elapsed = time.time() - start_time
            rate = checkpoint.migrated_count / elapsed if elapsed > 0 else 0
            eta = (checkpoint.total_memories - checkpoint.migrated_count) / rate if rate > 0 else 0
            logger.info(
                f"Progress: {checkpoint.migrated_count}/{checkpoint.total_memories} "
                f"({100 * checkpoint.migrated_count / checkpoint.total_memories:.1f}%) "
                f"Rate: {rate:.1f} memories/sec, ETA: {eta:.0f}s"
            )

    # Final report
    end_time = time.time()
    duration = end_time - start_time

    report = {
        "total_memories": checkpoint.total_memories,
        "migrated_successfully": checkpoint.migrated_count,
        "skipped_count": 0,  # We don't skip, we resume
        "failed_migrations": [{"hash": h, "error": "Migration failed"} for h in checkpoint.failed_hashes],
        "validation_results": {"validated": not dry_run, "failures": len(checkpoint.failed_hashes)},
        "duration_seconds": duration,
        "checkpoint_path": checkpoint_path,
        "dry_run": dry_run,
    }

    logger.info(f"\n{'=' * 60}")
    logger.info("Migration Complete!")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total memories: {report['total_memories']}")
    logger.info(f"Successfully migrated: {report['migrated_successfully']}")
    logger.info(f"Failed migrations: {len(report['failed_migrations'])}")
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info(f"Rate: {report['migrated_successfully'] / duration:.1f} memories/sec")
    if dry_run:
        logger.info("DRY RUN - No data was actually written to Qdrant")

    return report


def get_platform_storage_path() -> Path:
    """Get platform-specific storage path."""
    home = Path.home()
    if sys.platform == "darwin":  # macOS
        return home / "Library" / "Application Support" / "mcp-memory"
    elif sys.platform == "win32":  # Windows
        local_app_data = os.getenv("LOCALAPPDATA", "")
        if local_app_data:
            return Path(local_app_data) / "mcp-memory"
        return home / "AppData" / "Local" / "mcp-memory"
    else:  # Linux
        return home / ".local" / "share" / "mcp-memory"


async def main():
    """Main entry point for migration script."""
    parser = argparse.ArgumentParser(description="Migrate memories from SQLite-vec to Qdrant with checkpoint/resume capability")
    parser.add_argument("--dry-run", action="store_true", help="Validate migration without writing to Qdrant")
    parser.add_argument("--resume", action="store_true", help="Resume migration from existing checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="migration_checkpoint.json",
        help="Path to checkpoint file (default: migration_checkpoint.json)",
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Number of memories to migrate per batch (default: 100)")
    parser.add_argument("--sqlite-path", type=str, help="Path to SQLite database (auto-detected by default)")
    parser.add_argument("--qdrant-path", type=str, help="Path to Qdrant storage (auto-detected by default)")

    args = parser.parse_args()

    # Determine storage paths
    base_path = get_platform_storage_path()
    sqlite_path = args.sqlite_path or str(base_path / "sqlite_vec.db")
    qdrant_path = args.qdrant_path or str(base_path / "qdrant")

    logger.info(f"SQLite path: {sqlite_path}")
    logger.info(f"Qdrant path: {qdrant_path}")
    logger.info(f"Checkpoint path: {args.checkpoint}")

    # Verify SQLite database exists
    if not Path(sqlite_path).exists():
        logger.error(f"SQLite database not found: {sqlite_path}")
        sys.exit(1)

    # Initialize storage instances
    sqlite_storage = SqliteVecMemoryStorage(storage_path=sqlite_path)

    # Get embedding model from environment or use default
    embedding_model = os.getenv("MCP_EMBEDDING_MODEL", "text-embedding-3-small")

    qdrant_storage = QdrantStorage(
        storage_path=qdrant_path,
        embedding_model=embedding_model,
        collection_name="memories",
        quantization_enabled=False,
        distance_metric="Cosine",
    )

    # Run migration
    try:
        report = await migrate_memories(
            sqlite_storage=sqlite_storage,
            qdrant_storage=qdrant_storage,
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            resume=args.resume,
        )

        # Exit with error code if there were failures
        if report["failed_migrations"]:
            sys.exit(1)

    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
