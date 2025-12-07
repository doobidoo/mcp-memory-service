#!/usr/bin/env python3
"""
Retroactively apply association-based quality boosts to all existing memories.

This script runs the consolidation's relevance calculation on all memories
to apply quality boosts based on connection counts, without running the full
consolidation pipeline (no archiving, no compression, just quality updates).

Usage:
    python scripts/maintenance/apply_quality_boost_retroactively.py [--dry-run] [--batch-size 500]
"""

import asyncio
import argparse
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp_memory_service.storage.factory import create_storage_instance
from mcp_memory_service.consolidation.decay import ExponentialDecayCalculator
from mcp_memory_service.consolidation.base import ConsolidationConfig
from mcp_memory_service.config import CONSOLIDATION_CONFIG, SQLITE_VEC_PATH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def get_connection_counts(storage, memories):
    """Calculate connection counts for all memories."""
    logger.info(f"Calculating connection counts for {len(memories)} memories...")

    # Build connection graph from associations stored as memories
    connections = {}

    # Get all association memories
    all_memories = await storage.get_all_memories()
    association_memories = [m for m in all_memories if m.memory_type == 'association']

    logger.info(f"Found {len(association_memories)} existing associations")

    # Count connections
    for assoc in association_memories:
        # Association metadata contains source_memory_hashes
        source_hashes = assoc.metadata.get('source_memory_hashes', [])
        for hash_val in source_hashes:
            connections[hash_val] = connections.get(hash_val, 0) + 1

    return connections


async def apply_quality_boosts(dry_run=False, batch_size=500):
    """Apply quality boosts to all memories based on connections."""

    logger.info("=" * 80)
    logger.info("Association-Based Quality Boost - Retroactive Application")
    logger.info("=" * 80)
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    logger.info(f"Batch size: {batch_size}")
    logger.info("")

    # Initialize storage
    logger.info("Initializing storage backend...")
    storage = await create_storage_instance(SQLITE_VEC_PATH, server_type='script')

    try:
        # Get all memories
        logger.info("Retrieving all memories...")
        all_memories = await storage.get_all_memories()
        logger.info(f"Found {len(all_memories)} total memories")

        # Calculate connection counts
        connections = await get_connection_counts(storage, all_memories)
        logger.info(f"Connection graph built: {len(connections)} memories have connections")

        # Initialize decay calculator
        config_obj = ConsolidationConfig(**CONSOLIDATION_CONFIG)
        decay_calculator = ExponentialDecayCalculator(config_obj)

        # Process in batches
        total_boosted = 0
        total_processed = 0

        for i in range(0, len(all_memories), batch_size):
            batch = all_memories[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(all_memories) + batch_size - 1) // batch_size

            logger.info("")
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} memories)...")

            # Calculate relevance scores (includes quality boost)
            relevance_scores = await decay_calculator.process(
                batch,
                connections=connections,
                access_patterns={}
            )

            # Count boosts applied in this batch
            batch_boosted = sum(
                1 for score in relevance_scores
                if score.metadata.get('association_boost_applied', False)
            )

            # Update memories with boosted quality scores
            if not dry_run:
                for memory, score in zip(batch, relevance_scores):
                    if score.metadata.get('association_boost_applied', False):
                        updated_memory = await decay_calculator.update_memory_relevance_metadata(memory, score)
                        await storage.update_memory_metadata(
                            updated_memory.content_hash,
                            updated_memory.metadata,
                            preserve_timestamps=True
                        )

            total_boosted += batch_boosted
            total_processed += len(batch)

            logger.info(f"  Batch {batch_num}: {batch_boosted} memories received quality boost")
            logger.info(f"  Progress: {total_processed}/{len(all_memories)} memories processed ({total_boosted} boosted)")

        # Summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total memories processed: {total_processed}")
        logger.info(f"Total memories boosted: {total_boosted}")
        logger.info(f"Boost rate: {(total_boosted / total_processed * 100):.1f}%")

        if dry_run:
            logger.info("")
            logger.info("DRY RUN MODE - No changes were written to storage")
            logger.info("Run without --dry-run to apply changes")
        else:
            logger.info("")
            logger.info("✅ Quality boosts applied successfully!")
            logger.info("")
            logger.info("Verification:")
            logger.info("  Check logs: grep 'Association quality boost' ~/.local/share/mcp-memory/logs/*.log")
            logger.info("  Check distribution: curl http://127.0.0.1:8000/api/quality/distribution")

    finally:
        # Cleanup
        if hasattr(storage, 'close'):
            await storage.close()


async def main():
    parser = argparse.ArgumentParser(
        description="Retroactively apply association-based quality boosts"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Preview changes without applying them"
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=500,
        help="Number of memories to process per batch (default: 500)"
    )

    args = parser.parse_args()

    try:
        await apply_quality_boosts(
            dry_run=args.dry_run,
            batch_size=args.batch_size
        )
        return 0
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
