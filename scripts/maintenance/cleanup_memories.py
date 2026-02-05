#!/usr/bin/env python3
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
Script to clean up erroneous memory entries from ChromaDB.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path so we can import from the src directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mcp_memory_service.config import CHROMA_PATH
from src.mcp_memory_service.storage.chroma import ChromaMemoryStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("memory_cleanup")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Clean up erroneous memory entries")
    parser.add_argument("--error-text", help="Text pattern found in erroneous entries", type=str)
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without actually deleting")
    parser.add_argument("--reset", action="store_true", help="Completely reset the database (use with caution!)")
    return parser.parse_args()


async def cleanup_memories(error_text=None, dry_run=False, reset=False):
    """
    Clean up erroneous memory entries from ChromaDB.

    Args:
        error_text: Text pattern found in erroneous entries
        dry_run: If True, only show what would be deleted without actually deleting
        reset: If True, completely reset the database
    """
    logger.info(f"Initializing ChromaDB storage at {CHROMA_PATH}")
    storage = ChromaMemoryStorage(CHROMA_PATH)

    if reset:
        if dry_run:
            logger.warning("[DRY RUN] Would reset the entire database")
        else:
            logger.warning("Resetting the entire database")
            try:
                storage.client.delete_collection("memory_collection")
                logger.info("Deleted existing collection")

                # Reinitialize collection
                storage.collection = storage.client.create_collection(
                    name="memory_collection", metadata={"hnsw:space": "cosine"}, embedding_function=storage.embedding_function
                )
                logger.info("Created new empty collection")
            except Exception as e:
                logger.error(f"Error resetting collection: {str(e)}")
        return

    # Get all memory entries
    try:
        # Query all entries
        result = storage.collection.get()
        total_memories = len(result["ids"]) if "ids" in result else 0
        logger.info(f"Found {total_memories} total memories in the database")

        if total_memories == 0:
            logger.info("No memories found in the database")
            return

        # Find erroneous entries
        error_ids = []

        if error_text:
            logger.info(f"Searching for entries containing text pattern: '{error_text}'")
            for i, doc in enumerate(result["documents"]):
                if error_text in doc:
                    error_ids.append(result["ids"][i])
                    if len(error_ids) <= 5:  # Show a few examples
                        logger.info(f"Found erroneous entry: {doc[:100]}...")

        # If no specific error text, look for common error patterns
        if not error_text and not error_ids:
            logger.info("No specific error text provided, looking for common error patterns")
            for i, doc in enumerate(result["documents"]):
                # Look for very short documents (likely errors)
                if len(doc.strip()) < 10:
                    error_ids.append(result["ids"][i])
                    logger.info(f"Found suspiciously short entry: '{doc}'")
                # Look for error messages
                elif any(err in doc.lower() for err in ["error", "exception", "failed", "invalid"]):
                    error_ids.append(result["ids"][i])
                    if len(error_ids) <= 5:  # Show a few examples
                        logger.info(f"Found likely error entry: {doc[:100]}...")

        if not error_ids:
            logger.info("No erroneous entries found")
            return

        logger.info(f"Found {len(error_ids)} erroneous entries")

        # Delete erroneous entries
        if dry_run:
            logger.info(f"[DRY RUN] Would delete {len(error_ids)} erroneous entries")
        else:
            logger.info(f"Deleting {len(error_ids)} erroneous entries")
            # Process in batches to avoid overwhelming the database
            batch_size = 100
            for i in range(0, len(error_ids), batch_size):
                batch = error_ids[i : i + batch_size]
                logger.info(f"Deleting batch {i // batch_size + 1}/{(len(error_ids) - 1) // batch_size + 1}")
                storage.collection.delete(ids=batch)

            logger.info("Deletion completed")

    except Exception as e:
        logger.error(f"Error cleaning up memories: {str(e)}")
        raise


async def main():
    """Main function to run the cleanup."""
    args = parse_args()

    logger.info("=== Starting memory cleanup ===")

    try:
        await cleanup_memories(args.error_text, args.dry_run, args.reset)
        logger.info("=== Cleanup completed successfully ===")
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
