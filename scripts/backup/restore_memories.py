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
Restoration script to import memories from a backup JSON file into the database.
This can be used to restore memories after a database issue or migration problem.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import from the src directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mcp_memory_service.config import BACKUPS_PATH, CHROMA_PATH
from src.mcp_memory_service.storage.chroma import ChromaMemoryStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("memory_restore")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Restore memories from backup file")
    parser.add_argument("backup_file", help="Path to backup JSON file", type=str)
    parser.add_argument("--reset", action="store_true", help="Reset database before restoration")
    return parser.parse_args()


async def restore_memories(backup_file, reset_db=False):
    """
    Import memories from a backup JSON file into the database.

    Args:
        backup_file: Path to the backup JSON file
        reset_db: If True, reset the database before restoration
    """
    logger.info(f"Initializing ChromaDB storage at {CHROMA_PATH}")
    storage = ChromaMemoryStorage(CHROMA_PATH)

    # Check if backup file exists
    if not os.path.exists(backup_file):
        # Check if it's a filename in the backups directory
        potential_path = os.path.join(BACKUPS_PATH, backup_file)
        if os.path.exists(potential_path):
            backup_file = potential_path
        else:
            raise FileNotFoundError(f"Backup file not found: {backup_file}")

    logger.info(f"Loading backup from {backup_file}")

    try:
        # Load backup data
        with open(backup_file, encoding="utf-8") as f:
            backup_data = json.load(f)

        backup_data.get("total_memories", 0)
        memories = backup_data.get("memories", [])

        if not memories:
            logger.warning("No memories found in backup file")
            return

        logger.info(f"Found {len(memories)} memories in backup file")

        # Reset database if requested
        if reset_db:
            logger.warning("Resetting database before restoration")
            try:
                storage.client.delete_collection("memory_collection")
                logger.info("Deleted existing collection")
            except Exception as e:
                logger.error(f"Error deleting collection: {str(e)}")

            # Reinitialize collection
            storage.collection = storage.client.create_collection(
                name="memory_collection", metadata={"hnsw:space": "cosine"}, embedding_function=storage.embedding_function
            )
            logger.info("Created new collection")

        # Process memories in batches
        batch_size = 50
        success_count = 0
        error_count = 0

        for i in range(0, len(memories), batch_size):
            batch = memories[i : i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(memories) - 1) // batch_size + 1}")

            # Prepare batch data
            batch_ids = []
            batch_documents = []
            batch_metadatas = []
            batch_embeddings = []

            for memory in batch:
                batch_ids.append(memory["id"])
                batch_documents.append(memory["document"])
                batch_metadatas.append(memory["metadata"])
                if memory.get("embedding") is not None:
                    batch_embeddings.append(memory["embedding"])

            try:
                # Use upsert to avoid duplicates
                if batch_embeddings and len(batch_embeddings) > 0 and len(batch_embeddings) == len(batch_ids):
                    storage.collection.upsert(
                        ids=batch_ids, documents=batch_documents, metadatas=batch_metadatas, embeddings=batch_embeddings
                    )
                else:
                    storage.collection.upsert(ids=batch_ids, documents=batch_documents, metadatas=batch_metadatas)
                success_count += len(batch)
            except Exception as e:
                logger.error(f"Error restoring batch: {str(e)}")
                error_count += len(batch)

        logger.info(f"Restoration completed: {success_count} memories restored, {error_count} errors")

    except Exception as e:
        logger.error(f"Error restoring backup: {str(e)}")
        raise


async def main():
    """Main function to run the restoration."""
    args = parse_args()

    logger.info("=== Starting memory restoration ===")

    try:
        await restore_memories(args.backup_file, args.reset)
        logger.info("=== Restoration completed successfully ===")
    except Exception as e:
        logger.error(f"Restoration failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
