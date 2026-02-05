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
Backup script to export all memories from the database to a JSON file.
This provides a safe backup before running migrations or making database changes.
"""

import asyncio
import datetime
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
logger = logging.getLogger("memory_backup")


async def backup_memories():
    """
    Export all memories from the database to a JSON file.
    """
    logger.info(f"Initializing ChromaDB storage at {CHROMA_PATH}")
    storage = ChromaMemoryStorage(CHROMA_PATH)

    # Create backups directory if it doesn't exist
    os.makedirs(BACKUPS_PATH, exist_ok=True)

    # Generate backup filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(BACKUPS_PATH, f"memory_backup_{timestamp}.json")

    logger.info(f"Creating backup at {backup_file}")

    try:
        # Retrieve all memories from the database
        try:
            # First try with embeddings
            logger.info("Attempting to retrieve memories with embeddings")
            results = storage.collection.get(include=["metadatas", "documents"])
            include_embeddings = False
        except Exception as e:
            logger.warning(f"Failed to retrieve with embeddings: {e}")
            logger.info("Falling back to retrieving memories without embeddings")
            # Fall back to no embeddings
            results = storage.collection.get(include=["metadatas", "documents"])
            include_embeddings = False

        if not results["ids"]:
            logger.info("No memories found in database")
            return backup_file

        total_memories = len(results["ids"])
        logger.info(f"Found {total_memories} memories to backup")

        # Create backup data structure
        backup_data = {"timestamp": datetime.datetime.now().isoformat(), "total_memories": total_memories, "memories": []}

        # Process each memory
        for i, memory_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            document = results["documents"][i]
            embedding = None
            if include_embeddings and "embeddings" in results and results["embeddings"] is not None:
                if i < len(results["embeddings"]):
                    embedding = results["embeddings"][i]

            memory_data = {"id": memory_id, "document": document, "metadata": metadata, "embedding": embedding}

            backup_data["memories"].append(memory_data)

        # Write backup to file
        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully backed up {total_memories} memories to {backup_file}")
        return backup_file

    except Exception as e:
        logger.error(f"Error creating backup: {str(e)}")
        raise


async def main():
    """Main function to run the backup."""
    logger.info("=== Starting memory backup ===")

    try:
        backup_file = await backup_memories()
        logger.info(f"=== Backup completed successfully: {backup_file} ===")
    except Exception as e:
        logger.error(f"Backup failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
