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

# scripts/migrate_tags.py
# python scripts/validate_memories.py --db-path /path/to/your/chroma_db

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from mcp_memory_service.storage.chroma import ChromaMemoryStorage

logger = logging.getLogger(__name__)


async def analyze_tag_formats(metadatas):
    """Analyze the current tag formats in the database"""
    formats = {"json_string": 0, "raw_list": 0, "comma_string": 0, "empty": 0, "invalid": 0}

    for meta in metadatas:
        tags = meta.get("tags")
        if tags is None:
            formats["empty"] += 1
            continue

        if isinstance(tags, list):
            formats["raw_list"] += 1
        elif isinstance(tags, str):
            try:
                parsed = json.loads(tags)
                if isinstance(parsed, list):
                    formats["json_string"] += 1
                else:
                    formats["invalid"] += 1
            except json.JSONDecodeError:
                if "," in tags:
                    formats["comma_string"] += 1
                else:
                    formats["invalid"] += 1
        else:
            formats["invalid"] += 1

    return formats


async def find_invalid_tags(metadatas):
    """Find any invalid tag formats"""
    invalid_entries = []

    for i, meta in enumerate(metadatas):
        tags = meta.get("tags")
        if tags is None:
            continue

        try:
            if isinstance(tags, str):
                json.loads(tags)
        except json.JSONDecodeError:
            invalid_entries.append({"memory_id": meta.get("content_hash", f"index_{i}"), "tags": tags})

    return invalid_entries


async def backup_memories(storage):
    """Create a backup of all memories"""
    results = storage.collection.get(include=["metadatas", "documents"])

    backup_data = {
        "timestamp": datetime.now().isoformat(),
        "memories": [
            {"id": results["ids"][i], "content": results["documents"][i], "metadata": results["metadatas"][i]}
            for i in range(len(results["ids"]))
        ],
    }

    backup_path = Path("backups")
    backup_path.mkdir(exist_ok=True)

    backup_file = backup_path / f"memory_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(backup_file, "w") as f:
        json.dump(backup_data, f)

    return backup_file


async def validate_current_state(storage):
    """Validate the current state of the database"""
    results = storage.collection.get(include=["metadatas"])
    return {
        "total_memories": len(results["ids"]),
        "tag_formats": await analyze_tag_formats(results["metadatas"]),
        "invalid_tags": await find_invalid_tags(results["metadatas"]),
    }


async def migrate_tags(storage):
    """Perform the tag migration"""
    results = storage.collection.get(include=["metadatas", "documents"])

    migrated_count = 0
    error_count = 0

    for i, meta in enumerate(results["metadatas"]):
        try:
            # Extract current tags
            current_tags = meta.get("tags", "[]")

            # Normalize to list format
            if isinstance(current_tags, str):
                try:
                    # Try parsing as JSON first
                    tags = json.loads(current_tags)
                    if isinstance(tags, str):
                        tags = [t.strip() for t in tags.split(",")]
                    elif isinstance(tags, list):
                        tags = [str(t).strip() for t in tags]
                    else:
                        tags = []
                except json.JSONDecodeError:
                    # Handle as comma-separated string
                    tags = [t.strip() for t in current_tags.split(",")]
            elif isinstance(current_tags, list):
                tags = [str(t).strip() for t in current_tags]
            else:
                tags = []

            # Update with normalized format
            new_meta = meta.copy()
            new_meta["tags"] = json.dumps(tags)

            # Update memory
            storage.collection.update(ids=[results["ids"][i]], metadatas=[new_meta])

            migrated_count += 1

        except Exception as e:
            error_count += 1
            logger.error(f"Error migrating memory {results['ids'][i]}: {str(e)}")

    return migrated_count, error_count


async def verify_migration(storage):
    """Verify the migration was successful"""
    results = storage.collection.get(include=["metadatas"])

    verification = {
        "total_memories": len(results["ids"]),
        "tag_formats": await analyze_tag_formats(results["metadatas"]),
        "invalid_tags": await find_invalid_tags(results["metadatas"]),
    }

    return verification


async def rollback_migration(storage, backup_file):
    """Rollback to the backup if needed"""
    with open(backup_file) as f:
        backup = json.load(f)

    for memory in backup["memories"]:
        storage.collection.update(ids=[memory["id"]], metadatas=[memory["metadata"]], documents=[memory["content"]])


async def main():
    # Configure logging
    log_level = os.getenv("LOG_LEVEL", "ERROR").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.ERROR),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    # Initialize storage
    # storage = ChromaMemoryStorage("path/to/your/db")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Validate memory data tags")
    parser.add_argument("--db-path", required=True, help="Path to ChromaDB database")
    args = parser.parse_args()

    # Initialize storage with provided path
    logger.info(f"Connecting to database at: {args.db_path}")
    storage = ChromaMemoryStorage(args.db_path)

    # 1. Create backup
    logger.info("Creating backup...")
    backup_file = await backup_memories(storage)
    logger.info(f"Backup created at: {backup_file}")

    # 2. Validate current state
    logger.info("Validating current state...")
    current_state = await validate_current_state(storage)
    logger.info("\nCurrent state:")
    logger.info(json.dumps(current_state, indent=2))

    # 3. Confirm migration
    proceed = input("\nProceed with migration? (yes/no): ")
    if proceed.lower() == "yes":
        # 4. Run migration
        logger.info("Running migration...")
        migrated_count, error_count = await migrate_tags(storage)
        logger.info(f"Migration completed. Migrated: {migrated_count}, Errors: {error_count}")

        # 5. Verify migration
        logger.info("Verifying migration...")
        verification = await verify_migration(storage)
        logger.info("\nMigration verification:")
        logger.info(json.dumps(verification, indent=2))

        # 6. Check if rollback needed
        if error_count > 0:
            rollback = input("\nErrors detected. Rollback to backup? (yes/no): ")
            if rollback.lower() == "yes":
                logger.info("Rolling back...")
                await rollback_migration(storage, backup_file)
                logger.info("Rollback completed")
    else:
        logger.info("Migration cancelled")


if __name__ == "__main__":
    asyncio.run(main())
