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
Storage Migration Tool for MCP Memory Service

This script helps migrate memory data between different storage backends
(ChromaDB and sqlite-vec).

Usage:
    python scripts/migrate_storage.py --from chroma --to sqlite-vec
    python scripts/migrate_storage.py --from sqlite-vec --to chroma --backup
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.base import MemoryStorage
from mcp_memory_service.storage.chroma import ChromaMemoryStorage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MigrationTool:
    """Tool for migrating memory data between storage backends."""

    def __init__(self):
        self.source_storage = None
        self.target_storage = None

    async def export_memories(self, storage: MemoryStorage) -> list[dict[str, Any]]:
        """Export all memories from a storage backend."""
        logger.info("Exporting memories from source storage...")

        exported_memories = []

        try:
            # For ChromaDB, we need to get all memories via a broad search
            if hasattr(storage, "collection") and storage.collection:
                # Get all memories from ChromaDB
                results = storage.collection.get()

                if results and results.get("ids"):
                    for i, memory_id in enumerate(results["ids"]):
                        try:
                            metadata = results["metadatas"][i] if results.get("metadatas") else {}
                            document = results["documents"][i] if results.get("documents") else ""
                            embedding = results["embeddings"][i] if results.get("embeddings") else None

                            # Convert metadata to memory format
                            memory_data = {
                                "content": document,
                                "content_hash": metadata.get("content_hash", ""),
                                "tags": metadata.get("tags_str", "").split(",") if metadata.get("tags_str") else [],
                                "memory_type": metadata.get("type"),
                                "metadata": {
                                    k: v
                                    for k, v in metadata.items()
                                    if k
                                    not in [
                                        "content_hash",
                                        "tags_str",
                                        "type",
                                        "timestamp",
                                        "timestamp_float",
                                        "timestamp_str",
                                        "created_at",
                                        "created_at_iso",
                                        "updated_at",
                                        "updated_at_iso",
                                    ]
                                },
                                "embedding": embedding,
                                "created_at": metadata.get("created_at"),
                                "created_at_iso": metadata.get("created_at_iso"),
                                "updated_at": metadata.get("updated_at"),
                                "updated_at_iso": metadata.get("updated_at_iso"),
                            }

                            exported_memories.append(memory_data)

                        except Exception as e:
                            logger.warning(f"Failed to export memory {memory_id}: {e}")
                            continue

            elif hasattr(storage, "conn") and storage.conn:
                # Get all memories from SQLite-vec
                cursor = storage.conn.execute("""
                    SELECT content_hash, content, tags, memory_type, metadata,
                           created_at, updated_at, created_at_iso, updated_at_iso
                    FROM memories
                    ORDER BY created_at
                """)

                for row in cursor.fetchall():
                    try:
                        content_hash, content, tags_str, memory_type, metadata_str = row[:5]
                        created_at, updated_at, created_at_iso, updated_at_iso = row[5:]

                        # Parse tags and metadata
                        tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
                        metadata = json.loads(metadata_str) if metadata_str else {}

                        memory_data = {
                            "content": content,
                            "content_hash": content_hash,
                            "tags": tags,
                            "memory_type": memory_type,
                            "metadata": metadata,
                            "embedding": None,  # Will be regenerated on import
                            "created_at": created_at,
                            "created_at_iso": created_at_iso,
                            "updated_at": updated_at,
                            "updated_at_iso": updated_at_iso,
                        }

                        exported_memories.append(memory_data)

                    except Exception as e:
                        logger.warning(f"Failed to export memory: {e}")
                        continue

            logger.info(f"Exported {len(exported_memories)} memories")
            return exported_memories

        except Exception as e:
            logger.error(f"Failed to export memories: {e}")
            raise

    async def import_memories(self, storage: MemoryStorage, memories: list[dict[str, Any]]) -> int:
        """Import memories into a storage backend."""
        logger.info(f"Importing {len(memories)} memories to target storage...")

        imported_count = 0
        failed_count = 0

        for memory_data in memories:
            try:
                # Create Memory object
                memory = Memory(
                    content=memory_data["content"],
                    content_hash=memory_data["content_hash"],
                    tags=memory_data.get("tags", []),
                    memory_type=memory_data.get("memory_type"),
                    metadata=memory_data.get("metadata", {}),
                    embedding=memory_data.get("embedding"),
                    created_at=memory_data.get("created_at"),
                    created_at_iso=memory_data.get("created_at_iso"),
                    updated_at=memory_data.get("updated_at"),
                    updated_at_iso=memory_data.get("updated_at_iso"),
                )

                # Store the memory
                success, message = await storage.store(memory)

                if success:
                    imported_count += 1
                    if imported_count % 100 == 0:
                        logger.info(f"Imported {imported_count} memories...")
                else:
                    failed_count += 1
                    logger.warning(f"Failed to import memory {memory_data['content_hash']}: {message}")

            except Exception as e:
                failed_count += 1
                logger.warning(f"Failed to import memory: {e}")
                continue

        logger.info(f"Import complete: {imported_count} successful, {failed_count} failed")
        return imported_count

    async def create_backup(self, memories: list[dict[str, Any]], backup_path: str) -> str:
        """Create a JSON backup of exported memories."""
        backup_data = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "total_memories": len(memories),
            "memories": memories,
        }

        os.makedirs(os.path.dirname(backup_path), exist_ok=True)

        with open(backup_path, "w") as f:
            json.dump(backup_data, f, indent=2)

        logger.info(f"Created backup at: {backup_path}")
        return backup_path

    async def load_backup(self, backup_path: str) -> list[dict[str, Any]]:
        """Load memories from a JSON backup file."""
        with open(backup_path) as f:
            backup_data = json.load(f)

        memories = backup_data.get("memories", [])
        logger.info(f"Loaded {len(memories)} memories from backup: {backup_path}")
        return memories

    async def migrate(
        self,
        from_backend: str,
        to_backend: str,
        source_path: str,
        target_path: str,
        create_backup: bool = False,
        backup_path: str = None,
    ) -> bool:
        """Perform migration between storage backends."""
        try:
            logger.info(f"Starting migration from {from_backend} to {to_backend}")

            # Initialize source storage
            if from_backend == "chroma":
                self.source_storage = ChromaMemoryStorage(source_path)
            elif from_backend == "sqlite_vec":
                self.source_storage = SqliteVecMemoryStorage(source_path)
            else:
                raise ValueError(f"Unsupported source backend: {from_backend}")

            await self.source_storage.initialize()
            logger.info(f"Initialized source storage ({from_backend})")

            # Export memories
            memories = await self.export_memories(self.source_storage)

            if not memories:
                logger.warning("No memories found in source storage")
                return False

            # Create backup if requested
            if create_backup:
                if not backup_path:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = f"memory_backup_{from_backend}_to_{to_backend}_{timestamp}.json"

                await self.create_backup(memories, backup_path)

            # Initialize target storage
            if to_backend == "chroma":
                self.target_storage = ChromaMemoryStorage(target_path)
            elif to_backend == "sqlite_vec":
                self.target_storage = SqliteVecMemoryStorage(target_path)
            else:
                raise ValueError(f"Unsupported target backend: {to_backend}")

            await self.target_storage.initialize()
            logger.info(f"Initialized target storage ({to_backend})")

            # Import memories
            imported_count = await self.import_memories(self.target_storage, memories)

            logger.info(f"Migration completed successfully: {imported_count} memories migrated")
            return True

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False

        finally:
            # Clean up connections
            if self.source_storage and hasattr(self.source_storage, "close"):
                self.source_storage.close()
            if self.target_storage and hasattr(self.target_storage, "close"):
                self.target_storage.close()


async def main():
    """Main entry point for the migration tool."""
    parser = argparse.ArgumentParser(
        description="Migrate memory data between storage backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate from ChromaDB to sqlite-vec
  python scripts/migrate_storage.py --from chroma --to sqlite_vec \\
    --source-path /path/to/chroma_db --target-path /path/to/sqlite_vec.db

  # Migrate with backup
  python scripts/migrate_storage.py --from chroma --to sqlite_vec \\
    --source-path /path/to/chroma_db --target-path /path/to/sqlite_vec.db \\
    --backup --backup-path backup.json

  # Restore from backup
  python scripts/migrate_storage.py --restore backup.json \\
    --to sqlite_vec --target-path /path/to/sqlite_vec.db
        """,
    )

    parser.add_argument("--from", dest="from_backend", choices=["chroma", "sqlite_vec"], help="Source storage backend")
    parser.add_argument(
        "--to", dest="to_backend", choices=["chroma", "sqlite_vec"], required=True, help="Target storage backend"
    )
    parser.add_argument("--source-path", help="Path to source storage")
    parser.add_argument("--target-path", required=True, help="Path to target storage")
    parser.add_argument("--backup", action="store_true", help="Create backup before migration")
    parser.add_argument("--backup-path", help="Custom backup file path")
    parser.add_argument("--restore", help="Restore from backup file instead of migrating")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without actually doing it")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if not args.restore and not args.from_backend:
        parser.error("--from is required unless using --restore")

    if not args.restore and not args.source_path:
        parser.error("--source-path is required unless using --restore")

    if args.from_backend == args.to_backend:
        parser.error("Source and target backends cannot be the same")

    migration_tool = MigrationTool()

    try:
        if args.restore:
            # Restore from backup
            logger.info(f"Restoring from backup: {args.restore}")

            if not os.path.exists(args.restore):
                logger.error(f"Backup file not found: {args.restore}")
                return 1

            memories = await migration_tool.load_backup(args.restore)

            if args.dry_run:
                logger.info(f"DRY RUN: Would restore {len(memories)} memories to {args.to_backend}")
                return 0

            # Initialize target storage
            if args.to_backend == "chroma":
                target_storage = ChromaMemoryStorage(args.target_path)
            else:
                target_storage = SqliteVecMemoryStorage(args.target_path)

            await target_storage.initialize()
            imported_count = await migration_tool.import_memories(target_storage, memories)

            if hasattr(target_storage, "close"):
                target_storage.close()

            logger.info(f"Restoration completed: {imported_count} memories restored")

        else:
            # Regular migration
            if args.dry_run:
                logger.info(f"DRY RUN: Would migrate from {args.from_backend} to {args.to_backend}")

                # Initialize source storage and count memories
                if args.from_backend == "chroma":
                    source_storage = ChromaMemoryStorage(args.source_path)
                else:
                    source_storage = SqliteVecMemoryStorage(args.source_path)

                await source_storage.initialize()
                memories = await migration_tool.export_memories(source_storage)

                if hasattr(source_storage, "close"):
                    source_storage.close()

                logger.info(f"DRY RUN: Found {len(memories)} memories to migrate")
                return 0

            # Perform actual migration
            success = await migration_tool.migrate(
                from_backend=args.from_backend,
                to_backend=args.to_backend,
                source_path=args.source_path,
                target_path=args.target_path,
                create_backup=args.backup,
                backup_path=args.backup_path,
            )

            if not success:
                logger.error("Migration failed")
                return 1

        logger.info("Operation completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
