#!/usr/bin/env python3
"""
Migration script to fix existing SQLite-vec databases with embedding issues.

This script:
1. Backs up the existing database
2. Extracts all memories from the old database
3. Creates a new database with proper schema
4. Re-generates embeddings for all memories
5. Restores all memories with correct embeddings
"""

import asyncio
import json
import logging
import os
import shutil
import sqlite3
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mcp_memory_service.models.memory import Memory
from src.mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from src.mcp_memory_service.utils.hashing import generate_content_hash

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SqliteVecMigration:
    """Migrate existing SQLite-vec database to fix embedding issues."""

    def __init__(self, db_path: str):
        self.original_db_path = db_path
        self.backup_path = f"{db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.temp_db_path = f"{db_path}.temp"
        self.memories_recovered = []

    async def migrate(self):
        """Perform the migration."""
        print("\n" + "=" * 60)
        print("SQLite-vec Embedding Migration")
        print("=" * 60 + "\n")

        try:
            # Step 1: Backup original database
            self.backup_database()

            # Step 2: Extract memories from original database
            self.extract_memories()

            # Step 3: Create new database with correct schema
            await self.create_new_database()

            # Step 4: Restore memories with regenerated embeddings
            await self.restore_memories()

            # Step 5: Replace old database with new one
            self.finalize_migration()

            print("\n✅ Migration completed successfully!")
            print(f"   Migrated {len(self.memories_recovered)} memories")
            print(f"   Backup saved at: {self.backup_path}")

        except Exception as e:
            print(f"\n❌ Migration failed: {str(e)}")
            print("   Original database unchanged")
            print(f"   Backup available at: {self.backup_path}" if os.path.exists(self.backup_path) else "")

            # Cleanup temp database if exists
            if os.path.exists(self.temp_db_path):
                os.remove(self.temp_db_path)

            raise

    def backup_database(self):
        """Create a backup of the original database."""
        print("Step 1: Creating backup...")

        if not os.path.exists(self.original_db_path):
            raise FileNotFoundError(f"Database not found: {self.original_db_path}")

        shutil.copy2(self.original_db_path, self.backup_path)
        print(f"   ✓ Backup created: {self.backup_path}")

    def extract_memories(self):
        """Extract all memories from the original database."""
        print("\nStep 2: Extracting memories from original database...")

        conn = sqlite3.connect(self.original_db_path)
        cursor = conn.cursor()

        try:
            # Check if memories table exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='memories'
            """)
            if not cursor.fetchone():
                raise ValueError("No 'memories' table found in database")

            # Extract all memories
            cursor.execute("""
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, created_at_iso, updated_at_iso
                FROM memories
                ORDER BY created_at
            """)

            rows = cursor.fetchall()
            print(f"   ✓ Found {len(rows)} memories")

            for row in rows:
                try:
                    content_hash, content, tags_str, memory_type, metadata_str = row[:5]
                    created_at, updated_at, created_at_iso, updated_at_iso = row[5:]

                    # Parse tags and metadata
                    tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
                    metadata = json.loads(metadata_str) if metadata_str else {}

                    # Create Memory object
                    memory = Memory(
                        content=content,
                        content_hash=content_hash,
                        tags=tags,
                        memory_type=memory_type or "general",
                        metadata=metadata,
                        created_at=created_at,
                        updated_at=updated_at,
                        created_at_iso=created_at_iso,
                        updated_at_iso=updated_at_iso,
                    )

                    self.memories_recovered.append(memory)

                except Exception as e:
                    logger.warning(f"Failed to parse memory: {e}")
                    # Try to at least save the content
                    if row[1]:  # content
                        try:
                            memory = Memory(
                                content=row[1], content_hash=generate_content_hash(row[1]), tags=[], memory_type="general"
                            )
                            self.memories_recovered.append(memory)
                        except:
                            logger.error(f"Could not recover memory with content: {row[1][:50]}...")

        finally:
            conn.close()

        print(f"   ✓ Successfully recovered {len(self.memories_recovered)} memories")

    async def create_new_database(self):
        """Create a new database with proper schema."""
        print("\nStep 3: Creating new database with correct schema...")

        # Remove temp database if it exists
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

        # Create new storage instance
        self.new_storage = SqliteVecMemoryStorage(self.temp_db_path)

        # Initialize will create the database with correct schema
        await self.new_storage.initialize()

        print(f"   ✓ New database created with embedding dimension: {self.new_storage.embedding_dimension}")

    async def restore_memories(self):
        """Restore all memories with regenerated embeddings."""
        print("\nStep 4: Restoring memories with new embeddings...")

        if not self.new_storage:
            raise RuntimeError("New storage not initialized")

        successful = 0
        failed = 0

        # Show progress
        total = len(self.memories_recovered)

        for i, memory in enumerate(self.memories_recovered):
            try:
                # Update content hash to ensure it's correct
                if not memory.content_hash:
                    memory.content_hash = generate_content_hash(memory.content)

                # Store memory (this will generate new embeddings)
                success, message = await self.new_storage.store(memory)

                if success:
                    successful += 1
                else:
                    # If duplicate, that's okay
                    if "Duplicate" in message:
                        successful += 1
                    else:
                        failed += 1
                        logger.warning(f"Failed to store memory: {message}")

                # Show progress every 10%
                if (i + 1) % max(1, total // 10) == 0:
                    print(f"   ... {i + 1}/{total} memories processed ({(i + 1) * 100 // total}%)")

            except Exception as e:
                failed += 1
                logger.error(f"Error storing memory {memory.content_hash}: {e}")

        print(f"   ✓ Restored {successful} memories successfully")
        if failed > 0:
            print(f"   ⚠ Failed to restore {failed} memories")

    def finalize_migration(self):
        """Replace old database with new one."""
        print("\nStep 5: Finalizing migration...")

        # Close connections
        if hasattr(self, "new_storage") and self.new_storage.conn:
            self.new_storage.conn.close()

        # Move original to .old (just in case)
        old_path = f"{self.original_db_path}.old"
        if os.path.exists(old_path):
            os.remove(old_path)
        os.rename(self.original_db_path, old_path)

        # Move temp to original
        os.rename(self.temp_db_path, self.original_db_path)

        # Remove .old file
        os.remove(old_path)

        print("   ✓ Database migration completed")


async def main():
    """Run the migration."""
    if len(sys.argv) < 2:
        print("Usage: python migrate_sqlite_vec_embeddings.py <database_path>")
        print("\nExample:")
        print("  python migrate_sqlite_vec_embeddings.py ~/.mcp_memory/sqlite_vec.db")
        sys.exit(1)

    db_path = sys.argv[1]

    if not os.path.exists(db_path):
        print(f"Error: Database file not found: {db_path}")
        sys.exit(1)

    # Confirm with user
    print(f"This will migrate the database: {db_path}")
    print("A backup will be created before any changes are made.")
    response = input("\nContinue? (y/N): ").strip().lower()

    if response != "y":
        print("Migration cancelled.")
        sys.exit(0)

    # Run migration
    migration = SqliteVecMigration(db_path)
    await migration.migrate()


if __name__ == "__main__":
    asyncio.run(main())
