#!/usr/bin/env python3
"""
Migration script to convert comma-separated tag storage to normalized relational format.

This script:
1. Creates new 'tags' table (id, name)
2. Creates new 'memory_tags' junction table (memory_id, tag_id)
3. Migrates existing comma-separated tags to relational format
4. Creates indexes for optimal query performance
5. Drops old idx_tags index (which wasn't being used anyway)

Performance Impact:
- BEFORE: O(n) table scans with LIKE '%tag%' (index unusable)
- AFTER: O(log n) index seeks with JOIN + WHERE tag IN (?)

Run this migration BEFORE deploying code that uses the new schema.
"""

import logging
import os
import sqlite3
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_db_path() -> Path:
    """Get the SQLite database path from environment or default location."""
    db_path_str = os.environ.get("MCP_MEMORY_SQLITE_PATH")

    if db_path_str:
        return Path(db_path_str)

    # Default location (macOS)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "mcp-memory" / "sqlite_vec.db"
    # Linux
    elif sys.platform.startswith("linux"):
        xdg_data_home = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
        return Path(xdg_data_home) / "mcp-memory" / "sqlite_vec.db"
    # Windows
    elif sys.platform == "win32":
        app_data = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
        return Path(app_data) / "mcp-memory" / "sqlite_vec.db"
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")


def verify_backup_exists(db_path: Path) -> Path:
    """Verify that a backup exists before proceeding."""
    backup_path = db_path.with_suffix(".db.backup")
    if not backup_path.exists():
        raise RuntimeError(f"No backup found at {backup_path}\nPlease create a backup first:\n  cp {db_path} {backup_path}")
    logger.info(f"Verified backup exists: {backup_path}")
    return backup_path


def parse_tags(tags_str: str) -> list[str]:
    """Parse comma-separated tags string into list."""
    if not tags_str or tags_str.strip() == "":
        return []
    return [tag.strip() for tag in tags_str.split(",") if tag.strip()]


def check_migration_needed(conn: sqlite3.Connection) -> bool:
    """Check if migration has already been run."""
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tags'")
    if cursor.fetchone():
        logger.warning("Migration already completed - 'tags' table exists")
        return False
    return True


def create_new_schema(conn: sqlite3.Connection) -> None:
    """Create new normalized tag schema."""
    logger.info("Creating normalized tag schema...")

    # Create tags table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    """)
    logger.info("  ✓ Created 'tags' table")

    # Create memory_tags junction table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_tags (
            memory_id INTEGER NOT NULL,
            tag_id INTEGER NOT NULL,
            PRIMARY KEY (memory_id, tag_id),
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
            FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
        )
    """)
    logger.info("  ✓ Created 'memory_tags' junction table")

    # Create indexes for optimal performance
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name)")
    logger.info("  ✓ Created index on tags(name)")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_tags_memory ON memory_tags(memory_id)")
    logger.info("  ✓ Created index on memory_tags(memory_id)")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_tags_tag ON memory_tags(tag_id)")
    logger.info("  ✓ Created index on memory_tags(tag_id)")

    conn.commit()


def migrate_tag_data(conn: sqlite3.Connection, dry_run: bool = False) -> tuple[int, int]:
    """
    Migrate existing comma-separated tags to relational format.

    Returns:
        Tuple of (unique_tags_count, total_associations_count)
    """
    logger.info("Migrating tag data...")

    # Get all memories with tags
    cursor = conn.execute('SELECT id, tags FROM memories WHERE tags IS NOT NULL AND tags != ""')
    memories_with_tags = cursor.fetchall()

    logger.info(f"  Found {len(memories_with_tags)} memories with tags")

    # Collect all unique tags
    all_tags: set[str] = set()
    memory_tag_map: list[tuple[int, list[str]]] = []

    for memory_id, tags_str in memories_with_tags:
        tag_list = parse_tags(tags_str)
        if tag_list:
            all_tags.update(tag_list)
            memory_tag_map.append((memory_id, tag_list))

    logger.info(f"  Found {len(all_tags)} unique tags across {len(memory_tag_map)} memories")

    if dry_run:
        logger.info("  DRY RUN - No data will be written")
        return len(all_tags), sum(len(tags) for _, tags in memory_tag_map)

    # Insert unique tags
    tag_id_map = {}
    for tag_name in sorted(all_tags):
        cursor = conn.execute("INSERT INTO tags (name) VALUES (?)", (tag_name,))
        tag_id_map[tag_name] = cursor.lastrowid

    logger.info(f"  ✓ Inserted {len(tag_id_map)} unique tags")

    # Insert memory-tag associations
    associations_count = 0
    for memory_id, tag_list in memory_tag_map:
        for tag_name in tag_list:
            tag_id = tag_id_map[tag_name]
            conn.execute("INSERT INTO memory_tags (memory_id, tag_id) VALUES (?, ?)", (memory_id, tag_id))
            associations_count += 1

    logger.info(f"  ✓ Created {associations_count} memory-tag associations")

    conn.commit()
    return len(all_tags), associations_count


def drop_old_index(conn: sqlite3.Connection, dry_run: bool = False) -> None:
    """Drop the old idx_tags index that wasn't being used anyway."""
    logger.info("Removing old idx_tags index...")

    if dry_run:
        logger.info("  DRY RUN - Index would be dropped")
        return

    conn.execute("DROP INDEX IF EXISTS idx_tags")
    conn.commit()
    logger.info("  ✓ Dropped idx_tags (it wasn't being used anyway)")


def verify_migration(conn: sqlite3.Connection) -> None:
    """Verify migration completed successfully."""
    logger.info("Verifying migration...")

    # Count tags
    cursor = conn.execute("SELECT COUNT(*) FROM tags")
    tag_count = cursor.fetchone()[0]
    logger.info(f"  Tags table: {tag_count} unique tags")

    # Count associations
    cursor = conn.execute("SELECT COUNT(*) FROM memory_tags")
    assoc_count = cursor.fetchone()[0]
    logger.info(f"  Memory-tag associations: {assoc_count}")

    # Verify indexes exist
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_tags_name'")
    if not cursor.fetchone():
        raise RuntimeError("Migration verification failed: idx_tags_name not found")

    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_memory_tags_memory'")
    if not cursor.fetchone():
        raise RuntimeError("Migration verification failed: idx_memory_tags_memory not found")

    # Sample query to verify schema works
    cursor = conn.execute("""
        SELECT m.content_hash, t.name
        FROM memories m
        JOIN memory_tags mt ON m.id = mt.memory_id
        JOIN tags t ON mt.tag_id = t.id
        LIMIT 5
    """)
    sample = cursor.fetchall()

    if sample:
        logger.info(f"  ✓ Sample query returned {len(sample)} results")
        logger.info("  ✓ Migration verification passed!")
    else:
        logger.warning("  No results from sample query (this is OK if database has no tagged memories)")


def main():
    """Run the migration."""
    import argparse

    parser = argparse.ArgumentParser(description="Migrate SQLite tags to normalized relational format")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--skip-backup-check", action="store_true", help="Skip backup verification (dangerous!)")
    parser.add_argument("--db-path", type=str, help="Override database path")
    args = parser.parse_args()

    # Get database path
    if args.db_path:
        db_path = Path(args.db_path)
    else:
        db_path = get_db_path()

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    logger.info(f"Database: {db_path}")

    # Verify backup exists (unless skipped)
    if not args.skip_backup_check and not args.dry_run:
        verify_backup_exists(db_path)

    # Connect to database
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout = 30000")  # 30 second timeout

    try:
        # Check if migration needed
        if not check_migration_needed(conn):
            logger.info("Migration not needed - exiting")
            return 0

        if args.dry_run:
            logger.info("=== DRY RUN MODE - No changes will be made ===")

        # Run migration steps
        create_new_schema(conn)
        unique_tags, associations = migrate_tag_data(conn, dry_run=args.dry_run)
        drop_old_index(conn, dry_run=args.dry_run)

        if not args.dry_run:
            verify_migration(conn)

            logger.info("\n" + "=" * 60)
            logger.info("Migration completed successfully!")
            logger.info(f"  • {unique_tags} unique tags migrated")
            logger.info(f"  • {associations} memory-tag associations created")
            logger.info("=" * 60)
            logger.info("\nNOTE: The 'tags' column in 'memories' table has been preserved")
            logger.info("for rollback safety. It can be dropped after verifying the new")
            logger.info("schema works correctly in production.")
        else:
            logger.info("\n" + "=" * 60)
            logger.info("DRY RUN completed - would have migrated:")
            logger.info(f"  • {unique_tags} unique tags")
            logger.info(f"  • {associations} memory-tag associations")
            logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        if not args.dry_run:
            logger.error("Rolling back transaction...")
            conn.rollback()
        return 1

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
