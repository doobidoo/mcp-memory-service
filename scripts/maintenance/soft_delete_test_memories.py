#!/usr/bin/env python3
"""
MCP Memory Service - Soft Delete Test Memories Script

Comprehensive cleanup of test memories by:
- Test tags (from pytest scripts)
- Content patterns (test content, test memory, etc.)
- UUID patterns in content [xxxxxxxx-xxxx-...]
- Tag patterns (delete-*, all-tag-*, partial-*, etc.)

Works directly on SQLite database (no HTTP server required).
Uses soft-delete (sets deleted_at timestamp, 30-day retention).

Usage:
    python scripts/maintenance/soft_delete_test_memories.py [--dry-run] [--force]

Options:
    --dry-run   Show what would be deleted without actually deleting
    --force     Skip confirmation prompt
"""

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import List, Tuple


# Default database path
DEFAULT_DB_PATH = Path.home() / "AppData" / "Local" / "mcp-memory" / "sqlite_vec.db"

# Allowlist of safe tags for maintenance operations
# This prevents SQL injection via malicious tag values
ALLOWED_TAGS = {
    '__test__', '__perf__', '__integration__',
    'test', 'perf', 'integration', 'concurrent',
    'hybrid', 'sync-test', 'rapid', 'consistency',
    'health', 'token', 'multi', 'compat', 'demo', 'mixed',
    'test-data', 'showcase', 'test1',
    'unique-search-tag', 'before-date-tag', 'before-date-test',
    'timeframe-tag-filter', 'timeframe-delete', 'timeframe-test',
    'recall-test', 'quality-test', 'singletag',
    'tag1', 'tag2', 'tag3',
    'tag-a', 'tag-b', 'tag-c',
    'new-tag-1', 'new-tag-2',
    'delete-tag', 'all-tag-test', 'partial-tag',
}

def escape_glob_pattern(pattern: str) -> str:
    """Escape special GLOB characters for SQLite patterns."""
    # Escape GLOB metacharacters: [ ] * ?
    return (pattern
        .replace('[', '[[]')
        .replace(']', '[]]')
        .replace('*', '[*]')
        .replace('?', '[?]'))

# Test tags commonly used in pytest scripts
TEST_TAGS = [
    # High priority - direct test markers
    "test", "perf", "integration", "concurrent",
    "hybrid", "sync-test", "rapid", "consistency",
    "health", "token", "multi", "compat", "demo", "mixed",
    "test-data", "showcase", "test1",
    # Medium priority - specific test tags
    "unique-search-tag", "before-date-tag", "before-date-test",
    "timeframe-tag-filter", "timeframe-delete", "timeframe-test",
    "recall-test", "quality-test", "singletag",
    "tag1", "tag2", "tag3",
    "tag-a", "tag-b", "tag-c",
    "new-tag-1", "new-tag-2",
]

# Tag patterns (GLOB matching)
TAG_PATTERNS = [
    "*delete*",
    "*all-tag-*",
    "*delete-by-tag-*",
    "*partial-*",
]

# Content patterns indicating test data
CONTENT_PATTERNS = [
    "%test content%",
    "%test memory%",
    "%test for%",
    "%test hash%",
    "%backup test%",
]


def get_db_connection(db_path: Path) -> sqlite3.Connection:
    """Get database connection."""
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        sys.exit(1)
    return sqlite3.connect(str(db_path))


def count_by_tags(cursor: sqlite3.Cursor, tags: List[str]) -> int:
    """Count memories matching any of the given tags."""
    if not tags:
        return 0

    # Validate against allowlist
    safe_tags = [tag for tag in tags if tag in ALLOWED_TAGS]
    if not safe_tags:
        print(f"Warning: No allowed tags found in: {tags}")
        return 0

    conditions = []
    for tag in safe_tags:
        # Escape GLOB special characters
        escaped_tag = escape_glob_pattern(tag)
        conditions.append(f"(',' || REPLACE(tags, ' ', '') || ',') GLOB '*,{escaped_tag},*'")

    where = " OR ".join(conditions)
    cursor.execute(f"""
        SELECT COUNT(*) FROM memories
        WHERE deleted_at IS NULL AND ({where})
    """)
    return cursor.fetchone()[0]


def count_by_tag_patterns(cursor: sqlite3.Cursor, patterns: List[str]) -> int:
    """Count memories matching tag patterns (uses safe parameterized queries)."""
    if not patterns:
        return 0

    # Use parameterized queries instead of string interpolation
    conditions = []
    params = []
    for pattern in patterns:
        conditions.append("(',' || REPLACE(tags, ' ', '') || ',') GLOB ?")
        params.append(pattern)

    where = " OR ".join(conditions)
    query = f"""
        SELECT COUNT(*) FROM memories
        WHERE deleted_at IS NULL AND ({where})
    """
    cursor.execute(query, params)
    return cursor.fetchone()[0]


def count_by_content_patterns(cursor: sqlite3.Cursor, patterns: List[str]) -> int:
    """Count memories matching content patterns (uses parameterized queries)."""
    if not patterns:
        return 0

    # Use parameterized queries for LIKE patterns
    conditions = ["LOWER(content) LIKE ?" for _ in patterns]
    where = " OR ".join(conditions)
    query = f"""
        SELECT COUNT(*) FROM memories
        WHERE deleted_at IS NULL AND ({where})
    """
    cursor.execute(query, patterns)
    return cursor.fetchone()[0]


def count_uuid_patterns(cursor: sqlite3.Cursor) -> int:
    """Count memories with UUID patterns in content like [xxxxxxxx-xxxx-...]."""
    cursor.execute("""
        SELECT COUNT(*) FROM memories
        WHERE deleted_at IS NULL
        AND content GLOB '*[[][a-f0-9][a-f0-9][a-f0-9][a-f0-9][a-f0-9][a-f0-9][a-f0-9][a-f0-9]-*'
    """)
    return cursor.fetchone()[0]


def soft_delete_by_tags(cursor: sqlite3.Cursor, tags: List[str], deleted_at: float) -> int:
    """Soft-delete memories by tags (uses allowlist validation)."""
    if not tags:
        return 0

    # Validate against allowlist
    safe_tags = [tag for tag in tags if tag in ALLOWED_TAGS]
    if not safe_tags:
        print(f"Warning: No allowed tags found in: {tags}")
        return 0

    conditions = []
    for tag in safe_tags:
        # Escape GLOB special characters
        escaped_tag = escape_glob_pattern(tag)
        conditions.append(f"(',' || REPLACE(tags, ' ', '') || ',') GLOB '*,{escaped_tag},*'")

    where = " OR ".join(conditions)
    cursor.execute(f"""
        UPDATE memories
        SET deleted_at = ?
        WHERE deleted_at IS NULL AND ({where})
    """, (deleted_at,))
    return cursor.rowcount


def soft_delete_by_tag_patterns(cursor: sqlite3.Cursor, patterns: List[str], deleted_at: float) -> int:
    """Soft-delete memories by tag patterns (uses parameterized queries)."""
    if not patterns:
        return 0

    # Use parameterized queries
    conditions = []
    params = [deleted_at]  # Start with deleted_at parameter
    for pattern in patterns:
        conditions.append("(',' || REPLACE(tags, ' ', '') || ',') GLOB ?")
        params.append(pattern)

    where = " OR ".join(conditions)
    query = f"""
        UPDATE memories
        SET deleted_at = ?
        WHERE deleted_at IS NULL AND ({where})
    """
    cursor.execute(query, params)
    return cursor.rowcount


def soft_delete_by_content_patterns(cursor: sqlite3.Cursor, patterns: List[str], deleted_at: float) -> int:
    """Soft-delete memories by content patterns (uses parameterized queries)."""
    if not patterns:
        return 0

    # Use parameterized queries for LIKE patterns
    conditions = ["LOWER(content) LIKE ?" for _ in patterns]
    where = " OR ".join(conditions)
    params = [deleted_at] + patterns
    query = f"""
        UPDATE memories
        SET deleted_at = ?
        WHERE deleted_at IS NULL AND ({where})
    """
    cursor.execute(query, params)
    return cursor.rowcount


def soft_delete_uuid_patterns(cursor: sqlite3.Cursor, deleted_at: float) -> int:
    """Soft-delete memories with UUID patterns in content."""
    cursor.execute("""
        UPDATE memories
        SET deleted_at = ?
        WHERE deleted_at IS NULL
        AND content GLOB '*[[][a-f0-9][a-f0-9][a-f0-9][a-f0-9][a-f0-9][a-f0-9][a-f0-9][a-f0-9]-*'
    """, (deleted_at,))
    return cursor.rowcount


def get_stats(cursor: sqlite3.Cursor) -> Tuple[int, int]:
    """Get active and soft-deleted memory counts."""
    cursor.execute("SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL")
    active = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM memories WHERE deleted_at IS NOT NULL")
    tombstones = cursor.fetchone()[0]
    return active, tombstones


def main():
    parser = argparse.ArgumentParser(
        description="Soft-delete test memories from MCP Memory Service database"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("MCP Memory Service - Soft Delete Test Memories")
    print("=" * 70)
    print()

    conn = get_db_connection(args.db_path)
    cursor = conn.cursor()

    # Get initial stats
    active_before, tombstones_before = get_stats(cursor)
    print(f"Database: {args.db_path}")
    print(f"Active memories: {active_before}")
    print(f"Already soft-deleted: {tombstones_before}")
    print()

    # Analyze what will be deleted
    print("Analyzing test memories...")
    print("-" * 70)

    tag_count = count_by_tags(cursor, TEST_TAGS)
    print(f"  By test tags ({len(TEST_TAGS)} patterns): {tag_count}")

    tag_pattern_count = count_by_tag_patterns(cursor, TAG_PATTERNS)
    print(f"  By tag patterns ({len(TAG_PATTERNS)} patterns): {tag_pattern_count}")

    content_count = count_by_content_patterns(cursor, CONTENT_PATTERNS)
    print(f"  By content patterns ({len(CONTENT_PATTERNS)} patterns): {content_count}")

    uuid_count = count_uuid_patterns(cursor)
    print(f"  By UUID patterns in content: {uuid_count}")

    # Note: counts may overlap, actual deletion will be less
    estimated_total = tag_count + tag_pattern_count + content_count + uuid_count
    print("-" * 70)
    print(f"  Estimated total (may overlap): {estimated_total}")
    print()

    if estimated_total == 0:
        print("No test memories found to delete.")
        conn.close()
        return

    if args.dry_run:
        print("[DRY RUN] No changes made.")
        conn.close()
        return

    # Confirm deletion
    if not args.force:
        response = input("Proceed with soft-delete? (yes/no): ").strip().lower()
        if response != "yes":
            print("Deletion cancelled.")
            conn.close()
            return

    print()
    print("Soft-deleting test memories...")
    print("-" * 70)

    deleted_at = time.time()
    total_deleted = 0

    # Delete in order (later deletions won't re-count already deleted)
    deleted = soft_delete_by_tags(cursor, TEST_TAGS, deleted_at)
    print(f"  Deleted by tags: {deleted}")
    total_deleted += deleted

    deleted = soft_delete_by_tag_patterns(cursor, TAG_PATTERNS, deleted_at)
    print(f"  Deleted by tag patterns: {deleted}")
    total_deleted += deleted

    deleted = soft_delete_by_content_patterns(cursor, CONTENT_PATTERNS, deleted_at)
    print(f"  Deleted by content patterns: {deleted}")
    total_deleted += deleted

    deleted = soft_delete_uuid_patterns(cursor, deleted_at)
    print(f"  Deleted by UUID patterns: {deleted}")
    total_deleted += deleted

    conn.commit()

    # Get final stats
    active_after, tombstones_after = get_stats(cursor)

    print("-" * 70)
    print()
    print("Results:")
    print(f"  Total soft-deleted: {total_deleted}")
    print(f"  Active memories: {active_before} -> {active_after}")
    print(f"  Soft-deleted: {tombstones_before} -> {tombstones_after}")
    print()
    print("Note: Soft-deleted memories will be purged after 30 days.")
    print("=" * 70)

    conn.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
