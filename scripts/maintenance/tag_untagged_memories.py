#!/usr/bin/env python3
"""
Tag Untagged Memories - Cleanup Script

Finds all memories with empty tags in the production database and:
- Soft-deletes test artifacts
- Tags document chunks as "untagged,document"
- Tags real memories as "untagged"

Usage:
    python scripts/maintenance/tag_untagged_memories.py --dry-run   # Preview (default)
    python scripts/maintenance/tag_untagged_memories.py --apply      # Apply changes
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from datetime import datetime

DEFAULT_DB = Path.home() / ".local" / "share" / "mcp-memory" / "sqlite_vec.db"

TEST_PATTERNS = [
    "test content for hash length",
    "this is a test memory",
    "test content",
    "test memory for",
    "backup test",
    "__test__",
]


def find_untagged(conn):
    """Find all non-deleted memories with empty or NULL tags."""
    cursor = conn.execute(
        "SELECT content_hash, content, memory_type, tags "
        "FROM memories "
        "WHERE deleted_at IS NULL AND (tags IS NULL OR tags = '')"
    )
    return cursor.fetchall()


def classify(content, memory_type):
    """Classify a memory as test, document, or real."""
    content_lower = content.lower()
    for pattern in TEST_PATTERNS:
        if pattern in content_lower:
            return "test"
    if memory_type == "document":
        return "document"
    return "real"


def main():
    parser = argparse.ArgumentParser(description="Tag untagged memories in production DB")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to SQLite database")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Preview changes only")
    group.add_argument("--apply", action="store_true", help="Apply changes to database")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(args.db))
    rows = find_untagged(conn)
    print(f"Found {len(rows)} untagged memories\n")

    if not rows:
        print("Nothing to do.")
        conn.close()
        return

    # Classify
    test_rows = []
    doc_rows = []
    real_rows = []

    for content_hash, content, memory_type, tags in rows:
        cat = classify(content, memory_type)
        if cat == "test":
            test_rows.append((content_hash, content))
        elif cat == "document":
            doc_rows.append((content_hash, content))
        else:
            real_rows.append((content_hash, content))

    print(f"Classification:")
    print(f"  Test artifacts (soft-delete):  {len(test_rows)}")
    print(f"  Document chunks (tag):         {len(doc_rows)}")
    print(f"  Real memories (tag):           {len(real_rows)}")
    print()

    if args.dry_run:
        print("--- DRY RUN (no changes) ---\n")
        if test_rows:
            print("Would SOFT-DELETE:")
            for h, c in test_rows[:5]:
                print(f"  {h[:12]}... | {c[:60]}")
            if len(test_rows) > 5:
                print(f"  ... and {len(test_rows) - 5} more")
            print()
        if doc_rows:
            print("Would TAG as 'untagged,document':")
            for h, c in doc_rows[:5]:
                print(f"  {h[:12]}... | {c[:60]}")
            if len(doc_rows) > 5:
                print(f"  ... and {len(doc_rows) - 5} more")
            print()
        if real_rows:
            print("Would TAG as 'untagged':")
            for h, c in real_rows[:5]:
                print(f"  {h[:12]}... | {c[:60]}")
            if len(real_rows) > 5:
                print(f"  ... and {len(real_rows) - 5} more")
            print()
        print("Run with --apply to execute.")
        conn.close()
        return

    # Apply changes
    now = datetime.utcnow().isoformat() + "Z"
    deleted = 0
    tagged = 0

    for content_hash, _ in test_rows:
        conn.execute(
            "UPDATE memories SET deleted_at = ? WHERE content_hash = ?",
            (now, content_hash),
        )
        deleted += 1

    for content_hash, _ in doc_rows:
        conn.execute(
            "UPDATE memories SET tags = ? WHERE content_hash = ?",
            ("untagged,document", content_hash),
        )
        tagged += 1

    for content_hash, _ in real_rows:
        conn.execute(
            "UPDATE memories SET tags = ? WHERE content_hash = ?",
            ("untagged", content_hash),
        )
        tagged += 1

    conn.commit()
    conn.close()

    print(f"Done:")
    print(f"  Soft-deleted: {deleted}")
    print(f"  Tagged:       {tagged}")

    # Verify
    conn2 = sqlite3.connect(str(args.db))
    remaining = conn2.execute(
        "SELECT COUNT(*) FROM memories "
        "WHERE deleted_at IS NULL AND (tags IS NULL OR tags = '')"
    ).fetchone()[0]
    conn2.close()
    print(f"  Remaining untagged: {remaining}")


if __name__ == "__main__":
    main()
