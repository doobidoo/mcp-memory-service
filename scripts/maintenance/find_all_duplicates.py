#!/usr/bin/env python3
"""Find all near-duplicate memories across the database."""

import sqlite3
from pathlib import Path
from collections import defaultdict
import hashlib

import platform

# Platform-specific database path
if platform.system() == "Darwin":  # macOS
    DB_PATH = Path.home() / "Library/Application Support/mcp-memory/sqlite_vec.db"
else:  # Linux/Windows
    DB_PATH = Path.home() / ".local/share/mcp-memory/sqlite_vec.db"

def normalize_content(content):
    """Normalize content by removing timestamps and session-specific data."""
    # Remove common timestamp patterns
    import re
    normalized = content
    normalized = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z', 'TIMESTAMP', normalized)
    normalized = re.sub(r'\*\*Date\*\*: \d{2,4}[./]\d{2}[./]\d{2,4}', '**Date**: DATE', normalized)
    normalized = re.sub(r'Timestamp: \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', 'Timestamp: TIMESTAMP', normalized)

    return normalized.strip()

def content_hash(content):
    """Create a hash of normalized content."""
    normalized = normalize_content(content)
    return hashlib.md5(normalized.encode()).hexdigest()

def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("Analyzing memories for duplicates...")
    cursor.execute("SELECT content_hash, content, tags, created_at FROM memories ORDER BY created_at DESC")

    memories = cursor.fetchall()
    print(f"Total memories: {len(memories)}")

    # Group by normalized content
    content_groups = defaultdict(list)
    for mem_hash, content, tags, created_at in memories:
        norm_hash = content_hash(content)
        content_groups[norm_hash].append({
            'hash': mem_hash,
            'content': content[:200],  # First 200 chars
            'tags': tags,
            'created_at': created_at
        })

    # Find duplicates (groups with >1 memory)
    duplicates = {k: v for k, v in content_groups.items() if len(v) > 1}

    if not duplicates:
        print("✅ No duplicates found!")
        conn.close()
        return

    print(f"\n❌ Found {len(duplicates)} groups of duplicates:")

    total_duplicate_count = 0
    for i, (norm_hash, group) in enumerate(duplicates.items(), 1):
        count = len(group)
        total_duplicate_count += count - 1  # Keep one, delete rest

        print(f"\n{i}. Group with {count} duplicates:")
        print(f"   Content preview: {group[0]['content'][:100]}...")
        print(f"   Tags: {group[0]['tags'][:80]}...")
        print(f"   Hashes to keep: {group[0]['hash'][:16]}... (newest)")
        print(f"   Hashes to delete: {count-1} older duplicates")

        if i >= 10:  # Show only first 10 groups
            remaining = len(duplicates) - 10
            print(f"\n... and {remaining} more duplicate groups")
            break

    print(f"\n📊 Summary:")
    print(f"   Total duplicate groups: {len(duplicates)}")
    print(f"   Total memories to delete: {total_duplicate_count}")
    print(f"   Total memories after cleanup: {len(memories) - total_duplicate_count}")

    conn.close()

if __name__ == "__main__":
    main()
