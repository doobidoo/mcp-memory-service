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
Script to repair malformed tags in the memory database.

Detects and fixes tags that contain JSON serialization artifacts like:
- Tags with quotes: ["ai" or "bug-fix"
- Tags with brackets: ["important"] or ["note"]
- Double-serialized JSON arrays

Usage:
    python scripts/maintenance/repair_malformed_tags.py --dry-run  # Preview changes
    python scripts/maintenance/repair_malformed_tags.py            # Apply fixes
"""

import sys
import os
import json
import re
import sqlite3
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Set, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp_memory_service.config import SQLITE_VEC_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_malformed_tag(tag: str) -> List[str]:
    """
    Parse a malformed tag that may contain JSON serialization artifacts.

    Examples:
        ["ai" -> ["ai"]
        "bug-fix" -> ["bug-fix"]
        ["important"] -> ["important"]
        [\"ai\",\"bug\"] -> ["ai", "bug"]

    Returns:
        List of clean tag strings
    """
    # Remove leading/trailing whitespace
    tag = tag.strip()

    # If tag doesn't contain quotes or brackets, it's clean
    if '"' not in tag and '[' not in tag and ']' not in tag:
        return [tag] if tag else []

    # Try to parse as JSON first
    try:
        # Handle cases where tag is a JSON string like ["tag1","tag2"]
        if tag.startswith('[') and tag.endswith(']'):
            parsed = json.loads(tag)
            if isinstance(parsed, list):
                # Recursively clean each element
                result = []
                for item in parsed:
                    result.extend(parse_malformed_tag(str(item)))
                return result
    except json.JSONDecodeError:
        pass

    # Handle escaped quotes like [\"ai\" or \"bug-fix\"
    if '\\"' in tag or tag.startswith('["') or tag.startswith('"'):
        # Remove all quotes and brackets
        cleaned = tag.replace('\\"', '').replace('"', '').replace('[', '').replace(']', '').strip()
        if cleaned:
            return [cleaned]

    # Handle patterns like ["ai" (missing closing bracket/quote)
    if tag.startswith('["') or tag.startswith('"['):
        cleaned = tag.replace('[', '').replace('"', '').strip()
        if cleaned:
            return [cleaned]

    # If nothing worked, just remove quotes and brackets
    cleaned = tag.replace('"', '').replace('[', '').replace(']', '').strip()
    return [cleaned] if cleaned else []


def is_malformed_tag(tag: str) -> bool:
    """Check if a tag contains malformed characters."""
    return any(char in tag for char in ['"', '[', ']', '\\'])


def analyze_tags(db_path: str) -> Tuple[int, int, Set[str], Dict[str, int]]:
    """
    Analyze tags in the database to find malformed entries.

    Returns:
        (total_memories, malformed_count, malformed_tags_set, tag_frequency)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Get all memories with their tags
        cursor.execute("SELECT content_hash, tags FROM memories WHERE tags IS NOT NULL AND tags != ''")
        rows = cursor.fetchall()

        total_memories = len(rows)
        malformed_count = 0
        malformed_tags = set()
        tag_frequency = {}

        for content_hash, tags_str in rows:
            # Parse tags (comma-separated)
            tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()]

            has_malformed = False
            for tag in tags:
                if is_malformed_tag(tag):
                    has_malformed = True
                    malformed_tags.add(tag)
                    tag_frequency[tag] = tag_frequency.get(tag, 0) + 1

            if has_malformed:
                malformed_count += 1

        return total_memories, malformed_count, malformed_tags, tag_frequency

    finally:
        conn.close()


def repair_tags(db_path: str, dry_run: bool = False) -> Tuple[int, int, Dict[str, List[str]]]:
    """
    Repair malformed tags in the database.

    Returns:
        (memories_updated, tags_fixed, replacements_dict)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    memories_updated = 0
    tags_fixed = 0
    replacements = {}  # old_tag -> [new_tags]

    try:
        # Get all memories with tags
        cursor.execute("SELECT content_hash, tags FROM memories WHERE tags IS NOT NULL AND tags != ''")
        rows = cursor.fetchall()

        for content_hash, tags_str in rows:
            # Parse tags (comma-separated)
            original_tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()]

            # Check if any tags are malformed
            needs_repair = any(is_malformed_tag(tag) for tag in original_tags)

            if needs_repair:
                # Parse and clean each tag
                new_tags = []
                for tag in original_tags:
                    if is_malformed_tag(tag):
                        parsed = parse_malformed_tag(tag)
                        if parsed:
                            new_tags.extend(parsed)
                            replacements[tag] = parsed
                            tags_fixed += 1
                    else:
                        new_tags.append(tag)

                # Remove duplicates while preserving order
                seen = set()
                unique_tags = []
                for tag in new_tags:
                    if tag and tag not in seen:
                        seen.add(tag)
                        unique_tags.append(tag)

                # Update database
                new_tags_str = ",".join(unique_tags)

                if dry_run:
                    logger.info(f"[DRY RUN] Would update {content_hash[:8]}...")
                    logger.info(f"  Old: {tags_str}")
                    logger.info(f"  New: {new_tags_str}")
                else:
                    cursor.execute(
                        "UPDATE memories SET tags = ? WHERE content_hash = ?",
                        (new_tags_str, content_hash)
                    )

                memories_updated += 1

        if not dry_run:
            conn.commit()
            logger.info(f"✅ Database updated successfully")

        return memories_updated, tags_fixed, replacements

    finally:
        conn.close()


def create_backup(db_path: str) -> str:
    """Create a backup of the database before modifications."""
    backup_path = f"{db_path}.backup-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    import shutil
    shutil.copy2(db_path, backup_path)

    logger.info(f"✅ Backup created: {backup_path}")
    return backup_path


def main():
    parser = argparse.ArgumentParser(
        description="Repair malformed tags in the memory database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview changes without modifying database
  python repair_malformed_tags.py --dry-run

  # Apply fixes (creates backup automatically)
  python repair_malformed_tags.py

  # Verbose output with detailed logging
  python repair_malformed_tags.py --verbose
"""
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying the database'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default=SQLITE_VEC_PATH,
        help=f'Path to SQLite database (default: {SQLITE_VEC_PATH})'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check if database exists
    if not os.path.exists(args.db_path):
        logger.error(f"❌ Database not found: {args.db_path}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("🔧 Malformed Tag Repair Tool")
    logger.info("=" * 60)
    logger.info(f"Database: {args.db_path}")
    logger.info(f"Mode: {'DRY RUN (no changes)' if args.dry_run else 'REPAIR (will modify database)'}")
    logger.info("")

    # Analyze tags
    logger.info("📊 Analyzing tags...")
    total_memories, malformed_count, malformed_tags, tag_frequency = analyze_tags(args.db_path)

    logger.info(f"Total memories: {total_memories}")
    logger.info(f"Memories with malformed tags: {malformed_count}")
    logger.info(f"Unique malformed tags: {len(malformed_tags)}")
    logger.info("")

    if malformed_count == 0:
        logger.info("✅ No malformed tags found! Database is clean.")
        return

    # Show most common malformed tags
    logger.info("🔍 Most common malformed tags:")
    sorted_tags = sorted(tag_frequency.items(), key=lambda x: x[1], reverse=True)
    for tag, count in sorted_tags[:10]:
        logger.info(f"  {tag!r} -> appears {count} times")
        parsed = parse_malformed_tag(tag)
        logger.info(f"    Will become: {parsed}")
    logger.info("")

    # Create backup if not dry-run
    if not args.dry_run:
        logger.info("💾 Creating backup...")
        backup_path = create_backup(args.db_path)
        logger.info("")

    # Repair tags
    logger.info("🔧 Repairing tags...")
    memories_updated, tags_fixed, replacements = repair_tags(args.db_path, dry_run=args.dry_run)

    logger.info("")
    logger.info("=" * 60)
    logger.info("📈 Summary")
    logger.info("=" * 60)
    logger.info(f"Memories updated: {memories_updated}")
    logger.info(f"Tags fixed: {tags_fixed}")
    logger.info("")

    if replacements:
        logger.info("🔄 Tag replacements:")
        for old_tag, new_tags in list(replacements.items())[:10]:
            logger.info(f"  {old_tag!r} -> {new_tags}")
        if len(replacements) > 10:
            logger.info(f"  ... and {len(replacements) - 10} more")

    logger.info("")
    if args.dry_run:
        logger.info("⚠️  This was a DRY RUN - no changes were made")
        logger.info("   Run without --dry-run to apply fixes")
    else:
        logger.info("✅ Repair completed successfully!")
        logger.info(f"   Backup saved to: {backup_path}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
