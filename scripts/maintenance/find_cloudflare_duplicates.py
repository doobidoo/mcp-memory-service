#!/usr/bin/env python3
"""Find all near-duplicate memories in Cloudflare D1 database."""

import asyncio
import hashlib
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


async def main():
    # Set OAuth to false to avoid validation issues
    os.environ["MCP_OAUTH_ENABLED"] = "false"

    # Import after setting environment
    from mcp_memory_service.config import (
        CLOUDFLARE_ACCOUNT_ID,
        CLOUDFLARE_API_TOKEN,
        CLOUDFLARE_D1_DATABASE_ID,
        CLOUDFLARE_VECTORIZE_INDEX,
        EMBEDDING_MODEL_NAME,
    )
    from mcp_memory_service.storage.cloudflare import CloudflareStorage

    def normalize_content(content):
        """Normalize content by removing timestamps and session-specific data."""
        # Remove common timestamp patterns
        normalized = content
        normalized = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", "TIMESTAMP", normalized)
        normalized = re.sub(r"\*\*Date\*\*: \d{2,4}[./]\d{2}[./]\d{2,4}", "**Date**: DATE", normalized)
        normalized = re.sub(r"Timestamp: \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "Timestamp: TIMESTAMP", normalized)
        return normalized.strip()

    def content_hash(content):
        """Create a hash of normalized content."""
        normalized = normalize_content(content)
        return hashlib.md5(normalized.encode()).hexdigest()

    print("🔗 Connecting to Cloudflare...")

    # Initialize Cloudflare storage
    cloudflare = CloudflareStorage(
        api_token=CLOUDFLARE_API_TOKEN,
        account_id=CLOUDFLARE_ACCOUNT_ID,
        vectorize_index=CLOUDFLARE_VECTORIZE_INDEX,
        d1_database_id=CLOUDFLARE_D1_DATABASE_ID,
        embedding_model=EMBEDDING_MODEL_NAME,
    )

    await cloudflare.initialize()
    print("✅ Connected to Cloudflare\n")

    print("📊 Fetching all memories from Cloudflare D1...")

    # Use the public API method for better encapsulation and performance
    try:
        all_memories = await cloudflare.get_all_memories_bulk(include_tags=False)
    except Exception as e:
        print(f"❌ Failed to fetch memories from Cloudflare D1: {e}")
        return 1

    if not all_memories:
        print("✅ No memories found to check for duplicates.")
        return 0

    # Convert Memory objects to the expected format for the rest of the script
    memories = []
    for memory in all_memories:
        memories.append(
            {
                "content_hash": memory.content_hash,
                "content": memory.content,
                "tags": ",".join(memory.tags),  # Convert list to comma-separated string
                "created_at": memory.created_at,
            }
        )
    print(f"Total memories in Cloudflare: {len(memories)}\n")

    # Group by normalized content
    content_groups = defaultdict(list)
    for mem in memories:
        norm_hash = content_hash(mem["content"])
        content_groups[norm_hash].append(
            {
                "hash": mem["content_hash"],
                "content": mem["content"][:200],  # First 200 chars
                "tags": mem["tags"][:80] if mem["tags"] else "",
                "created_at": mem["created_at"],
            }
        )

    # Find duplicates (groups with >1 memory)
    duplicates = {k: v for k, v in content_groups.items() if len(v) > 1}

    if not duplicates:
        print("✅ No duplicates found in Cloudflare!")
        return 0

    print(f"\n❌ Found {len(duplicates)} groups of duplicates:\n")

    total_duplicate_count = 0
    for i, (norm_hash, group) in enumerate(duplicates.items(), 1):
        count = len(group)
        total_duplicate_count += count - 1  # Keep one, delete rest

        print(f"{i}. Group with {count} duplicates:")
        print(f"   Content preview: {group[0]['content'][:100]}...")
        print(f"   Tags: {group[0]['tags'][:80]}...")
        print(f"   Hashes to keep: {group[0]['hash'][:16]}... (newest)")
        print(f"   Hashes to delete: {count - 1} older duplicates")

        if i >= 10:  # Show only first 10 groups
            remaining = len(duplicates) - 10
            print(f"\n... and {remaining} more duplicate groups")
            break

    print("\n📊 Summary:")
    print(f"   Total duplicate groups: {len(duplicates)}")
    print(f"   Total memories to delete: {total_duplicate_count}")
    print(f"   Total memories after cleanup: {len(memories) - total_duplicate_count}")

    # Ask if user wants to save hashes for deletion
    save_hashes = input("\n💾 Save duplicate hashes for deletion? (y/n): ").strip().lower()

    if save_hashes == "y":
        hash_file = Path.home() / "cloudflare_duplicates.txt"

        # Collect hashes to delete (keep newest, delete older)
        hashes_to_delete = []
        for group in duplicates.values():
            for memory in group[1:]:  # Keep first (newest), delete rest
                hashes_to_delete.append(memory["hash"])

        with open(hash_file, "w") as f:
            for content_hash in hashes_to_delete:
                f.write(f"{content_hash}\n")

        print(f"\n✅ Saved {len(hashes_to_delete)} hashes to {hash_file}")
        print("📋 Next step: Delete from Cloudflare")
        print("   Update delete_cloudflare_duplicates.py to read from cloudflare_duplicates.txt")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
