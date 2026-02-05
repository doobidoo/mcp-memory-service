#!/usr/bin/env python3
"""
Migration script to convert Cloudflare Vectorize tag metadata from string to array format.

After commit ee1cac5, tags in Vectorize metadata should be stored as arrays (e.g., ["python", "code"])
instead of strings (e.g., "python,code"). This script migrates existing vectors to the new format.

This migration is REQUIRED for tag filtering to work correctly with Cloudflare backend.

Usage:
    python scripts/sync/migrate_cloudflare_tags.py --dry-run  # Preview changes
    python scripts/sync/migrate_cloudflare_tags.py           # Apply migration
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any

import httpx

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CloudflareMigrator:
    """Migrates Cloudflare Vectorize tag metadata from string to array format."""

    def __init__(self):
        """Initialize with Cloudflare credentials from environment."""
        self.api_token = os.environ.get("CLOUDFLARE_API_TOKEN")
        self.account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        self.d1_database_id = os.environ.get("CLOUDFLARE_D1_DATABASE_ID")
        self.vectorize_index = os.environ.get("CLOUDFLARE_VECTORIZE_INDEX", "mcp-memory-index")

        if not all([self.api_token, self.account_id, self.d1_database_id]):
            raise RuntimeError(
                "Missing required environment variables:\n"
                "  - CLOUDFLARE_API_TOKEN\n"
                "  - CLOUDFLARE_ACCOUNT_ID\n"
                "  - CLOUDFLARE_D1_DATABASE_ID"
            )

        self.d1_url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/d1/database/{self.d1_database_id}"
        self.vectorize_url = (
            f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/vectorize/v2/indexes/{self.vectorize_index}"
        )

        self.client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"}, timeout=30.0
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()

    async def _retry_request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make HTTP request with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPError as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise

    async def get_all_memories_from_d1(self) -> list[dict[str, Any]]:
        """Fetch all memories from D1 database."""
        logger.info("Fetching all memories from D1...")

        sql = "SELECT content_hash, tags FROM memories"
        payload = {"sql": sql}

        response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
        result = response.json()

        if not result.get("success"):
            raise RuntimeError(f"D1 query failed: {result}")

        rows = result.get("result", [{}])[0].get("results", [])
        logger.info(f"  Found {len(rows)} memories in D1")

        return rows

    async def get_vector_by_id(self, vector_id: str) -> dict[str, Any] | None:
        """Fetch a single vector from Vectorize by ID."""
        try:
            response = await self._retry_request("GET", f"{self.vectorize_url}/getByIds", params={"ids": vector_id})
            result = response.json()

            if not result.get("success"):
                logger.warning(f"Failed to fetch vector {vector_id}: {result}")
                return None

            vectors = result.get("result", {}).get("vectors", [])
            return vectors[0] if vectors else None

        except Exception as e:
            logger.warning(f"Error fetching vector {vector_id}: {e}")
            return None

    def needs_migration(self, tags_value: Any) -> bool:
        """Check if tags value needs migration (string → array)."""
        if tags_value is None:
            return False
        if isinstance(tags_value, str):
            return True  # String format needs conversion
        if isinstance(tags_value, list):
            return False  # Already array format
        logger.warning(f"Unexpected tags type: {type(tags_value)} = {tags_value}")
        return False

    def convert_tags(self, tags_value: Any) -> list[str]:
        """Convert tags from string to array format."""
        if tags_value is None or tags_value == "":
            return []
        if isinstance(tags_value, list):
            return tags_value  # Already array
        if isinstance(tags_value, str):
            # Split comma-separated string
            return [tag.strip() for tag in tags_value.split(",") if tag.strip()]
        return []

    async def update_vector_metadata(
        self, vector_id: str, embedding: list[float], metadata: dict[str, Any], dry_run: bool = False
    ) -> bool:
        """Update vector metadata via upsert operation."""
        if dry_run:
            logger.info(f"  DRY RUN - Would update vector {vector_id} with metadata: {metadata}")
            return True

        vector_data = {"id": vector_id, "values": embedding, "metadata": metadata}

        # Convert to NDJSON format
        ndjson_content = json.dumps(vector_data) + "\n"

        try:
            response = await self.client.request(
                "POST",
                f"{self.vectorize_url}/insert",
                content=ndjson_content,
                headers={"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/x-ndjson"},
            )
            response.raise_for_status()
            result = response.json()

            if not result.get("success"):
                logger.error(f"Failed to update vector {vector_id}: {result}")
                return False

            logger.info(f"  ✓ Updated vector {vector_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating vector {vector_id}: {e}")
            return False

    async def migrate(self, dry_run: bool = False) -> dict[str, int]:
        """
        Run the migration.

        Returns:
            Dictionary with counts: total, needs_migration, migrated, failed
        """
        logger.info("Starting Cloudflare tag migration...")
        logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")

        # Fetch all memories from D1
        memories = await self.get_all_memories_from_d1()

        stats = {"total": len(memories), "needs_migration": 0, "migrated": 0, "failed": 0, "skipped": 0}

        # Process each memory
        for i, memory in enumerate(memories, 1):
            content_hash = memory.get("content_hash")
            d1_tags = memory.get("tags")

            logger.info(f"\n[{i}/{stats['total']}] Processing {content_hash}...")

            # Fetch vector from Vectorize
            vector = await self.get_vector_by_id(content_hash)

            if not vector:
                logger.warning(f"  Vector not found in Vectorize: {content_hash}")
                stats["skipped"] += 1
                continue

            # Check vector metadata
            vector_metadata = vector.get("metadata", {})
            vector_tags = vector_metadata.get("tags")

            # Check if migration needed
            if not self.needs_migration(vector_tags):
                logger.info(f"  Already migrated (tags: {vector_tags})")
                continue

            stats["needs_migration"] += 1

            # Convert tags
            if vector_tags is not None:
                # Use vector tags as source
                new_tags = self.convert_tags(vector_tags)
            elif d1_tags:
                # Fall back to D1 tags
                new_tags = self.convert_tags(d1_tags)
            else:
                new_tags = []

            logger.info(f"  Converting: {repr(vector_tags)} → {new_tags}")

            # Update metadata
            updated_metadata = vector_metadata.copy()
            updated_metadata["tags"] = new_tags

            # Get embedding from vector
            embedding = vector.get("values", [])

            # Update vector
            success = await self.update_vector_metadata(content_hash, embedding, updated_metadata, dry_run=dry_run)

            if success:
                stats["migrated"] += 1
            else:
                stats["failed"] += 1

        return stats

    async def verify_migration(self, sample_size: int = 5) -> bool:
        """Verify migration by sampling vectors."""
        logger.info(f"\nVerifying migration (sampling {sample_size} vectors)...")

        memories = await self.get_all_memories_from_d1()
        sample = memories[:sample_size]

        all_correct = True

        for memory in sample:
            content_hash = memory.get("content_hash")
            vector = await self.get_vector_by_id(content_hash)

            if not vector:
                logger.warning(f"  Vector not found: {content_hash}")
                continue

            tags = vector.get("metadata", {}).get("tags")

            if isinstance(tags, list):
                logger.info(f"  ✓ {content_hash}: tags = {tags}")
            else:
                logger.error(f"  ✗ {content_hash}: tags = {tags} (type: {type(tags)})")
                all_correct = False

        return all_correct


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Migrate Cloudflare Vectorize tags to array format")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--verify", action="store_true", help="Verify migration (sample check)")
    parser.add_argument("--sample-size", type=int, default=5, help="Number of vectors to sample during verification")
    args = parser.parse_args()

    try:
        async with CloudflareMigrator() as migrator:
            if args.verify:
                # Verification mode
                success = await migrator.verify_migration(args.sample_size)
                if success:
                    logger.info("\n✓ Verification passed - all sampled vectors have array-format tags")
                    return 0
                else:
                    logger.error("\n✗ Verification failed - some vectors still have string-format tags")
                    return 1

            # Migration mode
            stats = await migrator.migrate(dry_run=args.dry_run)

            # Print summary
            logger.info("\n" + "=" * 60)
            logger.info("Migration Summary:")
            logger.info(f"  Total memories: {stats['total']}")
            logger.info(f"  Needed migration: {stats['needs_migration']}")
            logger.info(f"  Successfully migrated: {stats['migrated']}")
            logger.info(f"  Failed: {stats['failed']}")
            logger.info(f"  Skipped: {stats['skipped']}")
            logger.info("=" * 60)

            if not args.dry_run and stats["migrated"] > 0:
                logger.info("\nRun with --verify to confirm migration:")
                logger.info(f"  python {sys.argv[0]} --verify")

            return 0 if stats["failed"] == 0 else 1

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
