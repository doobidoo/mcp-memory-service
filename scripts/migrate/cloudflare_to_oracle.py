#!/usr/bin/env python3
"""
Cloudflare to Oracle Migration Script

Migrates memories from Cloudflare D1/Vectorize backend to Oracle Cloud
SQLite-vec HTTP server with cursor-based pagination and batch operations.

Usage:
    python scripts/migrate/cloudflare_to_oracle.py --oracle-endpoint http://100.x.x.x:8000 [--dry-run]

Environment Variables:
    CLOUDFLARE_API_TOKEN          - Cloudflare API token
    CLOUDFLARE_ACCOUNT_ID         - Cloudflare account ID
    CLOUDFLARE_D1_DATABASE_ID     - D1 database ID
    CLOUDFLARE_VECTORIZE_INDEX    - Vectorize index name
    MCP_API_KEY                   - Oracle server API key (if required)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.mcp_memory_service.storage.cloudflare import CloudflareStorage

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CloudflareMigrator:
    """Migrates memories from Cloudflare to Oracle HTTP server."""

    def __init__(self, oracle_endpoint: str, api_key: str | None = None, batch_size: int = 50, dry_run: bool = False):
        self.oracle_endpoint = oracle_endpoint.rstrip("/")
        self.api_key = api_key
        self.batch_size = batch_size
        self.dry_run = dry_run

        self.cloudflare_storage: CloudflareStorage | None = None
        self.stats = {
            "total_memories": 0,
            "migrated": 0,
            "skipped": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None,
        }

    async def initialize_cloudflare(self):
        """Initialize Cloudflare storage backend."""
        logger.info("Initializing Cloudflare storage...")

        # Validate environment variables
        required_vars = [
            "CLOUDFLARE_API_TOKEN",
            "CLOUDFLARE_ACCOUNT_ID",
            "CLOUDFLARE_D1_DATABASE_ID",
            "CLOUDFLARE_VECTORIZE_INDEX",
        ]

        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        self.cloudflare_storage = CloudflareStorage()
        logger.info("Cloudflare storage initialized successfully")

    async def fetch_all_memories(self) -> list[dict[str, Any]]:
        """
        Fetch all memories from Cloudflare using cursor-based pagination.

        Avoids D1 OFFSET limitations by using content_hash as cursor.

        Returns:
            List of memory dictionaries with all fields
        """
        logger.info("Fetching memories from Cloudflare...")
        memories = []
        cursor = None
        page = 0

        while True:
            page += 1

            # Use search with pagination (CloudflareStorage.search supports limits)
            # For full migration, use empty query to get all memories
            try:
                # Fetch batch using content hash cursor
                if cursor:
                    # Query memories greater than cursor (lexicographic order)
                    pass
                else:
                    pass

                # Use storage's internal methods to fetch raw data
                batch = await self._fetch_batch_from_cloudflare(cursor, self.batch_size)

                if not batch:
                    logger.info(f"No more memories to fetch (page {page})")
                    break

                memories.extend(batch)
                logger.info(f"Fetched page {page}: {len(batch)} memories (total: {len(memories)})")

                # Update cursor to last content_hash
                cursor = batch[-1]["content_hash"]

                # If batch is smaller than batch_size, we've reached the end
                if len(batch) < self.batch_size:
                    logger.info(f"Reached end of data (page {page})")
                    break

            except Exception as e:
                logger.error(f"Error fetching page {page}: {e}")
                raise

        logger.info(f"Total memories fetched: {len(memories)}")
        self.stats["total_memories"] = len(memories)
        return memories

    async def _fetch_batch_from_cloudflare(self, cursor: str | None, limit: int) -> list[dict[str, Any]]:
        """
        Fetch a batch of memories from Cloudflare D1.

        Uses content_hash cursor for pagination to avoid OFFSET limitations.
        """
        # Build SQL query with cursor
        if cursor:
            query = f"""
            SELECT content, content_hash, tags, metadata, memory_type, created_at, updated_at
            FROM memories
            WHERE content_hash > ?
            ORDER BY content_hash ASC
            LIMIT {limit}
            """
            params = [cursor]
        else:
            query = f"""
            SELECT content, content_hash, tags, metadata, memory_type, created_at, updated_at
            FROM memories
            ORDER BY content_hash ASC
            LIMIT {limit}
            """
            params = []

        # Execute query via Cloudflare storage
        results = await self.cloudflare_storage._query_d1(query, params)

        # Parse results
        memories = []
        for row in results:
            memory = {
                "content": row["content"],
                "content_hash": row["content_hash"],
                "tags": json.loads(row["tags"]) if isinstance(row["tags"], str) else row["tags"],
                "metadata": json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
                "memory_type": row.get("memory_type"),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
            }
            memories.append(memory)

        return memories

    async def migrate_batch(self, memories: list[dict[str, Any]]) -> dict[str, int]:
        """
        Migrate a batch of memories to Oracle HTTP server.

        Args:
            memories: List of memory dictionaries to migrate

        Returns:
            Dictionary with migration statistics
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Would migrate batch of {len(memories)} memories")
            return {"migrated": len(memories), "skipped": 0, "errors": 0}

        batch_stats = {"migrated": 0, "skipped": 0, "errors": 0}

        async with httpx.AsyncClient(timeout=30.0) as client:
            for memory in memories:
                try:
                    # Prepare request payload
                    payload = {
                        "content": memory["content"],
                        "tags": memory["tags"],
                        "memory_type": memory.get("memory_type"),
                        "metadata": memory.get("metadata", {}),
                    }

                    # Add timestamps to metadata for preservation
                    if "created_at" in memory and memory["created_at"]:
                        payload["metadata"]["migrated_created_at"] = memory["created_at"]
                    if "updated_at" in memory and memory["updated_at"]:
                        payload["metadata"]["migrated_updated_at"] = memory["updated_at"]

                    # Prepare headers
                    headers = {"Content-Type": "application/json"}
                    if self.api_key:
                        headers["Authorization"] = f"Bearer {self.api_key}"

                    # POST to Oracle server
                    response = await client.post(f"{self.oracle_endpoint}/api/memories", json=payload, headers=headers)

                    if response.status_code == 200:
                        batch_stats["migrated"] += 1
                    elif response.status_code == 409:
                        # Duplicate (content hash already exists) - skip
                        batch_stats["skipped"] += 1
                        logger.debug(f"Skipped duplicate: {memory['content_hash'][:16]}...")
                    else:
                        batch_stats["errors"] += 1
                        logger.error(
                            f"Failed to migrate memory {memory['content_hash'][:16]}: {response.status_code} {response.text}"
                        )

                except Exception as e:
                    batch_stats["errors"] += 1
                    logger.error(f"Error migrating memory {memory.get('content_hash', 'unknown')[:16]}: {e}")

        return batch_stats

    async def verify_migration(self) -> bool:
        """
        Verify migration by comparing memory counts.

        Returns:
            True if counts match, False otherwise
        """
        logger.info("Verifying migration...")

        # Get count from Oracle server
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = await client.get(f"{self.oracle_endpoint}/api/health/detailed", headers=headers)

            if response.status_code != 200:
                logger.error(f"Failed to get Oracle server health: {response.status_code}")
                return False

            health_data = response.json()
            oracle_count = health_data.get("statistics", {}).get("total_memories", 0)

        cloudflare_count = self.stats["total_memories"]
        migrated_count = self.stats["migrated"]

        logger.info("Verification results:")
        logger.info(f"  Cloudflare source: {cloudflare_count} memories")
        logger.info(f"  Oracle destination: {oracle_count} memories")
        logger.info(f"  Migrated this run: {migrated_count} memories")
        logger.info(f"  Skipped (duplicates): {self.stats['skipped']} memories")
        logger.info(f"  Errors: {self.stats['errors']} memories")

        # Verification passes if Oracle has at least as many memories as migrated
        if oracle_count >= migrated_count:
            logger.info("✓ Verification passed")
            return True
        else:
            logger.error("✗ Verification failed: Oracle count doesn't match migrated count")
            return False

    async def run(self):
        """Execute full migration workflow."""
        self.stats["start_time"] = datetime.now()

        try:
            # Initialize Cloudflare
            await self.initialize_cloudflare()

            # Fetch all memories
            memories = await self.fetch_all_memories()

            if not memories:
                logger.info("No memories to migrate")
                return

            # Migrate in batches
            logger.info(f"Migrating {len(memories)} memories in batches of {self.batch_size}...")

            for i in range(0, len(memories), self.batch_size):
                batch = memories[i : i + self.batch_size]
                batch_num = (i // self.batch_size) + 1
                total_batches = (len(memories) + self.batch_size - 1) // self.batch_size

                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} memories)")

                batch_stats = await self.migrate_batch(batch)

                self.stats["migrated"] += batch_stats["migrated"]
                self.stats["skipped"] += batch_stats["skipped"]
                self.stats["errors"] += batch_stats["errors"]

                logger.info(
                    f"Batch {batch_num} complete: {batch_stats['migrated']} migrated, {batch_stats['skipped']} skipped, {batch_stats['errors']} errors"
                )

            # Verify migration
            if not self.dry_run:
                verification_passed = await self.verify_migration()
            else:
                logger.info("[DRY RUN] Skipping verification")
                verification_passed = True

            # Print final report
            self.stats["end_time"] = datetime.now()
            duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()

            logger.info("")
            logger.info("=" * 60)
            logger.info("MIGRATION REPORT")
            logger.info("=" * 60)
            logger.info(f"Total memories: {self.stats['total_memories']}")
            logger.info(f"Successfully migrated: {self.stats['migrated']}")
            logger.info(f"Skipped (duplicates): {self.stats['skipped']}")
            logger.info(f"Errors: {self.stats['errors']}")
            logger.info(f"Duration: {duration:.1f} seconds")
            logger.info(f"Verification: {'PASSED' if verification_passed else 'FAILED'}")
            logger.info("=" * 60)

            if verification_passed and self.stats["errors"] == 0:
                logger.info("✓ Migration completed successfully!")
            else:
                logger.warning("⚠ Migration completed with warnings or errors")

        except Exception as e:
            logger.error(f"Migration failed: {e}", exc_info=True)
            raise


def main():
    parser = argparse.ArgumentParser(description="Migrate memories from Cloudflare to Oracle")
    parser.add_argument("--oracle-endpoint", required=True, help="Oracle HTTP server endpoint (e.g., http://100.x.x.x:8000)")
    parser.add_argument("--api-key", default=os.getenv("MCP_API_KEY"), help="Oracle server API key (or set MCP_API_KEY env var)")
    parser.add_argument("--batch-size", type=int, default=50, help="Number of memories to migrate per batch (default: 50)")
    parser.add_argument("--dry-run", action="store_true", help="Preview migration without actually migrating data")

    args = parser.parse_args()

    migrator = CloudflareMigrator(
        oracle_endpoint=args.oracle_endpoint, api_key=args.api_key, batch_size=args.batch_size, dry_run=args.dry_run
    )

    asyncio.run(migrator.run())


if __name__ == "__main__":
    main()
