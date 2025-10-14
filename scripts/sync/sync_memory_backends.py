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
Bidirectional sync script for MCP Memory Service backends.
Syncs memories between Cloudflare (primary) and SQLite-vec (backup).
"""
import sys
import os
import asyncio
import logging
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Add src directory to path so we can import from the mcp_memory_service package
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mcp_memory_service.storage.cloudflare import CloudflareStorage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.config import (
    CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_VECTORIZE_INDEX,
    CLOUDFLARE_D1_DATABASE_ID, BASE_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("memory_sync")

class MemorySync:
    """Handles bidirectional sync between Cloudflare and SQLite-vec backends."""

    def __init__(self, sqlite_path: str = None):
        """Initialize sync with storage backends."""
        self.sqlite_path = sqlite_path or os.path.join(BASE_DIR, 'backup_sqlite_vec.db')

        # Initialize storage backends
        self.cloudflare = CloudflareStorage(
            api_token=CLOUDFLARE_API_TOKEN,
            account_id=CLOUDFLARE_ACCOUNT_ID,
            vectorize_index=CLOUDFLARE_VECTORIZE_INDEX,
            d1_database_id=CLOUDFLARE_D1_DATABASE_ID
        )

        self.sqlite_vec = SqliteVecMemoryStorage(self.sqlite_path)

    async def get_all_memories_from_backend(self, backend_name: str) -> List[Dict[str, Any]]:
        """Get all memories from a specific backend."""
        if backend_name == 'cloudflare':
            backend = self.cloudflare
        elif backend_name == 'sqlite_vec':
            backend = self.sqlite_vec
        else:
            raise ValueError(f"Unknown backend: {backend_name}")

        try:
            # Get all memories from the backend
            memories_list = await backend.get_all_memories()

            memories = []
            for memory in memories_list:
                memory_dict = {
                    'id': memory.id,
                    'content': memory.content,
                    'metadata': memory.metadata,
                    'timestamp': memory.timestamp,
                    'hash': memory.hash
                }
                memories.append(memory_dict)

            logger.info(f"Retrieved {len(memories)} memories from {backend_name}")
            return memories

        except Exception as e:
            logger.error(f"Error retrieving memories from {backend_name}: {e}")
            return []

    def calculate_content_hash(self, content: str, metadata: Dict[str, Any]) -> str:
        """Calculate a hash for memory content to detect duplicates."""
        # Create a consistent string representation
        content_str = f"{content}_{sorted(metadata.items())}"
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    async def _sync_between_backends(self, source_backend: str, target_backend: str, dry_run: bool = False) -> Tuple[int, int]:
        """
        Generic method to sync memories between any two backends.

        Args:
            source_backend: Backend to sync from ('cloudflare' or 'sqlite_vec')
            target_backend: Backend to sync to ('cloudflare' or 'sqlite_vec')
            dry_run: If True, only show what would be synced without making changes

        Returns:
            Tuple of (added_count, skipped_count)
        """
        logger.info(f"Starting sync from {source_backend} to {target_backend}...")

        # Get memories from both backends
        source_memories = await self.get_all_memories_from_backend(source_backend)
        target_memories = await self.get_all_memories_from_backend(target_backend)

        # Create hash sets for quick lookup
        target_hashes = {mem['hash'] for mem in target_memories if mem.get('hash')}
        target_content_hashes = {
            self.calculate_content_hash(mem['content'], mem['metadata'])
            for mem in target_memories
        }

        added_count = 0
        skipped_count = 0

        # Get target backend instance for storing memories
        target_storage = self.cloudflare if target_backend == 'cloudflare' else self.sqlite_vec

        for source_memory in source_memories:
            # Check if memory already exists (by hash or content)
            content_hash = self.calculate_content_hash(source_memory['content'], source_memory['metadata'])

            if (source_memory.get('hash') in target_hashes or
                content_hash in target_content_hashes):
                skipped_count += 1
                continue

            if not dry_run:
                try:
                    await target_storage.store_memory(
                        content=source_memory['content'],
                        metadata=source_memory['metadata']
                    )
                    added_count += 1
                    logger.debug(f"Added memory: {source_memory['id'][:8]}...")
                except Exception as e:
                    logger.error(f"Error storing memory {source_memory['id']}: {e}")
            else:
                added_count += 1

        logger.info(f"{source_backend} → {target_backend}: {added_count} added, {skipped_count} skipped")
        return added_count, skipped_count

    async def sync_cloudflare_to_sqlite(self, dry_run: bool = False) -> Tuple[int, int]:
        """Sync memories from Cloudflare to SQLite-vec."""
        return await self._sync_between_backends('cloudflare', 'sqlite_vec', dry_run)

    async def sync_sqlite_to_cloudflare(self, dry_run: bool = False) -> Tuple[int, int]:
        """Sync memories from SQLite-vec to Cloudflare."""
        return await self._sync_between_backends('sqlite_vec', 'cloudflare', dry_run)

    async def bidirectional_sync(self, dry_run: bool = False) -> Dict[str, Tuple[int, int]]:
        """Perform bidirectional sync between backends."""
        logger.info("Starting bidirectional sync...")

        results = {}

        # Sync Cloudflare → SQLite-vec
        cf_to_sqlite = await self.sync_cloudflare_to_sqlite(dry_run)
        results['cloudflare_to_sqlite'] = cf_to_sqlite

        # Sync SQLite-vec → Cloudflare
        sqlite_to_cf = await self.sync_sqlite_to_cloudflare(dry_run)
        results['sqlite_to_cloudflare'] = sqlite_to_cf

        logger.info("Bidirectional sync completed")
        return results

    async def get_sync_status(self) -> Dict[str, Any]:
        """Get sync status showing memory counts in both backends."""
        cf_memories = await self.get_all_memories_from_backend('cloudflare')
        sqlite_memories = await self.get_all_memories_from_backend('sqlite_vec')

        status = {
            'cloudflare_count': len(cf_memories),
            'sqlite_vec_count': len(sqlite_memories),
            'sync_time': datetime.now().isoformat(),
            'backends_configured': {
                'cloudflare': bool(CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID),
                'sqlite_vec': os.path.exists(self.sqlite_path) if self.sqlite_path else False
            }
        }

        return status

async def main():
    """Main function to run memory sync operations."""
    parser = argparse.ArgumentParser(description='Sync memories between Cloudflare and SQLite-vec backends')
    parser.add_argument('--direction', choices=['cf-to-sqlite', 'sqlite-to-cf', 'bidirectional'],
                        default='bidirectional', help='Sync direction')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be synced without actually syncing')
    parser.add_argument('--status', action='store_true', help='Show sync status only')
    parser.add_argument('--sqlite-path', help='Path to SQLite-vec database file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize sync
    sync = MemorySync(sqlite_path=args.sqlite_path)

    try:
        if args.status:
            status = await sync.get_sync_status()
            print(f"\n=== Memory Sync Status ===")
            print(f"Cloudflare memories: {status['cloudflare_count']}")
            print(f"SQLite-vec memories: {status['sqlite_vec_count']}")
            print(f"Cloudflare configured: {status['backends_configured']['cloudflare']}")
            print(f"SQLite-vec file exists: {status['backends_configured']['sqlite_vec']}")
            print(f"Last check: {status['sync_time']}")
            return

        logger.info(f"=== Starting memory sync ({args.direction}) ===")
        if args.dry_run:
            logger.info("DRY RUN MODE - No changes will be made")

        if args.direction == 'cf-to-sqlite':
            added, skipped = await sync.sync_cloudflare_to_sqlite(dry_run=args.dry_run)
            print(f"Cloudflare → SQLite-vec: {added} added, {skipped} skipped")
        elif args.direction == 'sqlite-to-cf':
            added, skipped = await sync.sync_sqlite_to_cloudflare(dry_run=args.dry_run)
            print(f"SQLite-vec → Cloudflare: {added} added, {skipped} skipped")
        else:  # bidirectional
            results = await sync.bidirectional_sync(dry_run=args.dry_run)
            cf_to_sqlite = results['cloudflare_to_sqlite']
            sqlite_to_cf = results['sqlite_to_cloudflare']
            print(f"Cloudflare → SQLite-vec: {cf_to_sqlite[0]} added, {cf_to_sqlite[1]} skipped")
            print(f"SQLite-vec → Cloudflare: {sqlite_to_cf[0]} added, {sqlite_to_cf[1]} skipped")

        logger.info("=== Sync completed successfully ===")

    except Exception as e:
        logger.error(f"Sync failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
