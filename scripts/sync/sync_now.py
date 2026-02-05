#!/usr/bin/env python3
"""
Manual sync utility for Hybrid Storage Backend.
Triggers an immediate sync from SQLite-vec to Cloudflare.

Usage:
    python sync_now.py [--db-path PATH] [--verbose]

Environment Variables:
    MCP_MEMORY_SQLITE_PATH: Override default database path
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import TypedDict

try:
    from dotenv import load_dotenv

    from mcp_memory_service.config import SQLITE_VEC_PATH
    from mcp_memory_service.storage.factory import create_storage_instance
    from mcp_memory_service.storage.hybrid import HybridMemoryStorage
except ImportError as e:
    print(f"❌ Import error: {e}", file=sys.stderr)
    print("Make sure the package is installed: pip install -e .", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class SyncResult(TypedDict, total=False):
    """Type-safe structure for sync operation results."""

    status: str
    synced_to_secondary: int
    duration: float
    failed: int
    error: str


class SyncStatus(TypedDict, total=False):
    """Type-safe structure for sync status information."""

    queue_size: int
    cloudflare_available: bool
    failed_operations: int


async def main(db_path: str | None = None, verbose: bool = False) -> int:
    """
    Run immediate sync.

    Args:
        db_path: Optional path to SQLite database. If not provided,
                uses MCP_MEMORY_SQLITE_PATH env var or default config.
        verbose: Enable verbose error reporting with full tracebacks.

    Returns:
        0 on success, 1 on failure
    """
    # Load environment variables
    load_dotenv()

    logger.info("🔄 Starting manual sync...")

    # Determine database path
    sqlite_path = Path(db_path or os.getenv("MCP_MEMORY_SQLITE_PATH") or SQLITE_VEC_PATH)

    if not sqlite_path.exists():
        logger.error(f"❌ Database not found: {sqlite_path}")
        return 1

    logger.info(f"📁 Using database: {sqlite_path}")

    # Create storage instance
    try:
        storage = await create_storage_instance(str(sqlite_path))
    except (ValueError, RuntimeError, FileNotFoundError, OSError) as e:
        logger.error(f"❌ Failed to create storage instance: {e}")
        if verbose:
            logger.exception("Full traceback for failed storage instance creation:")
        return 1

    # Type-safe check for hybrid storage
    if not isinstance(storage, HybridMemoryStorage):
        logger.error("❌ Not a hybrid backend - sync not available")
        logger.error(f"   Found: {storage.__class__.__name__}")
        return 1

    # Get sync status before
    try:
        status_before = await storage.get_sync_status()
        logger.info("📊 Before sync:")
        logger.info(f"   Queue size: {status_before['queue_size']}")
        logger.info(f"   Cloudflare available: {status_before['cloudflare_available']}")
    except Exception as e:
        logger.warning(f"⚠️  Could not get sync status: {e}")

    # Trigger immediate sync
    logger.info("\n⏳ Triggering sync...")
    try:
        result = await storage.force_sync()

        # Check sync status
        if result.get("status") != "completed":
            logger.error(f"❌ Sync failed with status: {result.get('status')}")
            if result.get("error"):
                logger.error(f"   Error: {result.get('error')}")
            return 1

        logger.info("✅ Sync completed successfully!")
        logger.info(f"   Synced: {result.get('synced_to_secondary', 0)} operations")
        logger.info(f"   Duration: {result.get('duration', 0):.2f}s")

        # Report any failed operations
        if result.get("failed", 0) > 0:
            logger.warning(f"   ⚠️  Failed operations: {result.get('failed', 0)}")
    except Exception as e:
        logger.error(f"❌ Sync failed: {e}")
        if verbose:
            logger.exception("Full traceback for sync failure:")
        return 1

    # Get sync status after
    try:
        status_after = await storage.get_sync_status()
        logger.info("\n📊 After sync:")
        logger.info(f"   Queue size: {status_after['queue_size']}")
        logger.info(f"   Failed operations: {status_after['failed_operations']}")
    except Exception as e:
        logger.warning(f"⚠️  Could not get final sync status: {e}")

    return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Manual sync utility for Hybrid Storage Backend")
    parser.add_argument("--db-path", type=str, help="Path to SQLite database (default: from config or env)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose error reporting with full tracebacks")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exit_code = asyncio.run(main(db_path=args.db_path, verbose=args.verbose))
    sys.exit(exit_code)
