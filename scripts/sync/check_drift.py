#!/usr/bin/env python3
"""
Check for metadata drift between hybrid backends (dry-run support).

This script checks for memories with divergent metadata (tags, types, custom fields)
between local SQLite-vec and Cloudflare backends, without making any changes.

Usage:
    python scripts/sync/check_drift.py              # Dry-run mode (preview only)
    python scripts/sync/check_drift.py --apply      # Apply changes
    python scripts/sync/check_drift.py --limit 50   # Check 50 memories max

Requires:
    - Hybrid storage backend configured
    - MCP_HYBRID_SYNC_UPDATES=true (or enabled by default)

Output:
    - Number of memories checked
    - Number with metadata drift detected
    - Number that would be synced (or were synced with --apply)
    - Number of failures

Version: 8.25.0+
"""

import asyncio
import logging
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp_memory_service.storage.hybrid import HybridMemoryStorage
from mcp_memory_service import config as app_config
from mcp_memory_service.compat import _sanitize_log_value

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_cloudflare_config():
    """Build a Cloudflare config dict from the app configuration."""
    cloudflare_keys = [
        'CLOUDFLARE_API_TOKEN',
        'CLOUDFLARE_ACCOUNT_ID',
        'CLOUDFLARE_D1_DATABASE_ID',
        'CLOUDFLARE_VECTORIZE_INDEX',
        'CLOUDFLARE_R2_BUCKET',
        'CLOUDFLARE_EMBEDDING_MODEL',
        'CLOUDFLARE_LARGE_CONTENT_THRESHOLD',
        'CLOUDFLARE_MAX_RETRIES',
        'CLOUDFLARE_BASE_DELAY',
    ]
    return {
        key.lower().replace('cloudflare_', ''): getattr(app_config, key, None)
        for key in cloudflare_keys
    }


def print_drift_results(stats, apply):
    """Print the drift-detection result summary."""
    print("\n" + "=" * 60)
    print(f"DRIFT DETECTION RESULTS {'(DRY RUN)' if not apply else '(CHANGES APPLIED)'}")
    print("=" * 60)
    print(f"  Memories checked:    {stats['checked']}")
    print(f"  Drift detected:      {stats['drift_detected']}")
    print(f"  {'Would sync' if not apply else 'Synced'}:          {stats['synced']}")
    print(f"  Failed:              {stats['failed']}")
    print("=" * 60)

    if stats['drift_detected'] > 0 and not apply:
        print("\nℹ️  Run with --apply to synchronize these memories")
    elif stats['drift_detected'] > 0 and apply:
        print("\n✅ Metadata synchronized successfully")
    else:
        print("\n✅ No drift detected - backends are in sync")


async def main():
    """Run drift detection check."""
    parser = argparse.ArgumentParser(
        description="Check for metadata drift between hybrid backends"
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Apply changes (default is dry-run mode)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of memories to check (default: from config)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose debug logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check that hybrid backend is configured
    if app_config.STORAGE_BACKEND != 'hybrid':
        logger.error("Drift detection requires hybrid backend, but configured backend is: %s", _sanitize_log_value(f"{app_config.STORAGE_BACKEND}"))
        logger.error("Set MCP_MEMORY_STORAGE_BACKEND=hybrid in your environment or .env file")
        return 1

    # Override batch size if limit specified
    if args.limit:
        app_config.HYBRID_DRIFT_BATCH_SIZE = args.limit

    logger.info("=== Hybrid Backend Drift Detection ===")
    logger.info("Mode: %s", _sanitize_log_value(f"{'APPLY CHANGES' if args.apply else 'DRY RUN (preview only)'}"))
    logger.info("Batch size: %s", _sanitize_log_value(f"{args.limit or app_config.HYBRID_DRIFT_BATCH_SIZE}"))
    logger.info("Drift detection enabled: %s", _sanitize_log_value(f"{app_config.HYBRID_SYNC_UPDATES}"))

    if not app_config.HYBRID_SYNC_UPDATES:
        logger.warning("Drift detection is disabled (MCP_HYBRID_SYNC_UPDATES=false)")
        logger.warning("Set MCP_HYBRID_SYNC_UPDATES=true to enable this feature")
        return 1

    try:
        # Initialize hybrid storage with db path and Cloudflare config
        db_path = app_config.SQLITE_VEC_PATH

        # Build Cloudflare config from environment
        cloudflare_config = build_cloudflare_config()

        storage = HybridMemoryStorage(
            sqlite_db_path=db_path,
            cloudflare_config=cloudflare_config
        )
        await storage.initialize()

        # Check that sync service is available
        if not storage.sync_service:
            logger.error("Sync service not available - hybrid backend may not be configured correctly")
            return 1

        logger.info("Sync service initialized (drift check interval: %ss)", _sanitize_log_value(f"{storage.sync_service.drift_check_interval}"))

        # Run drift detection
        logger.info("\nStarting drift detection scan...\n")
        stats = await storage.sync_service._detect_and_sync_drift(dry_run=not args.apply)

        # Print results
        print_drift_results(stats, args.apply)

        return 0

    except Exception as e:
        logger.error("Error during drift detection: %s", _sanitize_log_value(f"{e}"), exc_info=True)
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
