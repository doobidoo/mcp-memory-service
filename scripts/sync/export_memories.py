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
Export memories from SQLite-vec database to JSON format.

This script exports all memories from a local SQLite-vec database,
preserving timestamps, metadata, and adding source tracking for
multi-machine synchronization.
"""

import argparse
import asyncio
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path

# Add project src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from mcp_memory_service.config import SQLITE_VEC_PATH, STORAGE_BACKEND
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.sync.exporter import MemoryExporter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_default_db_path() -> Path:
    """Get the default database path for this platform."""
    if STORAGE_BACKEND == "sqlite_vec" and SQLITE_VEC_PATH:
        return Path(SQLITE_VEC_PATH)
    else:
        # Fallback to BASE_DIR if not using sqlite_vec backend
        from mcp_memory_service.config import BASE_DIR

        return Path(BASE_DIR) / "sqlite_vec.db"


def get_default_output_filename() -> str:
    """Generate a default output filename based on machine and timestamp."""
    machine_name = platform.node().lower().replace(" ", "-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{machine_name}_memories_export_{timestamp}.json"


async def export_memories(db_path: Path, output_file: Path, include_embeddings: bool = False, filter_tags: list = None):
    """Export memories from database to JSON file."""
    logger.info(f"Starting memory export from {db_path}")

    # Check if database exists
    if not db_path.exists():
        logger.error(f"Database not found at {db_path}")
        logger.info("Available storage locations:")
        logger.info(f"  Default: {get_default_db_path()}")
        return False

    try:
        # Initialize storage
        logger.info("Initializing SQLite-vec storage...")
        storage = SqliteVecMemoryStorage(str(db_path))
        await storage.initialize()

        # Create exporter
        exporter = MemoryExporter(storage)

        # Show summary first
        logger.info("Analyzing database...")
        summary = await exporter.export_summary()

        logger.info("Database analysis:")
        logger.info(f"  Total memories: {summary['total_memories']}")
        logger.info(f"  Machine: {summary['machine_name']}")
        logger.info(f"  Date range: {summary['date_range']['earliest']} to {summary['date_range']['latest']}")
        logger.info(f"  Memory types: {summary['memory_types']}")
        logger.info(f"  Top tags: {list(summary['tag_counts'].items())[:5]}")
        logger.info(f"  Estimated size: {summary['estimated_json_size_mb']:.1f} MB")

        # Perform export
        logger.info(f"Exporting to {output_file}...")
        result = await exporter.export_to_json(
            output_file=output_file, include_embeddings=include_embeddings, filter_tags=filter_tags
        )

        if result["success"]:
            logger.info("Export completed successfully!")
            logger.info(f"  Exported: {result['exported_count']} memories")
            logger.info(f"  Output file: {result['output_file']}")
            logger.info(f"  File size: {result['file_size_bytes'] / 1024 / 1024:.2f} MB")
            logger.info(f"  Source machine: {result['source_machine']}")

            # Show next steps
            logger.info("")
            logger.info("Next steps:")
            logger.info("1. Transfer this JSON file to your central server")
            logger.info("2. Run import_memories.py on the central server")
            logger.info("3. Set up Litestream for ongoing synchronization")

            return True
        else:
            logger.error("Export failed")
            return False

    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        return False


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Export memories from SQLite-vec database to JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export all memories with default settings
  python export_memories.py

  # Export from specific database
  python export_memories.py --db-path /path/to/sqlite_vec.db

  # Export to specific file
  python export_memories.py --output my_export.json

  # Export only memories with specific tags
  python export_memories.py --filter-tags claude-code,architecture

  # Include embedding vectors (increases file size significantly)
  python export_memories.py --include-embeddings
        """,
    )

    parser.add_argument(
        "--db-path",
        type=Path,
        default=get_default_db_path(),
        help=f"Path to SQLite-vec database (default: {get_default_db_path()})",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=get_default_output_filename(),
        help=f"Output JSON file (default: {get_default_output_filename()})",
    )

    parser.add_argument(
        "--include-embeddings", action="store_true", help="Include embedding vectors in export (increases file size)"
    )

    parser.add_argument("--filter-tags", nargs="*", help="Only export memories with these tags")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Show configuration
    logger.info("Memory Export Configuration:")
    logger.info(f"  Database: {args.db_path}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Include embeddings: {args.include_embeddings}")
    logger.info(f"  Filter tags: {args.filter_tags}")
    logger.info(f"  Platform: {platform.system()} {platform.release()}")
    logger.info("")

    # Run export
    success = await export_memories(
        db_path=args.db_path, output_file=args.output, include_embeddings=args.include_embeddings, filter_tags=args.filter_tags
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
