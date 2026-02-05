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
Import memories from JSON exports into SQLite-vec database.

This script imports memories from one or more JSON export files into
a central SQLite-vec database, handling deduplication and preserving
original timestamps and metadata.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from mcp_memory_service.config import SQLITE_VEC_PATH, STORAGE_BACKEND
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.sync.importer import MemoryImporter

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


async def import_memories(
    json_files: list, db_path: Path, deduplicate: bool = True, add_source_tags: bool = True, dry_run: bool = False
):
    """Import memories from JSON files into database."""
    logger.info(f"Starting memory import to {db_path}")
    logger.info(f"JSON files: {[str(f) for f in json_files]}")

    # Validate input files
    for json_file in json_files:
        if not json_file.exists():
            logger.error(f"JSON file not found: {json_file}")
            return False

        # Quick validation of JSON format
        try:
            with open(json_file) as f:
                data = json.load(f)
                if "export_metadata" not in data or "memories" not in data:
                    logger.error(f"Invalid export format in {json_file}")
                    return False
        except Exception as e:
            logger.error(f"Error reading {json_file}: {str(e)}")
            return False

    try:
        # Initialize storage
        logger.info("Initializing SQLite-vec storage...")
        storage = SqliteVecMemoryStorage(str(db_path))
        await storage.initialize()

        # Create importer
        importer = MemoryImporter(storage)

        # Show analysis first
        logger.info("Analyzing import files...")
        analysis = await importer.analyze_import(json_files)

        logger.info("Import Analysis:")
        logger.info(f"  Total memories to process: {analysis['total_memories']}")
        logger.info(f"  Unique memories: {analysis['unique_memories']}")
        logger.info(f"  Potential duplicates: {analysis['potential_duplicates']}")
        logger.info(f"  Import conflicts: {len(analysis['conflicts'])}")

        logger.info("  Sources:")
        for source, stats in analysis["sources"].items():
            logger.info(f"    {source}: {stats['new_memories']}/{stats['total_memories']} new memories")

        if analysis["conflicts"]:
            logger.warning(f"Found {len(analysis['conflicts'])} conflicts between import files")

        # Ask for confirmation if not dry run
        if not dry_run:
            logger.info("")
            response = input("Proceed with import? (y/N): ")
            if response.lower() != "y":
                logger.info("Import cancelled by user")
                return False

        # Perform import
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Starting import...")
        result = await importer.import_from_json(
            json_files=json_files, deduplicate=deduplicate, add_source_tags=add_source_tags, dry_run=dry_run
        )

        # Show results
        logger.info(f"Import {'simulation ' if dry_run else ''}completed!")
        logger.info(f"  Files processed: {result['files_processed']}")
        logger.info(f"  Total processed: {result['total_processed']}")
        logger.info(f"  Successfully imported: {result['imported']}")
        logger.info(f"  Duplicates skipped: {result['duplicates_skipped']}")
        logger.info(f"  Errors: {result['errors']}")

        logger.info("  Source breakdown:")
        for source, stats in result["sources"].items():
            logger.info(f"    {source}: {stats['imported']}/{stats['total']} imported, {stats['duplicates']} duplicates")

        if not dry_run and result["imported"] > 0:
            # Show next steps
            logger.info("")
            logger.info("Next steps:")
            logger.info("1. Verify the imported memories using the web interface or API")
            logger.info("2. Set up Litestream for ongoing synchronization")
            logger.info("3. Configure replica nodes to sync from this central database")

        return result["errors"] == 0

    except Exception as e:
        logger.error(f"Import failed: {str(e)}")
        return False


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Import memories from JSON exports into SQLite-vec database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import single JSON file
  python import_memories.py windows_export.json

  # Import multiple JSON files
  python import_memories.py windows_export.json macbook_export.json

  # Import to specific database
  python import_memories.py --db-path /path/to/sqlite_vec.db exports/*.json

  # Dry run to see what would be imported
  python import_memories.py --dry-run exports/*.json

  # Import without deduplication (allow duplicates)
  python import_memories.py --no-deduplicate exports/*.json

  # Import without adding source tags
  python import_memories.py --no-source-tags exports/*.json
        """,
    )

    parser.add_argument("json_files", nargs="+", type=Path, help="JSON export files to import")

    parser.add_argument(
        "--db-path",
        type=Path,
        default=get_default_db_path(),
        help=f"Path to SQLite-vec database (default: {get_default_db_path()})",
    )

    parser.add_argument("--dry-run", action="store_true", help="Analyze imports without actually storing data")

    parser.add_argument(
        "--no-deduplicate", action="store_true", help="Allow duplicate memories (don't skip based on content hash)"
    )

    parser.add_argument("--no-source-tags", action="store_true", help="Don't add source machine tags to imported memories")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Show configuration
    logger.info("Memory Import Configuration:")
    logger.info(f"  Database: {args.db_path}")
    logger.info(f"  JSON files: {[str(f) for f in args.json_files]}")
    logger.info(f"  Dry run: {args.dry_run}")
    logger.info(f"  Deduplicate: {not args.no_deduplicate}")
    logger.info(f"  Add source tags: {not args.no_source_tags}")
    logger.info("")

    # Validate JSON files exist
    missing_files = [f for f in args.json_files if not f.exists()]
    if missing_files:
        logger.error(f"Missing JSON files: {missing_files}")
        sys.exit(1)

    # Run import
    success = await import_memories(
        json_files=args.json_files,
        db_path=args.db_path,
        deduplicate=not args.no_deduplicate,
        add_source_tags=not args.no_source_tags,
        dry_run=args.dry_run,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
