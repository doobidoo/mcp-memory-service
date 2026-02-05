#!/usr/bin/env python3
"""
Production-ready script to analyze timestamp health in MCP Memory Service databases.

This tool provides comprehensive timestamp analysis for SQLite-based memory storage,
helping identify and diagnose timestamp-related issues that could affect search functionality.
"""

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def analyze_timestamps(db_path: str, output_format: str = "text", verbose: bool = False) -> dict[str, Any]:
    """Analyze timestamp fields directly in the database.

    Args:
        db_path: Path to the SQLite database file
        output_format: Output format ('text', 'json', or 'summary')
        verbose: Enable verbose output

    Returns:
        Dictionary containing analysis results
    """
    results = {}

    if output_format == "text":
        print(f"=== Analyzing timestamps in {db_path} ===")

    # Validate database path
    db_file = Path(db_path)
    if not db_file.exists():
        error_msg = f"Database file not found: {db_path}"
        logger.error(error_msg)
        return {"error": error_msg, "success": False}

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # Get basic stats
        cursor = conn.execute("SELECT COUNT(*) as total FROM memories")
        total_count = cursor.fetchone()["total"]
        results["total_memories"] = total_count

        if output_format == "text":
            print(f"📊 Total memories in database: {total_count}")

        # Analyze timestamp fields
        cursor = conn.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(created_at) as has_created_at,
                COUNT(created_at_iso) as has_created_at_iso,
                COUNT(CASE WHEN created_at IS NULL AND created_at_iso IS NULL THEN 1 END) as missing_both,
                MIN(created_at) as earliest_ts,
                MAX(created_at) as latest_ts
            FROM memories
        """)

        stats = cursor.fetchone()

        # Store results
        results["timestamp_stats"] = {
            "total": stats["total"],
            "has_created_at": stats["has_created_at"],
            "has_created_at_iso": stats["has_created_at_iso"],
            "missing_both": stats["missing_both"],
            "missing_created_at": stats["total"] - stats["has_created_at"],
            "missing_created_at_iso": stats["total"] - stats["has_created_at_iso"],
        }

        if output_format == "text":
            print("\n🕐 TIMESTAMP ANALYSIS:")
            print(f"  Total entries: {stats['total']}")
            print(f"  Has created_at (float): {stats['has_created_at']}")
            print(f"  Has created_at_iso (ISO): {stats['has_created_at_iso']}")
            print(f"  Missing both timestamps: {stats['missing_both']}")

        if output_format == "text":
            if stats["has_created_at"] > 0:
                missing_created_at = stats["total"] - stats["has_created_at"]
                print(f"  Missing created_at: {missing_created_at}")

            if stats["has_created_at_iso"] > 0:
                missing_created_at_iso = stats["total"] - stats["has_created_at_iso"]
                print(f"  Missing created_at_iso: {missing_created_at_iso}")

        # Show timestamp range
        if stats["earliest_ts"] and stats["latest_ts"]:
            earliest = datetime.fromtimestamp(stats["earliest_ts"])
            latest = datetime.fromtimestamp(stats["latest_ts"])
            results["timestamp_range"] = {
                "earliest": earliest.isoformat(),
                "latest": latest.isoformat(),
                "earliest_float": stats["earliest_ts"],
                "latest_float": stats["latest_ts"],
            }

            if output_format == "text":
                print("\n📅 TIMESTAMP RANGE:")
                print(f"  Earliest: {earliest} ({stats['earliest_ts']})")
                print(f"  Latest: {latest} ({stats['latest_ts']})")

        # Find problematic entries
        cursor = conn.execute("""
            SELECT id, content_hash, created_at, created_at_iso,
                   SUBSTR(content, 1, 100) as content_preview
            FROM memories
            WHERE created_at IS NULL AND created_at_iso IS NULL
            LIMIT 10
        """)

        problematic = cursor.fetchall()
        results["missing_both_examples"] = len(problematic)

        if output_format == "text" and problematic:
            print(f"\n⚠️  ENTRIES MISSING BOTH TIMESTAMPS ({len(problematic)} shown):")
            for row in problematic:
                print(f"  ID {row['id']}: {row['content_preview']}...")
                if verbose:
                    print(f"    Hash: {row['content_hash']}")
                    print(f"    created_at: {row['created_at']}")
                    print(f"    created_at_iso: {row['created_at_iso']}")
                    print()

        # Find entries with only one timestamp type
        cursor = conn.execute("""
            SELECT COUNT(*) as count
            FROM memories
            WHERE (created_at IS NULL AND created_at_iso IS NOT NULL)
               OR (created_at IS NOT NULL AND created_at_iso IS NULL)
        """)

        partial_timestamps = cursor.fetchone()["count"]
        results["partial_timestamps"] = partial_timestamps

        if output_format == "text" and partial_timestamps > 0:
            print(f"\n⚠️  ENTRIES WITH PARTIAL TIMESTAMPS: {partial_timestamps}")

            # Show some examples
            cursor = conn.execute("""
                SELECT id, content_hash, created_at, created_at_iso,
                       SUBSTR(content, 1, 60) as content_preview
                FROM memories
                WHERE (created_at IS NULL AND created_at_iso IS NOT NULL)
                   OR (created_at IS NOT NULL AND created_at_iso IS NULL)
                LIMIT 5
            """)

            examples = cursor.fetchall()
            if output_format == "text" and verbose:
                for row in examples:
                    print(f"  ID {row['id']}: {row['content_preview']}...")
                    print(f"    created_at: {row['created_at']}")
                    print(f"    created_at_iso: {row['created_at_iso']}")
                    print()

        # Health assessment
        health_status = "EXCELLENT"
        health_message = "All memories have complete timestamps"

        if stats["missing_both"] > 0:
            if stats["missing_both"] < stats["total"] * 0.01:
                health_status = "GOOD"
                health_message = f"Only {stats['missing_both']}/{stats['total']} missing all timestamps"
            elif stats["missing_both"] < stats["total"] * 0.1:
                health_status = "WARNING"
                health_message = f"{stats['missing_both']}/{stats['total']} missing all timestamps"
            else:
                health_status = "CRITICAL"
                health_message = f"{stats['missing_both']}/{stats['total']} missing all timestamps"

        results["health"] = {"status": health_status, "message": health_message, "partial_timestamps": partial_timestamps}

        if output_format == "text":
            print("\n🏥 DATABASE HEALTH:")
            emoji = {"EXCELLENT": "✅", "GOOD": "✅", "WARNING": "⚠️", "CRITICAL": "❌"}
            print(f"  {emoji.get(health_status, '?')} {health_status}: {health_message}")

            if partial_timestamps > 0:
                print(f"  ⚠️  {partial_timestamps} entries have only partial timestamp data")
            else:
                print("  ✅ All entries with timestamps have both float and ISO formats")

        conn.close()
        results["success"] = True
        return results

    except sqlite3.OperationalError as e:
        if "no such table: memories" in str(e):
            error_msg = "Database does not contain 'memories' table. Is this a valid MCP Memory Service database?"
        else:
            error_msg = f"Database error: {e}"
        logger.error(error_msg)
        results["error"] = error_msg
        results["success"] = False
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        logger.error(error_msg)
        results["error"] = error_msg
        results["success"] = False
    finally:
        if "conn" in locals():
            conn.close()

    return results


def main():
    """Main entry point with CLI argument parsing."""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Analyze timestamp health in MCP Memory Service SQLite databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Use default database path
  %(prog)s /path/to/database.db              # Analyze specific database
  %(prog)s -f json -o results.json           # Output JSON to file
  %(prog)s --verbose                         # Show detailed analysis
  %(prog)s --format summary                  # Quick health check only
        """,
    )

    # Default database path for macOS
    default_db_path = Path.home() / "Library" / "Application Support" / "mcp-memory" / "sqlite_vec.db"

    parser.add_argument(
        "database", nargs="?", default=str(default_db_path), help=f"Path to SQLite database (default: {default_db_path})"
    )

    parser.add_argument(
        "-f", "--format", choices=["text", "json", "summary"], default="text", help="Output format (default: text)"
    )

    parser.add_argument("-o", "--output", help="Output file path (default: stdout)")

    parser.add_argument("-v", "--verbose", action="store_true", help="Show verbose output with additional details")

    parser.add_argument("--quiet", action="store_true", help="Suppress all output except errors")

    args = parser.parse_args()

    # Configure logging based on verbosity
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Analyze the database
    results = analyze_timestamps(args.database, args.format, args.verbose)

    # Handle output
    if args.format == "json":
        output = json.dumps(results, indent=2, default=str)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            if not args.quiet:
                print(f"Results written to {args.output}")
        else:
            print(output)
    elif args.format == "summary":
        if results.get("success"):
            health = results.get("health", {})
            print(f"Status: {health.get('status', 'UNKNOWN')}")
            print(f"Message: {health.get('message', 'No health data')}")
            print(f"Total Memories: {results.get('total_memories', 0)}")
            missing = results.get("timestamp_stats", {}).get("missing_both", 0)
            if missing > 0:
                print(f"Missing Timestamps: {missing}")
        else:
            print(f"Error: {results.get('error', 'Unknown error')}")

    # Return appropriate exit code
    if results.get("success"):
        health_status = results.get("health", {}).get("status", "UNKNOWN")
        if health_status in ["EXCELLENT", "GOOD"]:
            sys.exit(0)
        elif health_status == "WARNING":
            sys.exit(1)
        else:
            sys.exit(2)
    else:
        sys.exit(3)


if __name__ == "__main__":
    main()
