#!/usr/bin/env python3
"""
Check for memories without timestamps in the MCP memory service database.
This script analyzes the storage backend for entries missing timestamp data.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asyncio
import json
from datetime import datetime
from typing import Any

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


class TimestampAnalyzer:
    """Analyze memory database for missing timestamp entries."""

    def __init__(self, storage_backend: str = "sqlite_vec", db_path: str = None):
        self.storage_backend = storage_backend
        self.db_path = db_path or os.path.expanduser("~/.mcp_memory_service/storage.db")
        self.storage = None

    async def setup(self):
        """Initialize storage backend."""
        print(f"=== Analyzing {self.storage_backend} database for timestamp issues ===")
        print(f"Database path: {self.db_path}")

        if not os.path.exists(self.db_path):
            print(f"❌ Database file not found: {self.db_path}")
            print("Possible locations:")
            common_paths = [
                os.path.expanduser("~/.mcp_memory_service/storage.db"),
                os.path.expanduser("~/.mcp_memory_service/memory.db"),
                "storage.db",
                "memory.db",
            ]
            for path in common_paths:
                if os.path.exists(path):
                    print(f"  ✅ Found: {path}")
                else:
                    print(f"  ❌ Not found: {path}")
            return False

        try:
            self.storage = SqliteVecMemoryStorage(db_path=self.db_path, embedding_model="all-MiniLM-L6-v2")
            await self.storage.initialize()
            print("✅ Storage initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize storage: {e}")
            return False

    async def get_all_memories(self) -> list[Memory]:
        """Retrieve all memories from the database."""
        try:
            # SQLite-Vec has a limit of 4096 for k in knn queries, so use direct database access
            return await self.get_memories_direct_query()
        except Exception as e:
            print(f"❌ Error with direct query, trying search approach: {e}")
            return await self.get_memories_via_search()

    async def get_memories_direct_query(self) -> list[Memory]:
        """Get all memories using direct database queries."""
        import sqlite3

        memories = []

        try:
            # Connect directly to SQLite database
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row

            # Get all memory records
            cursor = conn.execute("""
                SELECT id, content, content_hash, tags, memory_type,
                       created_at, created_at_iso, updated_at, updated_at_iso, metadata
                FROM memories
                ORDER BY created_at DESC
            """)

            rows = cursor.fetchall()
            print(f"📊 Found {len(rows)} memory records in database")

            for i, row in enumerate(rows):
                try:
                    # Safely parse JSON fields
                    tags = []
                    if row["tags"]:
                        try:
                            tags = json.loads(row["tags"])
                        except (json.JSONDecodeError, TypeError):
                            tags = []

                    metadata = {}
                    if row["metadata"]:
                        try:
                            metadata = json.loads(row["metadata"])
                        except (json.JSONDecodeError, TypeError):
                            metadata = {}

                    # Reconstruct Memory object from database row
                    memory_dict = {
                        "content": row["content"] or "",
                        "content_hash": row["content_hash"] or "",
                        "tags": tags,
                        "memory_type": row["memory_type"] or "unknown",
                        "created_at": row["created_at"],
                        "created_at_iso": row["created_at_iso"],
                        "updated_at": row["updated_at"],
                        "updated_at_iso": row["updated_at_iso"],
                        "metadata": metadata,
                    }

                    memory = Memory.from_dict(memory_dict)
                    memories.append(memory)

                except Exception as e:
                    print(f"⚠️  Error processing memory {i + 1}: {e}")
                    # Continue processing other memories
                    continue

            conn.close()
            return memories

        except Exception as e:
            print(f"❌ Direct query failed: {e}")
            if "conn" in locals():
                conn.close()
            return []

    async def get_memories_via_search(self) -> list[Memory]:
        """Get memories using search with smaller batches."""
        memories = []

        try:
            # Try different search approaches with smaller limits
            search_queries = ["", "memory", "note", "session"]

            for query in search_queries:
                try:
                    results = await self.storage.retrieve(query, n_results=1000)  # Well under 4096 limit
                    batch_memories = [result.memory for result in results]

                    # Deduplicate based on content_hash
                    existing_hashes = {m.content_hash for m in memories}
                    new_memories = [m for m in batch_memories if m.content_hash not in existing_hashes]
                    memories.extend(new_memories)

                    print(f"📊 Query '{query}': {len(batch_memories)} results, {len(new_memories)} new")

                except Exception as e:
                    print(f"⚠️  Query '{query}' failed: {e}")
                    continue

            print(f"📊 Total unique memories retrieved: {len(memories)}")
            return memories

        except Exception as e:
            print(f"❌ All search approaches failed: {e}")
            return []

    def analyze_timestamp_fields(self, memories: list[Memory]) -> dict[str, Any]:
        """Analyze timestamp fields across all memories."""
        analysis = {
            "total_memories": len(memories),
            "missing_created_at": 0,
            "missing_created_at_iso": 0,
            "missing_both_timestamps": 0,
            "invalid_timestamps": 0,
            "problematic_memories": [],
            "timestamp_formats": set(),
            "timestamp_range": {"earliest": None, "latest": None},
        }

        for memory in memories:
            has_created_at = memory.created_at is not None
            has_created_at_iso = memory.created_at_iso is not None

            # Track missing timestamp fields
            if not has_created_at:
                analysis["missing_created_at"] += 1

            if not has_created_at_iso:
                analysis["missing_created_at_iso"] += 1

            if not has_created_at and not has_created_at_iso:
                analysis["missing_both_timestamps"] += 1
                analysis["problematic_memories"].append(
                    {
                        "content_hash": memory.content_hash,
                        "content_preview": memory.content[:100] + "..." if len(memory.content) > 100 else memory.content,
                        "tags": memory.tags,
                        "memory_type": memory.memory_type,
                        "issue": "missing_both_timestamps",
                    }
                )

            # Track timestamp formats and ranges
            if has_created_at_iso:
                analysis["timestamp_formats"].add(type(memory.created_at_iso).__name__)

            if has_created_at:
                try:
                    if (
                        analysis["timestamp_range"]["earliest"] is None
                        or memory.created_at < analysis["timestamp_range"]["earliest"]
                    ):
                        analysis["timestamp_range"]["earliest"] = memory.created_at
                    if (
                        analysis["timestamp_range"]["latest"] is None
                        or memory.created_at > analysis["timestamp_range"]["latest"]
                    ):
                        analysis["timestamp_range"]["latest"] = memory.created_at
                except:
                    analysis["invalid_timestamps"] += 1
                    analysis["problematic_memories"].append(
                        {
                            "content_hash": memory.content_hash,
                            "content_preview": memory.content[:100] + "..." if len(memory.content) > 100 else memory.content,
                            "created_at": str(memory.created_at),
                            "issue": "invalid_timestamp",
                        }
                    )

        # Convert set to list for JSON serialization
        analysis["timestamp_formats"] = list(analysis["timestamp_formats"])

        return analysis

    def print_analysis_report(self, analysis: dict[str, Any]):
        """Print a detailed analysis report."""
        print("\n" + "=" * 70)
        print("TIMESTAMP ANALYSIS REPORT")
        print("=" * 70)

        total = analysis["total_memories"]

        print("\n📊 OVERVIEW:")
        print(f"  Total memories analyzed: {total}")
        print(f"  Missing created_at (float): {analysis['missing_created_at']}")
        print(f"  Missing created_at_iso (ISO string): {analysis['missing_created_at_iso']}")
        print(f"  Missing both timestamps: {analysis['missing_both_timestamps']}")
        print(f"  Invalid timestamp values: {analysis['invalid_timestamps']}")

        if total > 0:
            print("\n📈 PERCENTAGES:")
            print(f"  Missing created_at: {analysis['missing_created_at'] / total * 100:.1f}%")
            print(f"  Missing created_at_iso: {analysis['missing_created_at_iso'] / total * 100:.1f}%")
            print(f"  Missing both: {analysis['missing_both_timestamps'] / total * 100:.1f}%")
            print(f"  Invalid timestamps: {analysis['invalid_timestamps'] / total * 100:.1f}%")

        print("\n🕐 TIMESTAMP RANGE:")
        if analysis["timestamp_range"]["earliest"] and analysis["timestamp_range"]["latest"]:
            earliest = datetime.fromtimestamp(analysis["timestamp_range"]["earliest"])
            latest = datetime.fromtimestamp(analysis["timestamp_range"]["latest"])
            print(f"  Earliest: {earliest} ({analysis['timestamp_range']['earliest']})")
            print(f"  Latest: {latest} ({analysis['timestamp_range']['latest']})")
        else:
            print("  No valid timestamps found")

        print("\n📝 TIMESTAMP FORMATS DETECTED:")
        for fmt in analysis["timestamp_formats"]:
            print(f"  - {fmt}")

        if analysis["problematic_memories"]:
            print(f"\n⚠️  PROBLEMATIC MEMORIES ({len(analysis['problematic_memories'])}):")
            for i, memory in enumerate(analysis["problematic_memories"][:10]):  # Show first 10
                print(f"  {i + 1}. Issue: {memory['issue']}")
                print(f"     Content: {memory['content_preview']}")
                print(f"     Hash: {memory['content_hash']}")
                if "tags" in memory:
                    print(f"     Tags: {memory.get('tags', [])}")
                print()

            if len(analysis["problematic_memories"]) > 10:
                print(f"  ... and {len(analysis['problematic_memories']) - 10} more")

        # Health assessment
        print("\n🏥 DATABASE HEALTH ASSESSMENT:")
        if analysis["missing_both_timestamps"] == 0:
            print("  ✅ EXCELLENT: All memories have at least one timestamp field")
        elif analysis["missing_both_timestamps"] < total * 0.1:
            print(f"  ⚠️  GOOD: Only {analysis['missing_both_timestamps']} memories missing all timestamps")
        elif analysis["missing_both_timestamps"] < total * 0.5:
            print(f"  ⚠️  CONCERNING: {analysis['missing_both_timestamps']} memories missing all timestamps")
        else:
            print(f"  ❌ CRITICAL: {analysis['missing_both_timestamps']} memories missing all timestamps")

        if analysis["missing_created_at"] > 0 or analysis["missing_created_at_iso"] > 0:
            print("  💡 RECOMMENDATION: Run timestamp migration script to fix missing fields")

    async def run_analysis(self):
        """Run the complete timestamp analysis."""
        if not await self.setup():
            return False

        memories = await self.get_all_memories()
        if not memories:
            print("⚠️  No memories found in database")
            return False

        analysis = self.analyze_timestamp_fields(memories)
        self.print_analysis_report(analysis)

        # Save detailed report to file
        report_file = "timestamp_analysis_report.json"
        with open(report_file, "w") as f:
            # Convert any datetime objects to strings for JSON serialization
            json_analysis = analysis.copy()
            if json_analysis["timestamp_range"]["earliest"]:
                json_analysis["timestamp_range"]["earliest_iso"] = datetime.fromtimestamp(
                    json_analysis["timestamp_range"]["earliest"]
                ).isoformat()
            if json_analysis["timestamp_range"]["latest"]:
                json_analysis["timestamp_range"]["latest_iso"] = datetime.fromtimestamp(
                    json_analysis["timestamp_range"]["latest"]
                ).isoformat()

            json.dump(json_analysis, f, indent=2, default=str)

        print(f"\n📄 Detailed report saved to: {report_file}")

        return analysis["missing_both_timestamps"] == 0


async def main():
    """Main analysis execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Check for memories without timestamps")
    parser.add_argument("--db-path", help="Path to database file")
    parser.add_argument("--storage", default="sqlite_vec", choices=["sqlite_vec"], help="Storage backend to analyze")

    args = parser.parse_args()

    analyzer = TimestampAnalyzer(storage_backend=args.storage, db_path=args.db_path)

    success = await analyzer.run_analysis()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
