#!/usr/bin/env python3
"""
Repair script to fix existing SQLite-vec databases without full migration.

This script attempts to repair the database in-place by:
1. Checking the current state
2. Regenerating missing embeddings
3. Fixing dimension mismatches if possible
"""

import logging
import os
import sqlite3
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import sqlite_vec
    from sqlite_vec import serialize_float32

    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SqliteVecRepair:
    """Repair SQLite-vec database embeddings."""

    def __init__(self, db_path: str, model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.model_name = model_name
        self.conn = None
        self.model = None
        self.embedding_dimension = 384  # Default for all-MiniLM-L6-v2

    def check_dependencies(self):
        """Check required dependencies."""
        print("Checking dependencies...")

        if not SQLITE_VEC_AVAILABLE:
            print("❌ sqlite-vec not installed. Run: pip install sqlite-vec")
            return False

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("❌ sentence-transformers not installed. Run: pip install sentence-transformers torch")
            return False

        print("✅ All dependencies available")
        return True

    def connect_database(self):
        """Connect to the database."""
        print(f"\nConnecting to database: {self.db_path}")

        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        self.conn = sqlite3.connect(self.db_path)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

        print("✅ Connected to database")

    def analyze_database(self) -> dict:
        """Analyze the current state of the database."""
        print("\nAnalyzing database...")

        analysis = {"memory_count": 0, "embedding_count": 0, "missing_embeddings": 0, "embedding_dimension": None, "issues": []}

        # Count memories
        cursor = self.conn.execute("SELECT COUNT(*) FROM memories")
        analysis["memory_count"] = cursor.fetchone()[0]

        # Count embeddings
        cursor = self.conn.execute("SELECT COUNT(*) FROM memory_embeddings")
        analysis["embedding_count"] = cursor.fetchone()[0]

        # Check embedding dimension
        cursor = self.conn.execute("""
            SELECT sql FROM sqlite_master
            WHERE type='table' AND name='memory_embeddings'
        """)
        schema = cursor.fetchone()
        if schema:
            # Extract dimension from schema
            import re

            match = re.search(r"FLOAT\[(\d+)\]", schema[0])
            if match:
                analysis["embedding_dimension"] = int(match.group(1))

        # Find memories without embeddings
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM memories m
            WHERE NOT EXISTS (
                SELECT 1 FROM memory_embeddings e WHERE e.rowid = m.id
            )
        """)
        analysis["missing_embeddings"] = cursor.fetchone()[0]

        # Identify issues
        if analysis["memory_count"] != analysis["embedding_count"]:
            analysis["issues"].append(
                f"Mismatch: {analysis['memory_count']} memories vs {analysis['embedding_count']} embeddings"
            )

        if analysis["missing_embeddings"] > 0:
            analysis["issues"].append(f"Missing embeddings: {analysis['missing_embeddings']} memories have no embeddings")

        print(f"  Memories: {analysis['memory_count']}")
        print(f"  Embeddings: {analysis['embedding_count']}")
        print(f"  Missing embeddings: {analysis['missing_embeddings']}")
        print(f"  Embedding dimension: {analysis['embedding_dimension']}")

        if analysis["issues"]:
            print("\n⚠️  Issues found:")
            for issue in analysis["issues"]:
                print(f"  - {issue}")
        else:
            print("\n✅ No issues found")

        return analysis

    def load_model(self):
        """Load the embedding model."""
        print(f"\nLoading embedding model: {self.model_name}")

        self.model = SentenceTransformer(self.model_name)

        # Get actual dimension
        test_embedding = self.model.encode(["test"], convert_to_numpy=True)
        self.embedding_dimension = test_embedding.shape[1]

        print(f"✅ Model loaded (dimension: {self.embedding_dimension})")

    def generate_missing_embeddings(self, analysis: dict) -> int:
        """Generate embeddings for memories that don't have them."""
        if analysis["missing_embeddings"] == 0:
            return 0

        print(f"\nGenerating {analysis['missing_embeddings']} missing embeddings...")

        # Check if dimensions match
        if analysis["embedding_dimension"] and analysis["embedding_dimension"] != self.embedding_dimension:
            print(
                f"⚠️  WARNING: Database expects dimension {analysis['embedding_dimension']}, "
                f"but model produces dimension {self.embedding_dimension}"
            )
            print("   This may cause errors. Consider full migration instead.")
            response = input("\nContinue anyway? (y/N): ").strip().lower()
            if response != "y":
                return 0

        # Find memories without embeddings
        cursor = self.conn.execute("""
            SELECT m.id, m.content FROM memories m
            WHERE NOT EXISTS (
                SELECT 1 FROM memory_embeddings e WHERE e.rowid = m.id
            )
        """)

        memories_to_fix = cursor.fetchall()
        fixed_count = 0

        for memory_id, content in memories_to_fix:
            try:
                # Generate embedding
                embedding = self.model.encode([content], convert_to_numpy=True)[0]

                # Insert embedding
                self.conn.execute(
                    "INSERT INTO memory_embeddings (rowid, content_embedding) VALUES (?, ?)",
                    (memory_id, serialize_float32(embedding)),
                )

                fixed_count += 1

                # Show progress
                if fixed_count % 10 == 0:
                    print(f"  ... {fixed_count}/{len(memories_to_fix)} embeddings generated")

            except Exception as e:
                logger.error(f"Failed to generate embedding for memory {memory_id}: {e}")

        self.conn.commit()
        print(f"✅ Generated {fixed_count} embeddings")

        return fixed_count

    def verify_search(self) -> bool:
        """Test if semantic search works."""
        print("\nTesting semantic search...")

        try:
            # Generate a test query embedding
            test_query = "test query"
            query_embedding = self.model.encode([test_query], convert_to_numpy=True)[0]

            # Try to search
            cursor = self.conn.execute(
                """
                SELECT COUNT(*) FROM memory_embeddings
                WHERE content_embedding MATCH ?
                LIMIT 1
            """,
                (serialize_float32(query_embedding),),
            )

            cursor.fetchone()
            print("✅ Semantic search is working")
            return True

        except Exception as e:
            print(f"❌ Semantic search failed: {e}")
            return False

    def run_repair(self):
        """Run the repair process."""
        print("\n" + "=" * 60)
        print("SQLite-vec Embedding Repair Tool")
        print("=" * 60)

        try:
            # Check dependencies
            if not self.check_dependencies():
                return

            # Connect to database
            self.connect_database()

            # Analyze current state
            analysis = self.analyze_database()

            if not analysis["issues"]:
                print("\n✅ Database appears healthy, no repair needed")
                return

            # Load model
            self.load_model()

            # Fix missing embeddings
            fixed = self.generate_missing_embeddings(analysis)

            # Verify search works
            self.verify_search()

            # Re-analyze
            print("\nRe-analyzing database after repair...")
            new_analysis = self.analyze_database()

            print("\n" + "=" * 60)
            print("Repair Summary")
            print("=" * 60)
            print(f"Fixed {fixed} missing embeddings")

            if new_analysis["issues"]:
                print("\n⚠️  Some issues remain:")
                for issue in new_analysis["issues"]:
                    print(f"  - {issue}")
                print("\nConsider running the full migration script instead.")
            else:
                print("\n✅ All issues resolved!")

        except Exception as e:
            print(f"\n❌ Repair failed: {e}")
            logger.exception("Repair failed")

        finally:
            if self.conn:
                self.conn.close()


def main():
    """Run the repair tool."""
    if len(sys.argv) < 2:
        print("Usage: python repair_sqlite_vec_embeddings.py <database_path>")
        print("\nExample:")
        print("  python repair_sqlite_vec_embeddings.py ~/.mcp_memory/sqlite_vec.db")
        print("\nThis tool will:")
        print("  - Check for missing embeddings")
        print("  - Generate embeddings for memories that don't have them")
        print("  - Verify semantic search functionality")
        print("\nFor more complex issues (dimension mismatches, schema problems),")
        print("use migrate_sqlite_vec_embeddings.py instead.")
        sys.exit(1)

    db_path = sys.argv[1]

    repair = SqliteVecRepair(db_path)
    repair.run_repair()


if __name__ == "__main__":
    main()
