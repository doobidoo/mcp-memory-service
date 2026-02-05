#!/usr/bin/env python3
"""
Enhanced repair script to fix zero vector embeddings in SQLite-vec databases.

This script detects and repairs embeddings that are all zeros (invalid) and
regenerates them with proper sentence transformer embeddings.
"""

import logging
import os
import sqlite3
import sys

import numpy as np

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


class ZeroEmbeddingRepair:
    """Repair SQLite-vec database with zero vector embeddings."""

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

        # Connect with timeout to handle concurrent access
        self.conn = sqlite3.connect(self.db_path, timeout=120.0)

        # Apply critical pragmas immediately for concurrent access support
        self.conn.execute("PRAGMA busy_timeout=120000")
        self.conn.execute("PRAGMA journal_mode=WAL")

        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

        print("✅ Connected to database")

    def analyze_database(self) -> dict:
        """Analyze the current state of the database including zero embeddings."""
        print("\nAnalyzing database...")

        analysis = {
            "memory_count": 0,
            "embedding_count": 0,
            "missing_embeddings": 0,
            "zero_embeddings": 0,
            "valid_embeddings": 0,
            "embedding_dimension": None,
            "issues": [],
        }

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
            import re

            match = re.search(r"FLOAT\\[(\\d+)\\]", schema[0])
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

        # Check for zero embeddings
        print("  Checking for zero vector embeddings...")
        cursor = self.conn.execute("""
            SELECT e.rowid, e.content_embedding, m.content
            FROM memory_embeddings e
            INNER JOIN memories m ON m.id = e.rowid
        """)

        zero_count = 0
        valid_count = 0

        for row in cursor.fetchall():
            rowid, embedding_blob, content = row

            if embedding_blob and len(embedding_blob) > 0:
                try:
                    # Convert to numpy array
                    embedding_array = np.frombuffer(embedding_blob, dtype=np.float32)

                    # Check if all zeros
                    if np.allclose(embedding_array, 0):
                        zero_count += 1
                        logger.debug(f"Zero embedding found for memory {rowid}: {content[:50]}...")
                    else:
                        valid_count += 1

                except Exception as e:
                    logger.warning(f"Failed to parse embedding for rowid {rowid}: {e}")
                    zero_count += 1  # Treat unparseable as invalid
            else:
                zero_count += 1

        analysis["zero_embeddings"] = zero_count
        analysis["valid_embeddings"] = valid_count

        # Identify issues
        if analysis["memory_count"] != analysis["embedding_count"]:
            analysis["issues"].append(
                f"Mismatch: {analysis['memory_count']} memories vs {analysis['embedding_count']} embeddings"
            )

        if analysis["missing_embeddings"] > 0:
            analysis["issues"].append(f"Missing embeddings: {analysis['missing_embeddings']} memories have no embeddings")

        if analysis["zero_embeddings"] > 0:
            analysis["issues"].append(
                f"Zero vector embeddings: {analysis['zero_embeddings']} embeddings are all zeros (invalid)"
            )

        print(f"  Memories: {analysis['memory_count']}")
        print(f"  Embeddings: {analysis['embedding_count']}")
        print(f"  Missing embeddings: {analysis['missing_embeddings']}")
        print(f"  Zero embeddings: {analysis['zero_embeddings']}")
        print(f"  Valid embeddings: {analysis['valid_embeddings']}")
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

    def regenerate_zero_embeddings(self, analysis: dict) -> int:
        """Regenerate embeddings that are zero vectors."""
        if analysis["zero_embeddings"] == 0:
            return 0

        print(f"\nRegenerating {analysis['zero_embeddings']} zero vector embeddings...")

        # Get all memories with zero embeddings
        cursor = self.conn.execute("""
            SELECT e.rowid, e.content_embedding, m.content
            FROM memory_embeddings e
            INNER JOIN memories m ON m.id = e.rowid
        """)

        zero_embeddings = []
        for row in cursor.fetchall():
            rowid, embedding_blob, content = row

            if embedding_blob and len(embedding_blob) > 0:
                try:
                    embedding_array = np.frombuffer(embedding_blob, dtype=np.float32)
                    if np.allclose(embedding_array, 0):
                        zero_embeddings.append((rowid, content))
                except:
                    zero_embeddings.append((rowid, content))
            else:
                zero_embeddings.append((rowid, content))

        fixed_count = 0

        for rowid, content in zero_embeddings:
            try:
                # Generate new embedding
                embedding = self.model.encode([content], convert_to_numpy=True)[0]

                # Validate the new embedding
                if not np.allclose(embedding, 0) and np.isfinite(embedding).all():
                    # Update the embedding in database
                    self.conn.execute(
                        "UPDATE memory_embeddings SET content_embedding = ? WHERE rowid = ?",
                        (serialize_float32(embedding), rowid),
                    )

                    fixed_count += 1

                    # Show progress
                    if fixed_count % 10 == 0:
                        print(f"  ... {fixed_count}/{len(zero_embeddings)} embeddings regenerated")

                else:
                    logger.error(f"Generated invalid embedding for memory {rowid}")

            except Exception as e:
                logger.error(f"Failed to regenerate embedding for memory {rowid}: {e}")

        self.conn.commit()
        print(f"✅ Regenerated {fixed_count} embeddings")

        return fixed_count

    def verify_search(self) -> bool:
        """Test if semantic search works with proper similarity scores."""
        print("\nTesting semantic search with similarity scores...")

        try:
            # Generate a test query embedding
            test_query = "test embedding verification"
            query_embedding = self.model.encode([test_query], convert_to_numpy=True)[0]

            # Try to search and get distances
            cursor = self.conn.execute(
                """
                SELECT m.content, e.distance
                FROM memories m
                INNER JOIN (
                    SELECT rowid, distance
                    FROM memory_embeddings
                    WHERE content_embedding MATCH ?
                    ORDER BY distance
                    LIMIT 3
                ) e ON m.id = e.rowid
                ORDER BY e.distance
            """,
                (serialize_float32(query_embedding),),
            )

            results = cursor.fetchall()

            if results:
                print("✅ Semantic search working with results:")
                for i, (content, distance) in enumerate(results, 1):
                    similarity = max(0.0, 1.0 - distance)
                    print(f"  {i}. Distance: {distance:.6f}, Similarity: {similarity:.6f}")
                    print(f"     Content: {content[:60]}...")

                # Check if we have reasonable similarity scores
                distances = [result[1] for result in results]
                if all(d >= 1.0 for d in distances):
                    print("⚠️  All distances are >= 1.0, similarities will be 0.0")
                    return False
                else:
                    print("✅ Found reasonable similarity scores")
                    return True
            else:
                print("❌ No results returned")
                return False

        except Exception as e:
            print(f"❌ Semantic search failed: {e}")
            return False

    def run_repair(self):
        """Run the repair process."""
        print("\\n" + "=" * 60)
        print("SQLite-vec Zero Embedding Repair Tool")
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
                print("\\n✅ Database appears healthy, no repair needed")
                return

            # Load model
            self.load_model()

            # Fix zero embeddings
            fixed = self.regenerate_zero_embeddings(analysis)

            # Verify search works
            search_working = self.verify_search()

            # Re-analyze
            print("\\nRe-analyzing database after repair...")
            new_analysis = self.analyze_database()

            print("\\n" + "=" * 60)
            print("Repair Summary")
            print("=" * 60)
            print(f"Fixed {fixed} zero vector embeddings")
            print(f"Search working: {'✅ Yes' if search_working else '❌ No'}")

            if new_analysis["issues"]:
                print("\\n⚠️  Some issues remain:")
                for issue in new_analysis["issues"]:
                    print(f"  - {issue}")
            else:
                print("\\n✅ All issues resolved!")

        except Exception as e:
            print(f"\\n❌ Repair failed: {e}")
            logger.exception("Repair failed")

        finally:
            if self.conn:
                self.conn.close()


def main():
    """Run the repair tool."""
    if len(sys.argv) < 2:
        print("Usage: python repair_zero_embeddings.py <database_path>")
        print("\\nExample:")
        print("  python repair_zero_embeddings.py ~/.local/share/mcp-memory/sqlite_vec.db")
        print("\\nThis tool will:")
        print("  - Check for zero vector embeddings (invalid)")
        print("  - Regenerate proper embeddings using sentence-transformers")
        print("  - Verify semantic search functionality with similarity scores")
        sys.exit(1)

    db_path = sys.argv[1]

    repair = ZeroEmbeddingRepair(db_path)
    repair.run_repair()


if __name__ == "__main__":
    main()
