#!/usr/bin/env python3
"""
Diagnostic script to test SQLite-vec embedding functionality.

This script performs comprehensive tests to identify and diagnose issues
with the embedding pipeline in the MCP Memory Service.
"""

import asyncio
import logging
import os
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mcp_memory_service.models.memory import Memory
from src.mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from src.mcp_memory_service.utils.hashing import generate_content_hash

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EmbeddingDiagnostics:
    """Test suite for SQLite-vec embedding functionality."""

    def __init__(self, db_path=None):
        self.db_path = db_path or tempfile.mktemp(suffix=".db")
        self.storage = None
        self.test_results = []

    async def run_all_tests(self):
        """Run all diagnostic tests."""
        print("\n" + "=" * 60)
        print("SQLite-vec Embedding Diagnostics")
        print("=" * 60 + "\n")

        tests = [
            self.test_dependencies,
            self.test_storage_initialization,
            self.test_embedding_generation,
            self.test_memory_storage,
            self.test_semantic_search,
            self.test_database_integrity,
            self.test_edge_cases,
        ]

        for test in tests:
            try:
                await test()
            except Exception as e:
                self.log_error(f"{test.__name__} failed", e)

        self.print_summary()

    async def test_dependencies(self):
        """Test 1: Check required dependencies."""
        print("\n[TEST 1] Checking dependencies...")

        # Check sqlite-vec
        try:
            import sqlite_vec

            self.log_success("sqlite-vec is installed")
        except ImportError:
            self.log_error("sqlite-vec is NOT installed", "pip install sqlite-vec")

        # Check sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer

            self.log_success("sentence-transformers is installed")
        except ImportError:
            self.log_error("sentence-transformers is NOT installed", "pip install sentence-transformers")

        # Check torch
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.log_success(f"torch is installed (device: {device})")
        except ImportError:
            self.log_error("torch is NOT installed", "pip install torch")

    async def test_storage_initialization(self):
        """Test 2: Initialize storage backend."""
        print("\n[TEST 2] Initializing storage...")

        try:
            self.storage = SqliteVecMemoryStorage(self.db_path)
            await self.storage.initialize()
            self.log_success(f"Storage initialized at {self.db_path}")

            # Check embedding model
            if self.storage.embedding_model:
                self.log_success(f"Embedding model loaded: {self.storage.embedding_model_name}")
                self.log_success(f"Embedding dimension: {self.storage.embedding_dimension}")
            else:
                self.log_error("Embedding model NOT loaded", None)

        except Exception as e:
            self.log_error("Storage initialization failed", e)

    async def test_embedding_generation(self):
        """Test 3: Generate embeddings."""
        print("\n[TEST 3] Testing embedding generation...")

        if not self.storage:
            self.log_error("Storage not initialized", "Previous test failed")
            return

        test_texts = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is transforming how we process information",
            "SQLite is a lightweight embedded database",
        ]

        for text in test_texts:
            try:
                embedding = self.storage._generate_embedding(text)

                # Validate embedding
                if not embedding:
                    self.log_error(f"Empty embedding for: {text[:30]}...", None)
                elif len(embedding) != self.storage.embedding_dimension:
                    self.log_error(
                        f"Dimension mismatch for: {text[:30]}...",
                        f"Expected {self.storage.embedding_dimension}, got {len(embedding)}",
                    )
                else:
                    self.log_success(f"Generated embedding for: {text[:30]}... (dim: {len(embedding)})")

            except Exception as e:
                self.log_error(f"Embedding generation failed for: {text[:30]}...", e)

    async def test_memory_storage(self):
        """Test 4: Store memories with embeddings."""
        print("\n[TEST 4] Testing memory storage...")

        if not self.storage:
            self.log_error("Storage not initialized", "Previous test failed")
            return

        test_memories = [
            Memory(
                content="Python is a versatile programming language",
                content_hash=generate_content_hash("Python is a versatile programming language"),
                tags=["programming", "python"],
                memory_type="reference",
            ),
            Memory(
                content="The Eiffel Tower is located in Paris, France",
                content_hash=generate_content_hash("The Eiffel Tower is located in Paris, France"),
                tags=["geography", "landmarks"],
                memory_type="fact",
            ),
            Memory(
                content="Machine learning models can learn patterns from data",
                content_hash=generate_content_hash("Machine learning models can learn patterns from data"),
                tags=["ml", "ai"],
                memory_type="concept",
            ),
        ]

        stored_count = 0
        for memory in test_memories:
            try:
                success, message = await self.storage.store(memory)
                if success:
                    self.log_success(f"Stored: {memory.content[:40]}...")
                    stored_count += 1
                else:
                    self.log_error(f"Failed to store: {memory.content[:40]}...", message)
            except Exception as e:
                self.log_error(f"Storage exception for: {memory.content[:40]}...", e)

        print(f"\nStored {stored_count}/{len(test_memories)} memories successfully")

    async def test_semantic_search(self):
        """Test 5: Perform semantic search."""
        print("\n[TEST 5] Testing semantic search...")

        if not self.storage:
            self.log_error("Storage not initialized", "Previous test failed")
            return

        test_queries = [
            ("programming languages", 2),
            ("tourist attractions in Europe", 2),
            ("artificial intelligence and data", 2),
            ("random unrelated query xyz123", 1),
        ]

        for query, _expected_min in test_queries:
            try:
                results = await self.storage.retrieve(query, n_results=5)

                if not results:
                    self.log_error(f"No results for query: '{query}'", "Semantic search returned empty")
                else:
                    self.log_success(f"Found {len(results)} results for: '{query}'")

                    # Show top result
                    if results:
                        top_result = results[0]
                        print(f"  Top match: {top_result.memory.content[:50]}...")
                        print(f"  Relevance: {top_result.relevance_score:.3f}")

            except Exception as e:
                self.log_error(f"Search failed for: '{query}'", e)

    async def test_database_integrity(self):
        """Test 6: Check database integrity."""
        print("\n[TEST 6] Checking database integrity...")

        if not self.storage or not self.storage.conn:
            self.log_error("Storage not initialized", "Previous test failed")
            return

        try:
            # Check memory count
            cursor = self.storage.conn.execute("SELECT COUNT(*) FROM memories")
            memory_count = cursor.fetchone()[0]

            # Check embedding count
            cursor = self.storage.conn.execute("SELECT COUNT(*) FROM memory_embeddings")
            embedding_count = cursor.fetchone()[0]

            print(f"  Memories table: {memory_count} rows")
            print(f"  Embeddings table: {embedding_count} rows")

            if memory_count != embedding_count:
                self.log_error("Row count mismatch", f"Memories: {memory_count}, Embeddings: {embedding_count}")
            else:
                self.log_success("Database row counts match")

            # Check for orphaned embeddings
            cursor = self.storage.conn.execute("""
                SELECT COUNT(*) FROM memory_embeddings e
                WHERE NOT EXISTS (
                    SELECT 1 FROM memories m WHERE m.id = e.rowid
                )
            """)
            orphaned = cursor.fetchone()[0]

            if orphaned > 0:
                self.log_error("Found orphaned embeddings", f"Count: {orphaned}")
            else:
                self.log_success("No orphaned embeddings found")

        except Exception as e:
            self.log_error("Database integrity check failed", e)

    async def test_edge_cases(self):
        """Test 7: Edge cases and error handling."""
        print("\n[TEST 7] Testing edge cases...")

        if not self.storage:
            self.log_error("Storage not initialized", "Previous test failed")
            return

        # Test empty content
        try:
            empty_memory = Memory(content="", content_hash=generate_content_hash(""), tags=["empty"])
            success, message = await self.storage.store(empty_memory)
            if success:
                self.log_error("Stored empty content", "Should have failed")
            else:
                self.log_success("Correctly rejected empty content")
        except Exception as e:
            self.log_success(f"Correctly raised exception for empty content: {type(e).__name__}")

        # Test very long content
        try:
            long_content = "x" * 10000
            long_memory = Memory(content=long_content, content_hash=generate_content_hash(long_content), tags=["long"])
            success, message = await self.storage.store(long_memory)
            if success:
                self.log_success("Handled long content")
            else:
                self.log_error("Failed on long content", message)
        except Exception as e:
            self.log_error("Exception on long content", e)

    def log_success(self, message):
        """Log a successful test result."""
        print(f"  ✓ {message}")
        self.test_results.append(("SUCCESS", message))

    def log_error(self, message, error):
        """Log a failed test result."""
        print(f"  ✗ {message}")
        if error:
            if isinstance(error, Exception):
                print(f"    Error: {type(error).__name__}: {str(error)}")
            else:
                print(f"    Info: {error}")
        self.test_results.append(("ERROR", message, error))

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)

        success_count = sum(1 for r in self.test_results if r[0] == "SUCCESS")
        error_count = sum(1 for r in self.test_results if r[0] == "ERROR")

        print(f"\nTotal tests: {len(self.test_results)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {error_count}")

        if error_count > 0:
            print("\nFailed tests:")
            for result in self.test_results:
                if result[0] == "ERROR":
                    print(f"  - {result[1]}")

        print("\n" + "=" * 60)


async def main():
    """Run diagnostics."""
    # Check if a database path was provided
    db_path = sys.argv[1] if len(sys.argv) > 1 else None

    if db_path and not os.path.exists(db_path):
        print(f"Warning: Database file does not exist: {db_path}")
        print("Creating new database for testing...")

    diagnostics = EmbeddingDiagnostics(db_path)
    await diagnostics.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
